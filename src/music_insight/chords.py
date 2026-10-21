"""和弦识别：HPCP + 三和弦/七和弦模板 + Viterbi 平滑。

只做测量，不做命名美化：输出的就是模板匹配结果 + 每段置信度。
"""

from __future__ import annotations

import numpy as np

from .dsp import PITCH_NAMES, stft
from .keydetect import HPCP_FFT, HPCP_HOP, _hpcp, estimate_tuning
from .salience import salience_chroma

# (quality 后缀, 音程, 权重)：根音最强，五音次之，三音再次
QUALITIES: list[tuple[str, tuple[int, ...], tuple[float, ...], str]] = [
    ("", (0, 4, 7), (1.0, 0.9, 0.8), "maj"),
    ("m", (0, 3, 7), (1.0, 0.9, 0.8), "min"),
    ("7", (0, 4, 7, 10), (1.0, 0.8, 0.7, 0.6), "dom7"),
    ("m7", (0, 3, 7, 10), (1.0, 0.8, 0.7, 0.6), "min7"),
    ("maj7", (0, 4, 7, 11), (1.0, 0.8, 0.7, 0.6), "maj7"),
    ("dim", (0, 3, 6), (1.0, 0.9, 0.8), "dim"),
]


def _root_pc(name: str) -> int:
    """'C#m7' → 1。"""
    root = name[:2] if len(name) > 1 and name[1] == "#" else name[:1]
    return PITCH_NAMES.index(root)


# 和弦性质先验：POP909 第 401–500 首的“持续时间加权”统计，
# 与评测用的前 20 首不相交。归一化到本模块的 6 种性质。
QUALITY_PRIORS = {"": 0.604, "m": 0.356, "7": 0.006, "m7": 0.018,
                  "maj7": 0.010, "dim": 0.006}


def _quality_of(name: str) -> str:
    """'C#m7' → 'm7'；'C' → ''。"""
    root_len = 2 if len(name) > 1 and name[1] == "#" else 1
    return name[root_len:]


def _quality_log_prior(names: list[str]) -> np.ndarray:
    return np.log(np.array([QUALITY_PRIORS.get(_quality_of(n), 1e-4) for n in names]))


def build_templates() -> tuple[list[str], np.ndarray]:
    names, rows = [], []
    for root in range(12):
        for suffix, intervals, weights, _ in QUALITIES:
            vec = np.zeros(12)
            for iv, w in zip(intervals, weights):
                vec[(root + iv) % 12] = w
            names.append(f"{PITCH_NAMES[root]}{suffix}")
            rows.append(vec / np.linalg.norm(vec))
    return names, np.array(rows)


def _viterbi(scores: np.ndarray, switch_penalty: float) -> np.ndarray:
    """scores: (n_states, n_frames) → 每帧状态索引。"""
    n_states, n_frames = scores.shape
    best = np.zeros((n_frames, n_states))
    back = np.zeros((n_frames, n_states), int)
    best[0] = scores[:, 0]
    for t in range(1, n_frames):
        prev = best[t - 1]
        stay = prev
        jump_from = int(np.argmax(prev))
        jump = prev[jump_from] - switch_penalty
        use_stay = stay >= jump
        back[t] = np.where(use_stay, np.arange(n_states), jump_from)
        best[t] = scores[:, t] + np.where(use_stay, stay, jump)
    path = [int(np.argmax(best[-1]))]
    for t in range(n_frames - 1, 0, -1):
        path.append(int(back[t][path[-1]]))
    return np.array(path[::-1])


def detect_beat_sync(x: np.ndarray, sr: int, beat_times: np.ndarray,
                     switch_penalty: float = 0.5,
                     smooth_frames: int = 5,
                     prior_weight: float = 0.3,
                     allowed: list[str] | None = None,
                     chroma_mode: str = "hpcp",
                     chroma_kwargs: dict | None = None) -> dict:
    """按拍聚合的和弦识别（beat-synchronous）。

    逐帧 chroma 会被琶音、经过音打散；POP909 的钢琴编曲尤其如此。
    先把每拍内的 chroma 取平均，再做 Viterbi，并对发射分数按拍标准化
    （不同曲目的余弦分数尺度不同，不标准化会让转移惩罚失效）。

    beat_times 需要单调递增，最后可以多给一个结束时间。
    """
    S = stft(x, n_fft=HPCP_FFT, hop=HPCP_HOP)
    if chroma_mode == "salience":
        hpcp, _, tuning = salience_chroma(
            x, sr, n_fft=HPCP_FFT * 2, hop=HPCP_HOP, **(chroma_kwargs or {}))
    else:
        tuning = estimate_tuning(S, sr, HPCP_FFT)
        hpcp = _hpcp(S, sr, tuning, HPCP_FFT)

    fps = sr / HPCP_HOP
    feats = []
    for a, b in zip(beat_times[:-1], beat_times[1:]):
        i0 = min(int(a * fps), hpcp.shape[1] - 1)
        i1 = min(max(i0 + 1, int(b * fps)), hpcp.shape[1])
        seg = hpcp[:, i0:i1]
        f = seg.mean(axis=1) if seg.size else np.zeros(12)
        feats.append(f / (np.linalg.norm(f) + 1e-9))
    feats = np.array(feats)

    names, templates = build_templates()
    if allowed:
        keep = [i for i, n in enumerate(names) if n in set(allowed)]
        if keep:
            names = [names[i] for i in keep]
            templates = templates[keep]

    scores = templates @ feats.T
    scores = (scores - scores.mean(axis=0, keepdims=True)) / (
        scores.std(axis=0, keepdims=True) + 1e-9)
    if prior_weight:
        scores = scores + prior_weight * _quality_log_prior(names)[:, None]
    path = _viterbi(scores, switch_penalty)

    centers = (beat_times[:-1] + beat_times[1:]) / 2
    labels = [names[i] for i in path]
    segments = []
    for i, lab in enumerate(labels):
        if segments and segments[-1]["chord"] == lab:
            segments[-1]["end"] = round(float(beat_times[i + 1]), 2)
        else:
            segments.append({"start": round(float(beat_times[i]), 2),
                             "end": round(float(beat_times[i + 1]), 2), "chord": lab})
    return {
        "tuning_cents": tuning,
        "beat_times": beat_times,
        "beat_centers": centers,
        "beat_labels": labels,
        "segments": segments,
        "n_segments": len(segments),
    }


def detect(x: np.ndarray, sr: int, min_duration: float = 0.4,
           switch_penalty: float = 0.50, bass_bonus: float = 0.0,
           bass_range: tuple[float, float] = (70.0, 300.0),
           smooth_frames: int = 5,
           allowed: list[str] | None = None,
           chroma_mode: str = "hpcp",
           chroma_kwargs: dict | None = None) -> dict:
    """返回逐帧标注与合并后的和弦段。

    bass_bonus > 0 时，用低频（默认 70–300 Hz）音级给“根音”加分，
    用来抑制转位/低音进行造成的根音误判。
    smooth_frames > 0 时对 HPCP 做时间中值滤波，抑制琶音与短音造成的抖动。
    allowed 非空时只在给定的和弦集合里选（用于调内约束实验）。
    """
    S = stft(x, n_fft=HPCP_FFT, hop=HPCP_HOP)
    if chroma_mode == "salience":
        hpcp, _, tuning = salience_chroma(
            x, sr, n_fft=HPCP_FFT * 2, hop=HPCP_HOP, **(chroma_kwargs or {}))
    else:
        tuning = estimate_tuning(S, sr, HPCP_FFT)
        hpcp = _hpcp(S, sr, tuning, HPCP_FFT)

    if smooth_frames > 1:
        k = smooth_frames
        pad = k // 2
        padded = np.pad(hpcp, ((0, 0), (pad, pad)), mode="edge")
        hpcp = np.stack([np.median(padded[:, i:i + k], axis=1)
                         for i in range(hpcp.shape[1])], axis=1)
        hpcp = hpcp / (np.linalg.norm(hpcp, axis=0, keepdims=True) + 1e-9)

    names, templates = build_templates()
    if allowed:
        keep = [i for i, n in enumerate(names) if n in set(allowed)]
        if keep:
            names = [names[i] for i in keep]
            templates = templates[keep]
    scores = templates @ hpcp                      # (n_chords, n_frames)

    if bass_bonus > 0:
        bins = np.fft.rfftfreq(HPCP_FFT, 1 / sr)
        low = (bins >= bass_range[0]) & (bins <= bass_range[1])
        midi = 69 + 12 * np.log2(np.maximum(bins[low], 1e-9) / 440.0) + tuning / 100.0
        pc = np.mod(np.round(midi).astype(int), 12)
        bass = np.zeros((12, S.shape[1]), np.float32)
        for k in range(12):
            sel = pc == k
            if sel.any():
                bass[k] = S[low][sel].mean(axis=0)
        bass /= bass.sum(axis=0, keepdims=True) + 1e-9
        roots = np.array([_root_pc(n) for n in names])
        scores = scores + bass_bonus * bass[roots]

    path = _viterbi(scores, switch_penalty)

    fps = sr / HPCP_HOP
    hop_sec = 1.0 / fps
    best_score = scores[path, np.arange(len(path))]

    # 合并成段
    segments = []
    start = 0
    for i in range(1, len(path) + 1):
        if i == len(path) or path[i] != path[start]:
            segments.append({
                "start": round(start * hop_sec, 2),
                "end": round(i * hop_sec, 2),
                "chord": names[path[start]],
                "score": round(float(best_score[start:i].mean()), 3),
            })
            start = i

    # 吞掉过短的段：并入右边（末尾则并入左边）
    merged = []
    for seg in segments:
        if seg["end"] - seg["start"] < min_duration and merged:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))
    if len(merged) > 1 and merged[-1]["end"] - merged[-1]["start"] < min_duration:
        merged[-2]["end"] = merged[-1]["end"]
        merged.pop()

    frame_labels = np.array([names[i] for i in path])
    return {
        "tuning_cents": tuning,
        "hop_sec": hop_sec,
        "frame_labels": frame_labels,
        "frame_scores": best_score,
        "segments": merged,
        "n_segments": len(merged),
    }
