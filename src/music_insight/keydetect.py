"""调性检测：多信息源投票（Krumhansl / Temperley / 贝斯根音）。

设计原则：不把任何一个算法当“真值”。三个信息源各自独立投票，
一致时置信度高，分歧时明确标注不可靠。

HPCP（谐波音级轮廓）+ 全局调音估计用于改善裸 chroma 的低频分辨率与音准偏差。
"""

from __future__ import annotations

import numpy as np

from .dsp import N_FFT, PITCH_NAMES, stft

# 两套公开的 key profile（Krumhansl-Kessler 1982；Temperley 2001 revised）
KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
TEMPERLEY_MAJOR = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
TEMPERLEY_MINOR = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])

HPCP_FFT = 8192
HPCP_HOP = 4096
HPCP_LOW, HPCP_HIGH = 100.0, 5000.0


def _hpcp(S: np.ndarray, sr: int, tuning_cents: float, n_fft: int) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    keep = (freqs >= HPCP_LOW) & (freqs <= HPCP_HIGH)
    freqs = freqs[keep]
    mag = S[keep]

    midi = 69 + 12 * np.log2(freqs / 440.0) + tuning_cents / 100.0
    frac = midi - np.round(midi)
    weight = np.exp(-0.5 * (frac / 0.15) ** 2)          # 离半音中心越远权重越低
    pc = np.mod(np.round(midi).astype(int), 12)

    out = np.zeros((12, mag.shape[1]), np.float32)
    for k in range(12):
        sel = pc == k
        if sel.any():
            out[k] = (mag[sel] * weight[sel, None]).sum(axis=0)
    return out / (np.linalg.norm(out, axis=0, keepdims=True) + 1e-9)


def estimate_tuning(S: np.ndarray, sr: int, n_fft: int) -> float:
    """全局调音偏移（音分）。歌曲常偏离 A440 十几音分，不校正会摊薄音级能量。"""
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    keep = (freqs >= HPCP_LOW) & (freqs <= HPCP_HIGH)
    mag = S[keep].mean(axis=1)
    freqs = freqs[keep]
    midi = 69 + 12 * np.log2(np.maximum(freqs, 1e-9) / 440.0)
    frac = midi - np.round(midi)
    best, best_score = 0.0, -1.0
    for cents in range(-50, 51, 5):
        f = frac - cents / 100.0
        f = f - np.round(f)
        score = float((mag * np.exp(-0.5 * (f / 0.15) ** 2)).sum())
        if score > best_score:
            best, best_score = float(cents), score
    return best


def _profiles() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "krumhansl": (KRUMHANSL_MAJOR, KRUMHANSL_MINOR),
        "temperley": (TEMPERLEY_MAJOR, TEMPERLEY_MINOR),
    }


def _rank(profile_pairs, feature: np.ndarray, top: int = 3) -> list[tuple[str, str, float]]:
    """返回 [(key, mode, score)] 降序。"""
    scores = []
    for root in range(12):
        for pair in profile_pairs:
            major, minor = pair
            for prof, mode in ((major, "maj"), (minor, "min")):
                c = float(np.corrcoef(np.roll(prof, root), feature)[0, 1])
                scores.append((PITCH_NAMES[root], mode, c))
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:top]


def ending_tonic(x: np.ndarray, sr: int, tuning_cents: float) -> dict:
    """用结尾段的和弦判断调式：作品通常结束在主和弦上。

    这是专治“关系大小调混淆”的信源——两者共享音级，靠结尾三和弦的大/小三度区分。
    """
    tail = x[int(len(x) * 0.9):] if len(x) > sr * 10 else x[-int(sr * 5):]
    if tail.size < 2048:
        tail = x[-2048:]
    S = stft(tail, n_fft=HPCP_FFT, hop=HPCP_HOP)
    hpcp = _hpcp(S, sr, tuning_cents, HPCP_FFT).mean(axis=1)
    hpcp = hpcp / (np.linalg.norm(hpcp) + 1e-9)

    ranked = []
    for root in range(12):
        for mode, intervals, weights in (("maj", (0, 4, 7), (1.0, 0.9, 0.8)),
                                         ("min", (0, 3, 7), (1.0, 0.9, 0.8))):
            score = sum(float(hpcp[(root + iv) % 12]) * w for iv, w in zip(intervals, weights))
            ranked.append((PITCH_NAMES[root], mode, score))
    ranked.sort(key=lambda v: v[2], reverse=True)
    return {
        "source": "ending_tonic",
        "key": f"{ranked[0][0]} {'大调' if ranked[0][1] == 'maj' else '小调'}",
        "score": round(ranked[0][2], 3),
        "candidates": [f"{r[0]} {'大调' if r[1] == 'maj' else '小调'} {r[2]:.3f}" for r in ranked[:3]],
    }


# 大调/小调的调内三和弦（音级 → 性质）
MAJOR_STEPS = {0: "maj", 2: "min", 4: "min", 5: "maj", 7: "maj", 9: "min", 11: "dim"}
MINOR_STEPS = {0: "min", 2: "dim", 3: "maj", 5: "min", 7: "min", 8: "maj", 10: "maj"}


def _chord_root_quality(name: str) -> tuple[int, str] | None:
    root = name[:2] if len(name) > 1 and name[1] == "#" else name[:1]
    if root not in PITCH_NAMES:
        return None
    qual = name[len(root):]
    if qual in ("", "maj7"):
        cat = "maj"
    elif qual in ("m", "m7"):
        cat = "min"
    elif qual == "dim":
        cat = "dim"
    else:
        cat = "other"
    return PITCH_NAMES.index(root), cat


def _relative_of(key: str) -> str:
    """'A# 大调' ↔ 'G 小调'（关系大小调，共享全部 7 个调内和弦）。"""
    name, mode = key.split()
    if name not in PITCH_NAMES:
        return ""
    pc = PITCH_NAMES.index(name)
    if mode == "大调":
        return f"{PITCH_NAMES[(pc + 9) % 12]} 小调"
    return f"{PITCH_NAMES[(pc + 3) % 12]} 大调"


def key_from_chords(chords: list[tuple[str, float]]) -> dict:
    """用已识别的和弦序列（和弦名, 持续秒数）反推调性。

    比纯音级统计可靠：大三/小三和弦直接携带调式信息，
    能绕开“关系大小调共享音级”的死结的一半——但**只限一半**：
    关系大小调共享全部调内和弦，仅凭和弦序列无法区分，所以输出两个候选并标注歧义。
    """
    durations: dict[str, float] = {}
    for name, dur in chords:
        durations[name] = durations.get(name, 0.0) + max(dur, 0.0)

    scored = []
    for tonic in range(12):
        for mode, steps in (("大调", MAJOR_STEPS), ("小调", MINOR_STEPS)):
            expected = {(tonic + k) % 12: v for k, v in steps.items()}
            score = 0.0
            for name, dur in durations.items():
                rq = _chord_root_quality(name)
                if rq is None:
                    continue
                root, cat = rq
                if expected.get(root) == cat:
                    score += dur * (2.0 if root == tonic else 1.0)
            scored.append((score, f"{PITCH_NAMES[tonic]} {mode}"))
    scored.sort(reverse=True)

    total = sum(d for d in durations.values()) + 1e-9
    best_key, best_score = scored[0][1], scored[0][0]
    runner, runner_score = scored[1][1], scored[1][0]

    # 关系大小调共享全部调内和弦：只要它的得分接近最优，就必须并列报出来
    relative = _relative_of(best_key)
    relative_score = next((s for s, k in scored if k == relative), 0.0)
    ambiguous = bool(relative) and relative_score >= 0.6 * best_score
    return {
        "key": best_key,
        "score": round(best_score / total, 3),
        "runner_up": runner,
        "runner_score": round(runner_score / total, 3),
        "margin": round((best_score - runner_score) / total, 3),
        "ambiguous_relative": ambiguous,
        "relative_pair": [best_key, relative] if ambiguous else [],
        "top_chords": [f"{n} {d:.1f}s" for n, d in
                       sorted(durations.items(), key=lambda kv: -kv[1])[:6]],
    }


def detect(x: np.ndarray, sr: int) -> dict:
    """多源调性估计。返回每个源的投票 + 一致性。"""
    S = stft(x, n_fft=HPCP_FFT, hop=HPCP_HOP)
    tuning = estimate_tuning(S, sr, HPCP_FFT)
    hpcp = _hpcp(S, sr, tuning, HPCP_FFT)
    profile_mean = hpcp.mean(axis=1)

    # 源 1/2：两套 profile 对同一个 HPCP
    votes = []
    primary = None
    for name, pair in _profiles().items():
        ranking = _rank([pair], profile_mean, top=3)
        if primary is None:
            primary = ranking
        votes.append({
            "source": name,
            "key": f"{ranking[0][0]} {'大调' if ranking[0][1] == 'maj' else '小调'}",
            "score": round(ranking[0][2], 3),
            "candidates": [f"{r[0]} {'大调' if r[1] == 'maj' else '小调'} {r[2]:.3f}"
                           for r in ranking],
            "margin": round(ranking[0][2] - ranking[1][2], 3),
        })

    # 源 3：贝斯/低音根音（不同信息源，只统计 40~250 Hz）
    bins = np.fft.rfftfreq(HPCP_FFT, 1 / sr)
    low = (bins >= 40) & (bins <= 250)
    low_chroma = np.zeros(12, np.float32)
    midi_low = 69 + 12 * np.log2(np.maximum(bins[low], 1e-9) / 440.0) + tuning / 100.0
    pc_low = np.mod(np.round(midi_low).astype(int), 12)
    energy_low = S[low].mean(axis=1)
    for k in range(12):
        sel = pc_low == k
        if sel.any():
            low_chroma[k] = energy_low[sel].sum()
    low_chroma /= low_chroma.sum() + 1e-9

    bass_scores = []
    for root in range(12):
        for mode, triad in (("maj", (0, 4, 7)), ("min", (0, 3, 7))):
            s = sum(float(low_chroma[(root + iv) % 12]) for iv in triad)
            bass_scores.append((PITCH_NAMES[root], mode, s))
    bass_scores.sort(key=lambda v: v[2], reverse=True)
    votes.append({
        "source": "bass_root",
        "key": f"{bass_scores[0][0]} {'大调' if bass_scores[0][1] == 'maj' else '小调'}",
        "score": round(bass_scores[0][2], 3),
        "candidates": [f"{r[0]} {'大调' if r[1] == 'maj' else '小调'} {r[2]:.3f}"
                       for r in bass_scores[:3]],
    })

    votes.append(ending_tonic(x, sr, tuning))

    names = [v["key"] for v in votes]
    agree = max(names.count(n) for n in set(names)) / len(names)
    modes = [v["key"].split()[-1] for v in votes]
    mode_agree = max(modes.count(m) for m in set(modes)) / len(modes)

    return {
        "key": names[0],
        "key_score": votes[0]["score"],
        "key_margin": votes[0]["margin"],
        "key_candidates": [{"key": f"{r[0]} {'大调' if r[1] == 'maj' else '小调'}", "corr": r[2]}
                           for r in primary],
        "key_votes": votes,
        "key_agreement": round(agree, 2),
        "mode_agreement": round(mode_agree, 2),
        "tuning_cents": tuning,
    }
