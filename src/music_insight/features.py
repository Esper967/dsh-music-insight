"""测量层：只用 numpy 的时频分析与客观指标。

这里每个数字都是确定性算法算出来的，可以被基准测试检验。
凡是需要“猜”的东西（和弦命名、情绪）都不放在这个文件里。
"""

from __future__ import annotations

import numpy as np

from .dsp import HOP, N_FFT, PITCH_NAMES, stft  # noqa: F401 - 供其他模块复用
from . import keydetect

N_MELS = 64
FMIN, FMAX = 30.0, 8000.0

# Krumhansl-Kessler 调性轮廓（近似值，仅用于相对相关比较）
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def mel_filterbank(sr: int, n_fft: int = N_FFT, n_mels: int = N_MELS,
                   fmin: float = FMIN, fmax: float = FMAX) -> np.ndarray:
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    fmax = min(fmax, sr / 2 - 1)
    edges = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, len(bins)), np.float32)
    for i in range(n_mels):
        lo, ce, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (bins - lo) / max(ce - lo, 1e-9)
        down = (hi - bins) / max(hi - ce, 1e-9)
        fb[i] = np.clip(np.minimum(up, down), 0, None)
    return fb


def mel_spectrogram(S: np.ndarray, fb: np.ndarray) -> np.ndarray:
    return np.log1p((fb @ (S ** 2)) * 1e4)


def chroma(S: np.ndarray, sr: int, n_fft: int = N_FFT,
           lo: float = 55.0, hi: float = 4200.0) -> np.ndarray:
    """音高类能量，(12, frames)，每帧归一化。"""
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    midi = 69 + 12 * np.log2(np.maximum(bins, 1e-9) / 440.0)
    pc = np.mod(np.round(midi).astype(int), 12)
    valid = (bins > lo) & (bins < hi)
    out = np.zeros((12, S.shape[1]), np.float32)
    for k in range(12):
        sel = valid & (pc == k)
        if sel.any():
            out[k] = S[sel].sum(axis=0)
    return out / (np.linalg.norm(out, axis=0, keepdims=True) + 1e-9)


def spectral_centroid(S: np.ndarray, sr: int, n_fft: int = N_FFT) -> np.ndarray:
    """频谱重心（Hz）：越大越“亮”。"""
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    return (bins[:, None] * S).sum(axis=0) / (S.sum(axis=0) + 1e-9)


def onset_flux(logmel: np.ndarray) -> np.ndarray:
    """半波整流谱通量，作为起音强度曲线。"""
    diff = np.diff(logmel, axis=1).clip(0, None)
    return diff.sum(axis=0)


def tempo(flux: np.ndarray, sr: int, hop: int = HOP,
          lo_bpm: float = 50.0, hi_bpm: float = 180.0) -> float:
    """自相关估计 BPM，并折到 70~160 区间以消除倍频歧义。"""
    if flux.size < 8:
        return 0.0
    f = flux - flux.mean()
    ac = np.correlate(f, f, mode="full")[len(f) - 1:]
    fps = sr / hop
    lo = max(1, int(fps * 60 / hi_bpm))
    hi = min(len(ac) - 1, int(fps * 60 / lo_bpm))
    if hi <= lo:
        return 0.0
    lag = lo + int(np.argmax(ac[lo:hi]))
    bpm = 60.0 * fps / lag
    while 0 < bpm < 70:
        bpm *= 2
    while bpm > 160:
        bpm /= 2
    return float(bpm)


def key_candidates(mean_chroma: np.ndarray, top: int = 3) -> list[tuple[str, float]]:
    """Krumhansl 相关法的前 N 个调性候选，按相关度降序。

    候选间隔（top1 - top2）很小时说明这个估计不可靠——尤其是 F 大调与 A 小调
    这种共享 6 个音级的调，chroma 方法本来就会来回摇。
    """
    scores = []
    for root in range(12):
        for profile, mode in ((_MAJOR, "大调"), (_MINOR, "小调")):
            c = float(np.corrcoef(np.roll(profile, root), mean_chroma)[0, 1])
            scores.append((round(c, 3), f"{PITCH_NAMES[root]} {mode}"))
    scores.sort(reverse=True)
    return [(name, score) for score, name in scores[:top]]


def key(mean_chroma: np.ndarray) -> tuple[str, float]:
    """取第一候选，返回 ("A 小调", 相关系数)。"""
    name, score = key_candidates(mean_chroma, 1)[0]
    return name, score


def loudness_curve(x: np.ndarray, hop: int = HOP, n_fft: int = N_FFT) -> np.ndarray:
    """逐帧 RMS，单位 dBFS。"""
    out = []
    for i in range(0, max(1, len(x) - n_fft), hop):
        seg = x[i:i + n_fft]
        out.append(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9) if seg.size else -99.0)
    return np.array(out, np.float32)


def band_energy(S: np.ndarray, sr: int, lo: float, hi: float,
                n_fft: int = N_FFT) -> np.ndarray:
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    sel = (bins >= lo) & (bins < hi)
    return np.log1p(S[sel].mean(axis=0) * 1e3) if sel.any() else np.zeros(S.shape[1], np.float32)


def vocal_band_ratio(S: np.ndarray, sr: int, n_fft: int = N_FFT) -> float:
    """300~3000 Hz 能量占比。人声主要落在这里，但乐器也在这里——只能当弱证据。"""
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    vocal = S[(bins >= 300) & (bins < 3000)].sum()
    total = S.sum() + 1e-9
    return float(vocal / total)


def extract(x: np.ndarray, sr: int) -> dict:
    """一次算完所有测量量，返回纯 numpy / 原生类型的字典。"""
    S = stft(x)
    fb = mel_filterbank(sr)
    logmel = mel_spectrogram(S, fb)
    ch = chroma(S, sr)
    cent = spectral_centroid(S, sr)
    flux = onset_flux(logmel)
    mean_chroma = ch.mean(axis=1)
    key_info = keydetect.detect(x, sr)
    loud = loudness_curve(x)
    rms_dbfs = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)

    return {
        "sample_rate": sr,
        "duration_sec": round(len(x) / sr, 2),
        "bpm": round(tempo(flux, sr), 1),
        "key": key_info["key"],
        "key_correlation": key_info["key_score"],
        "key_margin": key_info["key_margin"],
        "key_candidates": key_info["key_candidates"],
        "key_votes": key_info["key_votes"],
        "key_agreement": key_info["key_agreement"],
        "mode_agreement": key_info["mode_agreement"],
        "tuning_cents": key_info["tuning_cents"],
        "loudness_db": {
            "mean": round(float(rms_dbfs), 1),
            "min": round(float(loud.min()), 1),
            "max": round(float(loud.max()), 1),
        },
        "crest_db": round(float(rms_dbfs - 20 * np.log10(np.max(np.abs(x)) + 1e-9)), 1),
        "note_density_per_sec": round(float((flux > flux.std()).sum() / max(len(x) / sr, 1e-9)), 2),
        "vocal_band_ratio": round(vocal_band_ratio(S, sr), 3),
        "mean_chroma_top3": [PITCH_NAMES[i] for i in np.argsort(mean_chroma)[::-1][:3]],
        "_S": S, "_logmel": logmel, "_chroma": ch, "_centroid": cent,
        "_flux": flux, "_loud": loud,
    }
