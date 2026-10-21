"""音高显著性：谐波求和（harmonic summation），用来替代简单的音级直方图。

为什么需要它：简单 HPCP 把 100–5000 Hz 的所有能量按音级累加，
一个 C 音的 2/3/4/5 次谐波（C/G/C/E）会伪装成 C 大三和弦。
本模块只给“基频候选”记分：对每个半音频率 f0，检查 f0、2f0、3f0… 处是否有能量，
并按谐波序号衰减。单音的 E 处没有自己的 2E/3E，因此不会被误认成和弦音。
"""

from __future__ import annotations

import numpy as np

from .dsp import stft

N_FFT = 16384
HOP = 4096
FMIN_MIDI = 36
FMAX_MIDI = 90          # C2–F#6
N_HARMONICS = 6
HARMONIC_WEIGHTS = (1.0, 0.6, 0.4, 0.3, 0.2, 0.15)


def estimate_tuning(S: np.ndarray, sr: int, n_fft: int) -> float:
    """全局调音偏移（音分），用能量在音高网格上的集中度做网格搜索。"""
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    mag = S.mean(axis=1)
    keep = (freqs >= 60) & (freqs <= 3000)
    midi = 69 + 12 * np.log2(np.maximum(freqs[keep], 1e-9) / 440.0)
    frac = midi - np.round(midi)
    m = mag[keep]
    best, best_score = 0.0, -1.0
    for cents in range(-50, 51, 5):
        f = frac - cents / 100.0
        f = f - np.round(f)
        score = float((m * np.exp(-0.5 * (f / 0.12) ** 2)).sum())
        if score > best_score:
            best, best_score = float(cents), score
    return best


def _gather(spectrum: np.ndarray, bin_idx: np.ndarray) -> np.ndarray:
    """在频谱上按浮点 bin 位置做线性插值取值，越界补 0。

    spectrum: (n_bins, n_frames)；bin_idx: (n_pitches,)
    返回 (n_pitches, n_frames)。
    """
    n_bins = spectrum.shape[0]
    i0 = np.floor(bin_idx).astype(int)
    frac = (bin_idx - i0)[:, None]
    i0 = np.clip(i0, 0, n_bins - 1)
    i1 = np.clip(i0 + 1, 0, n_bins - 1)
    return spectrum[i0] * (1 - frac) + spectrum[i1] * frac


def harmonic_salience(x: np.ndarray, sr: int, n_fft: int = N_FFT, hop: int = HOP,
                      fmin_midi: int = FMIN_MIDI, fmax_midi: int = FMAX_MIDI,
                      n_harmonics: int = N_HARMONICS,
                      weights: tuple[float, ...] | None = None,
                      tuning_cents: float | None = None,
                      whiten: bool = False,
                      inharmonic_tol: float = 0.0) -> tuple[np.ndarray, np.ndarray, float]:
    """返回 (midi 列表, 显著度矩阵 (n_pitches, n_frames), 使用的调音偏移)。

    whiten: 每帧按频谱中位数做白化，压低踏板造成的宽带拖尾。
    inharmonic_tol: 在目标频率的 ±tol 比例范围内取最大值，容忍钢琴的非谐性。
    """
    S = stft(x, n_fft=n_fft, hop=hop)
    if tuning_cents is None:
        tuning_cents = estimate_tuning(S, sr, n_fft)

    if whiten:
        S = S / (np.median(S, axis=0, keepdims=True) + 1e-9)

    w = weights or HARMONIC_WEIGHTS[:n_harmonics]
    bin_per_hz = n_fft / sr
    pitches = np.arange(fmin_midi, fmax_midi + 1)
    f0 = 440.0 * 2 ** ((pitches - 69) / 12) * 2 ** (tuning_cents / 1200.0)

    salience = np.zeros((len(pitches), S.shape[1]), np.float32)
    nyquist = sr / 2.0
    ratios = [1.0] if inharmonic_tol <= 0 else [1.0 - inharmonic_tol, 1.0, 1.0 + inharmonic_tol]
    for h in range(1, len(w) + 1):
        target = f0 * h
        valid = target < nyquist * 0.98
        if not valid.any():
            continue
        best = None
        for ratio in ratios:
            idx = target[valid] * ratio * bin_per_hz
            val = _gather(S, idx)
            best = val if best is None else np.maximum(best, val)
        salience[valid] += w[h - 1] * best
    return pitches, salience, tuning_cents


def salience_chroma(x: np.ndarray, sr: int, octave_decay: float = 0.85,
                    normalize: bool = True, **kwargs) -> tuple[np.ndarray, np.ndarray, float]:
    """把显著度折叠成 12 维音级轮廓，(12, n_frames)，每帧归一化。

    低音区权重略高（和弦根音多在低音），octave_decay<1 时每升一个八度衰减。
    """
    pitches, salience, tuning = harmonic_salience(x, sr, **kwargs)
    chroma = np.zeros((12, salience.shape[1]), np.float32)
    ref = int(np.median(pitches))
    for i, midi in enumerate(pitches):
        weight = octave_decay ** abs((midi - ref) / 12.0)
        chroma[midi % 12] += salience[i] * weight
    if normalize:
        chroma /= np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9
    return chroma, pitches, tuning
