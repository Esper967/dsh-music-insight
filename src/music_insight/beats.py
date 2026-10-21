"""节拍跟踪：起音包络 + 自相关定速 + 动态规划定相位（Ellis 式简化版）。

只用 numpy。目的是把“和弦识别依赖真值拍点”这个作弊条件去掉。
"""

from __future__ import annotations

import numpy as np

from .dsp import stft


def onset_envelope(x: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512) -> tuple[np.ndarray, float]:
    """半波整流谱通量，返回 (包络, 帧率)。"""
    S = stft(x, n_fft=n_fft, hop=hop)
    logmag = np.log1p(S * 100.0)
    flux = np.diff(logmag, axis=1).clip(0, None).sum(axis=0)
    # 去掉直流并归一化
    flux = flux - flux.min()
    peak = flux.max()
    if peak > 0:
        flux = flux / peak
    fps = sr / hop
    return flux.astype(np.float32), fps


def estimate_tempo(flux: np.ndarray, fps: float,
                   lo_bpm: float = 50.0, hi_bpm: float = 180.0) -> float:
    f = flux - flux.mean()
    ac = np.correlate(f, f, mode="full")[len(f) - 1:]
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


def track_beats(x: np.ndarray, sr: int, bpm: float | None = None,
                tightness: float = 100.0) -> np.ndarray:
    """返回拍点时间（秒）。Ellis 2007 的动态规划简化版。"""
    flux, fps = onset_envelope(x, sr)
    if bpm is None:
        bpm = estimate_tempo(flux, fps)
    if bpm <= 0:
        return np.array([])

    period = 60.0 * fps / bpm
    n = len(flux)
    local = flux.copy()
    back = np.full(n, -1, dtype=int)
    window_back = max(1, int(round(2 * period)))
    window_fwd = max(1, int(round(period / 2)))

    for i in range(n):
        lo = max(0, i - window_back)
        hi = i - window_fwd
        if hi <= lo:
            continue
        idx = np.arange(lo, hi)
        intervals = (i - idx).astype(float)
        penalty = -tightness * (np.log(intervals / period) ** 2)
        scores = local[idx] + penalty
        j = int(np.argmax(scores))
        best = scores[j]
        if best > 0:
            local[i] = flux[i] + best
            back[i] = int(idx[j])

    # 从末尾一个周期内得分最高处回溯
    tail_start = max(0, n - int(round(period)))
    tail_idx = tail_start + int(np.argmax(local[tail_start:]))
    beats = []
    i = tail_idx
    while i >= 0:
        beats.append(i)
        i = back[i]
    beats.reverse()
    if len(beats) < 3:
        return np.array(beats, dtype=float) / fps

    times = np.array(beats, dtype=float) / fps
    times = _refine_phase(flux, fps, times)
    return times


def _refine_phase(flux: np.ndarray, fps: float, beats: np.ndarray) -> np.ndarray:
    """在起音包络上做相位微调：把拍点整体平移，使包络上的能量最大。

    DP 回溯出的起点常常落在半拍或一小节之外，不修相位的话 F-measure 会接近 0。
    """
    if len(beats) < 3:
        return beats
    period = float(np.median(np.diff(beats)))
    if period <= 0:
        return beats
    start, end = 0.0, len(flux) / fps
    grid_times = np.arange(len(flux)) / fps

    best_shift, best_score = 0.0, -1.0
    for shift in np.arange(0.0, period, 0.01):
        times = np.arange(shift, end, period)
        if len(times) < 3:
            continue
        # 只统计落在两拍之间的能量（避免把整首曲子都算进去）
        score = float(np.interp(times, grid_times, flux).mean())
        if score > best_score:
            best_shift, best_score = float(shift), score
    return np.arange(best_shift, end, period)
