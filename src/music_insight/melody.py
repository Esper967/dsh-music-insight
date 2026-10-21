"""旋律提取：谐波显著度 + 动态规划的主音高跟踪。

输出的是逐帧 F0（MIDI 浮点），不强行量化成音符——先让“旋律走向”这部分可用。
"""

from __future__ import annotations

import numpy as np

from .dsp import stft
from .salience import harmonic_salience

FMIN_MIDI = 50      # D3
FMAX_MIDI = 91      # G6
N_FFT = 8192
HOP = 1024          # 约 46 ms（22050 Hz）


def track_melody(x: np.ndarray, sr: int, jump_penalty: float = 0.25,
                 fmin_midi: int = FMIN_MIDI, fmax_midi: int = FMAX_MIDI) -> dict:
    """返回逐帧 (时间, F0(midi), 显著度)。无声帧 F0=0。"""
    pitches, salience, tuning = harmonic_salience(
        x, sr, n_fft=N_FFT, hop=HOP, fmin_midi=fmin_midi, fmax_midi=fmax_midi,
        n_harmonics=5)

    # 每帧归一化，抑制整体响度差异
    s = salience / (salience.max(axis=0, keepdims=True) + 1e-9)
    n_pitch, n_frames = s.shape

    # Viterbi：最大化 Σ(显著度) - jump_penalty * |Δpitch|
    score = s[:, 0].astype(np.float64)
    back = np.zeros((n_frames, n_pitch), dtype=np.int16)
    pitch_axis = np.arange(n_pitch)
    for t in range(1, n_frames):
        prev = score
        best_prev = int(np.argmax(prev))
        # 用“上一帧最优 + 跳进惩罚”近似所有转移（O(n_pitch) 而非 O(n^2)）
        jump = prev[best_prev] - jump_penalty * np.abs(pitch_axis - best_prev)
        stay = prev
        use_stay = stay >= jump
        back[t] = np.where(use_stay, pitch_axis + 0, best_prev)
        score = s[:, t] + np.where(use_stay, stay, jump)

    path = np.zeros(n_frames, dtype=int)
    path[-1] = int(np.argmax(score))
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = back[t][path[t]]

    f0 = pitches[path].astype(float)
    conf = s[path, np.arange(n_frames)]
    # 低显著度帧判为无声
    f0[conf < 0.15] = 0.0
    times = np.arange(n_frames) * HOP / sr
    return {"times": times, "f0_midi": f0, "confidence": conf, "tuning_cents": tuning}


def midi_to_hz(m: np.ndarray) -> np.ndarray:
    return 440.0 * 2 ** ((m - 69) / 12)


def track_melody_from_file(path: str) -> dict:
    from . import audio

    samples, sr = audio.load_audio(path)
    return track_melody(samples, sr)
