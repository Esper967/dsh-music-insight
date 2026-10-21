"""合成一段可复现的测试音乐（不依赖任何外部音频文件，避免版权问题）。"""

from __future__ import annotations

import wave

import numpy as np


def synth_demo(path: str, sr: int = 44100, bpm: float = 100.0, bars: int = 8) -> float:
    """I–V–vi–IV 进行的电子风格循环，约 19 秒。返回时长（秒）。"""
    beat = 60.0 / bpm
    bar = 4 * beat
    duration = bars * bar
    n = int(duration * sr)
    out = np.zeros(n, np.float32)

    def tone(freqs, start, length, amp=0.18):
        i0 = int(start * sr)
        i1 = min(n, i0 + int(length * sr))
        if i1 <= i0:
            return
        t = np.arange(i1 - i0) / sr
        seg = np.zeros(i1 - i0, np.float32)
        for f in freqs:
            seg += amp / len(freqs) * np.sin(2 * np.pi * f * t)
        env = np.minimum(1.0, np.minimum(t / 0.02, (t[-1] - t) / 0.15)).clip(0, 1)
        out[i0:i1] += seg * env

    def kick(start, amp=0.7):
        i0 = int(start * sr)
        i1 = min(n, i0 + int(0.25 * sr))
        if i1 <= i0:
            return
        t = np.arange(i1 - i0) / sr
        freq = 120 * np.exp(-t * 18) + 45
        out[i0:i1] += amp * np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-t * 12)

    def noise(start, length, amp):
        i0 = int(start * sr)
        i1 = min(n, i0 + int(length * sr))
        if i1 <= i0:
            return
        t = np.arange(i1 - i0) / sr
        rng = np.random.RandomState(int(start * 1000) % 2**31)
        out[i0:i1] += amp * rng.randn(i1 - i0).astype(np.float32) * np.exp(-t / (length * 0.35))

    c_major = [261.63, 329.63, 392.00]
    g_major = [196.00, 246.94, 293.66]
    a_minor = [220.00, 261.63, 329.63]
    f_major = [174.61, 220.00, 261.63]
    progression = [c_major, g_major, a_minor, f_major]

    for b in range(bars):
        chord = progression[(b // 2) % 4]
        for k in range(4):
            start = b * bar + k * beat
            if k in (0, 2):
                kick(start)
            else:
                noise(start, 0.18, 0.22)
            noise(start, 0.05, 0.10)
            tone(chord, start, beat * 0.95)

    peak = np.max(np.abs(out)) or 1.0
    pcm = (out / peak * 0.85 * 32767).astype(np.int16)
    with wave.open(path, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(pcm.tobytes())
    return duration
