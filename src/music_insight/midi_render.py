"""把 MIDI 渲染成音频，用于在没有版权录音的情况下评测音频算法。

渲染出来的是干净的单声道合成钢琴：谐波 + 指数衰减包络，没有踏板与混响。
因此它的准确率会偏高，只能当“算法正确性测试”，不能代表真实录音表现。
"""

from __future__ import annotations

import numpy as np

SR_DEFAULT = 22050


def render_midi(path: str, sr: int = SR_DEFAULT, track_name: str | None = None) -> tuple[np.ndarray, float]:
    """返回 (音频, 时长秒)。track_name 非空时只渲染名字匹配的轨（如 'MELODY'）。"""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(path)
    duration = float(pm.get_end_time()) + 0.1
    n = int(duration * sr) + 1
    out = np.zeros(n, np.float32)

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        if track_name and track_name.lower() not in (inst.name or "").lower():
            continue
        for note in inst.notes:
            start = int(note.start * sr)
            if start >= n:
                continue
            end = min(n, int(note.end * sr))
            length = end - start
            if length <= 0:
                continue
            t = np.arange(length) / sr
            freq = 440.0 * 2 ** ((note.pitch - 69) / 12)
            env = np.exp(-t * 2.5) * (1 - np.exp(-t * 250))
            wave = (np.sin(2 * np.pi * freq * t)
                    + 0.35 * np.sin(4 * np.pi * freq * t)
                    + 0.15 * np.sin(6 * np.pi * freq * t))
            out[start:end] += (note.velocity / 127.0) * env * wave

    peak = np.max(np.abs(out)) or 1.0
    return (out / peak * 0.8).astype(np.float32), duration
