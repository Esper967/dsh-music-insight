"""生成一条「有弧线」的测试音轨：空气感前奏 → 主体 → 抽空尾奏。

比 ``make_demo.py`` 多一个用途：**验证感官层的骨架判断**。
那段 19 秒的演示音是平的（响度差 < 1 dB），vibe 层只会说「平铺」；
这条 36 秒的轨道有明确的起—满—收，适合看 `report.md` 里「它怎么走的」那张表。

用法::

    python examples/make_arc_demo.py out/arc_demo.wav
"""

from __future__ import annotations

import sys
import wave

import numpy as np

SR = 22050


def _hiss(seconds: float, level: float, rng: np.random.Generator) -> np.ndarray:
    """高通噪声：模拟空气感 / 水汽，而不是乐器。"""
    n = int(SR * seconds)
    x = rng.normal(0.0, 1.0, n).astype(np.float32)
    x = np.diff(x, prepend=0.0)
    return (level * x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)


def _pad(seconds: float, freqs: tuple[float, ...], level: float) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    x = sum(np.sin(2 * np.pi * f * t) for f in freqs).astype(np.float32)
    return (level * x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)


def synth_arc_demo(path: str) -> float:
    rng = np.random.default_rng(7)
    parts = [
        _hiss(10.0, 0.06, rng),                       # 前奏：只有空气
        _pad(4.0, (261.63, 329.63, 392.00), 0.45),    # C
        _pad(4.0, (349.23, 440.00, 523.25), 0.45),    # F
        _pad(4.0, (392.00, 493.88, 587.33), 0.45),    # G
        _pad(4.0, (220.00, 261.63, 329.63), 0.45),    # Am
        _pad(4.0, (261.63, 329.63, 392.00), 0.45),    # C
    ]
    tail = _hiss(6.0, 0.03, rng)
    fade = np.linspace(1.0, 0.0, tail.size, dtype=np.float32) ** 2
    parts.append(tail * fade)                         # 尾奏：抽空

    x = np.concatenate(parts)
    x = 0.5 * x / (np.max(np.abs(x)) + 1e-9)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())
    return len(x) / SR


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "out/arc_demo.wav"
    seconds = synth_arc_demo(out)
    print(f"wrote {out} ({seconds:.1f}s)")
