"""音频解码：优先 PyAV（mp3/m4a/flac 等），WAV 走标准库兜底。"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def _decode_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())

    if width == 2:
        x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    elif width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        x = np.frombuffer(raw, np.int32).astype(np.float32) / 2**31
    else:
        raise ValueError(f"unsupported WAV sample width: {width}")

    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return x, sr


def _decode_av(path: str) -> tuple[np.ndarray, int]:
    try:
        import av  # type: ignore
    except ImportError as exc:  # pragma: no cover - 取决于安装方式
        raise RuntimeError(
            "PyAV is required to decode non-WAV audio: pip install av"
        ) from exc

    with av.open(path) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError(f"no audio stream in {path}")
        sr = stream.rate
        chunks = []
        for frame in container.decode(stream):
            arr = frame.to_ndarray()
            if arr.ndim == 2:
                arr = arr.mean(axis=0) if arr.shape[0] <= 8 else arr.mean(axis=1)
            chunks.append(arr.astype(np.float32).reshape(-1))

    audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    if audio.size and np.max(np.abs(audio)) > 2:  # 整数 PCM 形态
        audio = audio / 32768.0
    return audio, sr


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """读取音频为单声道 float32 与采样率。"""
    p = str(path)
    if p.lower().endswith(".wav"):
        try:
            return _decode_wav(p)
        except Exception:
            pass  # 非标准 WAV 时退回到 PyAV
    return _decode_av(p)


def resample(audio: np.ndarray, sr: int, target: int = 16000) -> np.ndarray:
    """线性插值重采样。音乐分析用 44.1k 即可，语音识别降到 16k。"""
    if sr == target or audio.size == 0:
        return audio.astype(np.float32)
    n = max(1, int(len(audio) * target / sr))
    return np.interp(
        np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio
    ).astype(np.float32)
