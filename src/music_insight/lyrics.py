"""可选的人声/歌词转写。

只在装了 faster-whisper 时可用；失败一律返回结构化原因，不抛到主流程。
注意：纯音乐上 whisper 会产生幻觉（输出 IPA 或无关语言），所以默认关闭，
且结果里必须带上 no_speech 概率，让解读层自己判断可信度。
"""

from __future__ import annotations

import numpy as np

from . import audio as audio_mod

MAX_SECONDS = 600


def transcribe_lyrics(path: str, model_size: str = "small") -> dict:
    try:
        import av  # noqa: F401
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("需要 faster-whisper 与 av：pip install faster-whisper av") from exc

    samples, sr = audio_mod.load_audio(path)
    samples = samples[: int(MAX_SECONDS * sr)]
    samples = audio_mod.resample(samples, sr, 16000)

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(samples, language="zh", beam_size=5, vad_filter=False)

    lines = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        lines.append({
            "start": round(seg.start, 1),
            "end": round(seg.end, 1),
            "text": text,
            "no_speech_prob": round(float(seg.no_speech_prob), 2),
        })

    reliable = [ln for ln in lines if ln["no_speech_prob"] < 0.5]
    return {
        "available": True,
        "model": model_size,
        "language": info.language,
        "language_probability": round(float(info.language_probability), 2),
        "segments": lines,
        "likely_instrumental": len(reliable) == 0,
        "caveat": "no_speech_prob 高说明该段可能没有人声，不要据此编歌词。",
    }
