"""回归测试：报告里几个曾经给出误导数字的地方。"""

from __future__ import annotations

import numpy as np

from music_insight import audio, features, report, structure, synth


def test_loudness_floor_clamps_silence():
    sr = 22050
    tone = 0.1 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    x = np.concatenate([np.zeros(sr, np.float32), tone])
    feat = features.extract(x, sr)
    assert feat["loudness_db"]["min"] >= -90.0
    assert feat["loudness_db"]["min"] >= features.LOUDNESS_FLOOR_DB - 0.01


def test_onset_rate_ignores_steady_tone():
    sr = 22050
    tone = 0.05 * np.sin(2 * np.pi * 440 * np.arange(sr * 5) / sr).astype(np.float32)
    feat = features.extract(tone, sr)
    # 稳态正弦没有起音：旧实现（均值/标准差阈值）会数出 20+ 次/秒
    assert feat["note_density_per_sec"] < 0.5


def test_onset_rate_detects_synth_beats(tmp_path):
    wav = tmp_path / "demo.wav"
    synth.synth_demo(str(wav), bpm=100.0, bars=8)
    samples, sr = audio.load_audio(wav)
    feat = features.extract(samples, sr)
    assert feat["note_density_per_sec"] > 0.5


def test_mood_priors_flag_saturation():
    extreme = {
        "bpm": 220,
        "loudness_db": {"mean": -3.0},
        "note_density_per_sec": 30.0,
        "brightness_mean_hz": 9000.0,
        "key": "C 大调",
    }
    result = report.mood_priors(extreme)
    assert result["saturated"] is True
    assert result["confidence"] == "invalid"
    assert "失效" in result["labels"]


def test_mood_priors_normal_range_not_saturated():
    normal = {
        "bpm": 90,
        "loudness_db": {"mean": -15.0},
        "note_density_per_sec": 5.0,
        "brightness_mean_hz": 2000.0,
        "key": "C 大调",
    }
    result = report.mood_priors(normal)
    assert result["saturated"] is False
    assert result["confidence"] == "low"


def test_report_drops_legacy_chord_columns(tmp_path):
    wav = tmp_path / "demo.wav"
    synth.synth_demo(str(wav), bpm=100.0, bars=4)
    samples, sr = audio.load_audio(wav)
    feat = features.extract(samples, sr)
    struct = structure.analyze(feat)
    result = report.build(feat, struct, str(wav))
    md = open(report.to_markdown(result, tmp_path / "report.md"), encoding="utf-8").read()

    assert "和弦猜测" not in md          # 旧的结构表列已移除
    assert "2 秒块" not in md            # 旧的已知限制文案已移除
    assert "每段的和弦见第 4 节" in md
