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


def test_report_never_claims_vocals(tmp_path):
    """回归：纯钢琴曲也曾被判成「人声频段占比 82%」。

    300~3000 Hz 能量占比只是亮度描述（18 首纯器乐样本实测与频谱重心 r=-0.89），
    报告里不能再出现旧字段名或「弱证据」式的人声暗示。
    """
    wav = tmp_path / "demo.wav"
    synth.synth_demo(str(wav), bpm=100.0, bars=4)
    samples, sr = audio.load_audio(wav)
    feat = features.extract(samples, sr)
    struct = structure.analyze(feat)
    result = report.build(feat, struct, str(wav))
    md = open(report.to_markdown(result, tmp_path / "report.md"), encoding="utf-8").read()

    assert "vocal_band_ratio" not in md
    assert "人声频段占比" not in md
    assert "中频带能量占比" in md
    assert "mid_band_ratio" in result["measurements"]
    assert "vocal_band_ratio" not in result["measurements"]


def test_mid_band_ratio_is_timbre_not_vocal():
    """中频带占比跟随亮度，不跟随“有没有人声”。

    两个样本都没有人声：中频密集的谐波堆占比很高，铺满全频带的白噪声很低。
    这就是为什么它不能当人声证据——纯钢琴曲能轻松超过 80%。
    """
    sr = 22050
    t = np.arange(sr * 2) / sr
    harmonic = sum(0.1 / (k + 1) * np.sin(2 * np.pi * 220 * (k + 1) * t) for k in range(8))
    noise = np.random.default_rng(0).normal(0.0, 0.05, sr * 2)

    dark = features.extract(harmonic.astype(np.float32), sr)["mid_band_ratio"]
    bright = features.extract(noise.astype(np.float32), sr)["mid_band_ratio"]

    assert dark > 0.55          # 纯谐波、无人声，照样过半
    assert bright < 0.4         # 全频带噪声拉低占比
    assert dark > bright        # 它度量的是亮度分布
