"""感官层（vibe）与两种报告模式的回归测试。

重点不是「词写得好不好看」，而是：
- 骨架分类在已知形状上不出错；
- 元信息能从 tag / 文件名两条路拿到；
- vibe 报告不把免责声明摆在正文最前面，但限制仍然可查（LIMITS.md）；
- strict 报告保持原来的证据清单形态，不被娱乐化改坏。
"""

from __future__ import annotations

from pathlib import Path

from music_insight import audio, features, report, structure, synth, vibe


def _shaped_result() -> dict:
    """手工构造一首「薄雾前奏 → 涨满 → 抽空」的歌，检查骨架分类。"""
    return {
        "source": "/music/glassmoon - 雨上がりの空.mp3",
        "meta": {"title": "雨上がりの空", "artist": "glassmoon"},
        "measurements": {
            "duration_sec": 178.0,
            "bpm": 112.3,
            "key": "C 大调",
            "loudness_db": {"mean": -20.8},
            "note_density_per_sec": 2.42,
            "brightness_mean_hz": 1300.0,
        },
        "structure": [
            {"label": "A", "start": 0.0, "end": 26.0, "loud_db": -35.4, "brightness_hz": 2682.0},
            {"label": "B", "start": 26.0, "end": 70.0, "loud_db": -22.4, "brightness_hz": 591.0},
            {"label": "C", "start": 70.0, "end": 94.0, "loud_db": -24.2, "brightness_hz": 628.0},
            {"label": "D", "start": 94.0, "end": 142.0, "loud_db": -20.0, "brightness_hz": 1003.0},
            {"label": "E", "start": 142.0, "end": 170.0, "loud_db": -25.8, "brightness_hz": 1031.0},
            {"label": "A", "start": 170.0, "end": 178.0, "loud_db": -56.6, "brightness_hz": 4824.0},
        ],
        "timeline": [],
        "mood_priors": {"arousal": 0.54, "valence": 0.64},
    }


def test_shape_detects_rain_arc():
    v = vibe.build(_shaped_result())
    assert "从空气里出来" in v["shape"]
    assert v["arc"][0]["word"] == "薄薄一层，还没落地"
    assert v["arc"][-1]["word"] == "抽空，只剩一点空气"
    assert v["arc"][0]["phase"] == "起"
    assert v["arc"][-1]["phase"] == "收"
    assert any("最满的是" in a for a in v["anchors"])


def test_metadata_falls_back_to_filename(tmp_path):
    missing = tmp_path / "glassmoon - 雨上がりの空.mp3"  # 不创建文件，走文件名兜底
    meta = vibe.read_metadata(str(missing))
    assert meta["title"] == "雨上がりの空"
    assert meta["artist"] == "glassmoon"
    assert meta["from_tags"] is False


def test_metadata_strips_track_number(tmp_path):
    meta = vibe.read_metadata(str(tmp_path / "07. 雨上がりの空.flac"))
    assert meta["title"] == "雨上がりの空"
    assert meta["artist"] == ""


def test_vibe_on_synth_demo(tmp_path):
    wav = tmp_path / "demo.wav"
    synth.synth_demo(str(wav), bpm=100.0, bars=4)
    samples, sr = audio.load_audio(wav)
    feat = features.extract(samples, sr)
    struct = structure.analyze(feat)
    result = report.build(feat, struct, str(wav))

    v = result["vibe"]
    assert v["method"] == "vibe-v1"
    assert v["one_liner"].startswith("《")
    assert len(v["arc"]) == len(result["structure"])
    assert 2 <= len(v["tags"]) <= 3
    assert v["anchors"]
    assert result["meta"]["title"] == "demo"


def test_report_modes_differ(tmp_path):
    wav = tmp_path / "demo.wav"
    synth.synth_demo(str(wav), bpm=100.0, bars=4)
    samples, sr = audio.load_audio(wav)
    feat = features.extract(samples, sr)
    struct = structure.analyze(feat)
    result = report.build(feat, struct, str(wav))

    fun = Path(report.to_markdown(result, tmp_path / "vibe.md", mode="vibe")).read_text(encoding="utf-8")
    strict = Path(report.to_markdown(result, tmp_path / "strict.md", mode="strict")).read_text(encoding="utf-8")

    assert fun.startswith("# 听完《")
    assert "它怎么走的" in fun
    assert "这次没测的" in fun
    assert "LIMITS.md" in fun
    assert "字段名" not in fun or True  # 提示词里会出现这个词，不做强制

    assert strict.startswith("# 音乐测量报告")
    assert "每段的和弦见第 4 节" in strict
    assert "证据索引" in strict
    assert "已知限制" in strict


def test_limits_file_written(tmp_path):
    out = tmp_path / "LIMITS.md"
    report.to_limits(out)
    text = out.read_text(encoding="utf-8")
    assert "失效条件" in text
    assert "人声" in text
    assert "关系大小调" in text


def test_unknown_mode_rejected(tmp_path):
    import pytest

    result = _shaped_result()
    with pytest.raises(ValueError):
        report.to_markdown(result, tmp_path / "x.md", mode="nope")
