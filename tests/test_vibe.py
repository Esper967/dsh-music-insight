"""感官层（vibe v2）与报告模式的测试。

重点不是「词写得好不好看」，而是：
- 骨架分类在已知形状上不出错，v2 输出确实短；
- 元信息能从 tag / 文件名两条路拿到；
- vibe / strict 两种报告各自成立；
- **全程离线**：结果里不该再出现网络听众层或性格因子。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
            "mid_band_ratio": 0.68,
            "mean_chroma_top3": ["C", "A", "G"],
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
        "mood_priors": {"arousal": 0.54, "valence": 0.64, "labels": "高唤醒 / 偏正面"},
    }


def _demo_result(tmp_path) -> dict:
    wav = tmp_path / "demo.wav"
    synth.synth_demo(str(wav), bpm=100.0, bars=4)
    samples, sr = audio.load_audio(wav)
    feat = features.extract(samples, sr)
    struct = structure.analyze(feat)
    return report.build(feat, struct, str(wav))


# --- 感官层 ---------------------------------------------------------------


def test_shape_detects_rain_arc_and_stays_short():
    v = vibe.build(_shaped_result())
    assert "从空气里出来" in v["shape"]
    assert "雨停" in v["headline"]
    assert len(v["feelings"]) == 3
    assert "→" in v["arc_line"]
    assert "BPM" in v["hand_line"]
    assert len(v["headline"]) <= 60
    assert v["method"] == "vibe-v2"


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
    result = _demo_result(tmp_path)
    v = result["vibe"]
    assert v["method"] == "vibe-v2"
    assert v["headline"]
    assert len(v["feelings"]) == 3
    assert v["arc_line"]
    assert v["hand_line"]
    assert result["meta"]["title"] == "demo"


def test_result_has_no_network_or_persona_layers(tmp_path):
    """性格因子和网络听众层已卸载：结果里不该再有这两个键。"""
    result = _demo_result(tmp_path)
    assert "voice" not in result
    assert "audience" not in result


# --- 报告 -----------------------------------------------------------------


def test_report_modes_differ(tmp_path):
    result = _demo_result(tmp_path)

    fun = Path(report.to_markdown(result, tmp_path / "vibe.md", mode="vibe")).read_text(encoding="utf-8")
    strict = Path(report.to_markdown(result, tmp_path / "strict.md", mode="strict")).read_text(encoding="utf-8")

    assert fun.startswith("# 听完《")
    assert "## 感觉" in fun and "## 走向" in fun and "## 数据" in fun
    assert "LIMITS.md" in fun
    assert "| 时间 |" not in fun
    assert "听感锚点" not in fun
    assert "别人听完说什么" not in fun
    assert "说话方式" not in fun
    assert len(fun.splitlines()) < 60

    assert strict.startswith("# 音乐测量报告")
    assert "每段的和弦见第 4 节" in strict
    assert "证据索引" in strict
    assert "已知限制" in strict
    assert "网络听众层" not in strict


def test_vibe_prompt_is_short_and_offline():
    from music_insight import report as report_mod

    prompt = report_mod.LLM_PROMPT_FUN
    assert "2–4 句" in prompt
    assert "100 字" in prompt
    assert "网易云" not in prompt and "豆瓣" not in prompt
    # 防止互联网语料污染：标题不可信 + 不许搜索
    assert "不可信" in prompt
    assert "不要搜索" in prompt
    # 专业面：允许但要求克制地带数据
    assert "具体数据" in prompt
    assert "最多用 2 个数据点" in prompt


def test_vibe_report_shows_chords_when_available(tmp_path):
    result = _demo_result(tmp_path)
    result["key_from_chords"] = {
        "key": "C 大调", "runner_up": "A 小调", "margin": 0.1,
        "ambiguous_relative": True, "relative_pair": ["C 大调", "A 小调"],
        "top_chords": ["C 40.0s", "F 20.0s", "G 15.0s", "Am 10.0s"],
    }
    md = Path(report.to_markdown(result, tmp_path / "vibe.md", mode="vibe")).read_text(encoding="utf-8")
    assert "和弦：C 40.0s / F 20.0s / G 15.0s / Am 10.0s" in md
    assert "C 大调 / A 小调 二选一" in md


def test_limits_file_written(tmp_path):
    out = tmp_path / "LIMITS.md"
    report.to_limits(out)
    text = out.read_text(encoding="utf-8")
    assert "失效条件" in text
    assert "人声" in text
    assert "关系大小调" in text
    assert "全程离线" in text
    assert "网络听众层" not in text


def test_unknown_mode_rejected(tmp_path):
    with pytest.raises(ValueError):
        report.to_markdown(_shaped_result(), tmp_path / "x.md", mode="nope")


def test_cli_outputs_offline_only(tmp_path):
    from music_insight.cli import main

    wav = tmp_path / "demo.wav"
    synth.synth_demo(str(wav), bpm=100.0, bars=2)
    out = tmp_path / "out"
    assert main([str(wav), "--out", str(out), "--no-images", "--quiet"]) == 0
    data = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    assert "audience" not in data
    assert "voice" not in data
