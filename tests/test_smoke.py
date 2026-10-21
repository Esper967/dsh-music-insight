from __future__ import annotations

from pathlib import Path

from music_insight import audio, features, report, structure, synth


def test_synth_then_measure(tmp_path: Path) -> None:
    wav = tmp_path / "demo.wav"
    duration = synth.synth_demo(str(wav), bpm=100.0, bars=4)
    assert duration > 5

    samples, sr = audio.load_audio(wav)
    assert samples.size > 0 and sr == 44100

    feat = features.extract(samples, sr)
    assert abs(feat["bpm"] - 100.0) < 6.0, feat["bpm"]
    assert feat["key_correlation"] > 0.3
    assert feat["duration_sec"] > 5


def test_structure_and_report(tmp_path: Path) -> None:
    wav = tmp_path / "demo.wav"
    synth.synth_demo(str(wav), bpm=100.0, bars=4)
    samples, sr = audio.load_audio(wav)

    feat = features.extract(samples, sr)
    struct = structure.analyze(feat)
    assert struct["segments"]
    assert struct["timeline"]

    result = report.build(feat, struct, str(wav))
    assert result["mood_priors"]["method"] == "heuristic-v0"
    assert result["mood_priors"]["confidence"] == "low"

    md = Path(report.to_markdown(result, tmp_path / "report.md")).read_text(encoding="utf-8")
    assert "BPM" in md
    assert "已知限制" in md

    js = Path(report.to_json(result, tmp_path / "analysis.json")).read_text(encoding="utf-8")
    assert '"mood_priors"' in js


def test_cli_smoke(tmp_path: Path) -> None:
    from music_insight.cli import main

    wav = tmp_path / "demo.wav"
    synth.synth_demo(str(wav), bpm=100.0, bars=2)
    out = tmp_path / "out"
    code = main([str(wav), "--out", str(out), "--no-images", "--quiet"])
    assert code == 0
    assert (out / "report.md").exists()
    assert (out / "analysis.json").exists()
    assert (out / "timeline.csv").exists()
