"""命令行入口：音频 → 证据包（图 + JSON + CSV + Markdown）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import audio as audio_mod
from . import beats as beats_mod
from . import chords as chords_mod
from . import features, keydetect, render, report, structure, vibe


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="music-insight",
        description="把音频转成 LLM 可读的证据包：测量指标 + 图 + 事实性报告。",
    )
    p.add_argument("input", help="音频文件（mp3/m4a/flac/wav…）")
    p.add_argument("--out", default="out", help="输出目录，默认 ./out")
    p.add_argument("--no-images", action="store_true", help="不生成图")
    p.add_argument("--lyrics", action="store_true", help="尝试转写歌词（需要 faster-whisper）")
    p.add_argument("--chords", action="store_true",
                   help="节拍跟踪 + 和弦识别，并由和弦反推调性（纯本地）")
    p.add_argument("--mode", choices=("vibe", "strict"), default="vibe",
                   help="报告风格：vibe=先说听感、数字压后（默认，给人听）；"
                        "strict=可引用的证据清单（给评估/核对）")
    p.add_argument("--quiet", action="store_true", help="只输出文件路径")
    p.add_argument("--version", action="version", version="music-insight 0.1.0")
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows 控制台默认 GBK，中文会乱码
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

    args = build_parser().parse_args(argv)
    src = Path(args.input)
    if not src.exists():
        print(f"error: {src} not found", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, sr = audio_mod.load_audio(src)
    if samples.size == 0:
        print("error: decoded audio is empty", file=sys.stderr)
        return 3

    feat = features.extract(samples, sr)
    struct = structure.analyze(feat)

    image_path = None
    if not args.no_images:
        image_path = render.figure(feat, struct, samples, out_dir / "panel.png")

    result = report.build(feat, struct, str(src), image_path)

    if args.chords:
        beat_times = beats_mod.track_beats(samples, sr)
        if len(beat_times) >= 4:
            beat_times = np.append(beat_times, samples.size / sr)
            sync = chords_mod.detect_beat_sync(samples, sr, beat_times)
            durations = [(s["chord"], s["end"] - s["start"]) for s in sync["segments"]]
            result["chords"] = {
                "n_segments": sync["n_segments"],
                "segments": sync["segments"],
                "beat_count": len(beat_times) - 1,
            }
            result["key_from_chords"] = keydetect.key_from_chords(durations)
        else:
            result["chords"] = {"error": "beat tracking failed"}
        # 和弦会影响「和弦主要绕着谁转」这句听感锚点，重算一次感官层。
        result["vibe"] = vibe.build(result)

    if args.lyrics:
        try:
            from .lyrics import transcribe_lyrics
            result["lyrics"] = transcribe_lyrics(str(src))
        except Exception as exc:  # noqa: BLE001 - 可选功能失败不应中断主流程
            result["lyrics"] = {"available": False, "reason": str(exc)}

    written = [
        report.to_json(result, out_dir / "analysis.json"),
        report.to_timeline_csv(result, out_dir / "timeline.csv"),
        report.to_markdown(result, out_dir / "report.md", mode=args.mode),
        report.to_limits(out_dir / "LIMITS.md"),
    ]
    if image_path:
        written.append(image_path)

    if args.quiet:
        for path in written:
            print(path)
    else:
        m = result["measurements"]
        print(f"music-insight: {src.name}")
        print(f"  {m['duration_sec']}s · {m['bpm']} BPM · {m['key']} "
              f"· {len(result['structure'])} 段 · 平均 {m['loudness_db']['mean']} dB")
        print(f"  mode: {args.mode}")
        print(f"  {result['vibe']['one_liner']}")
        print("  written:")
        for path in written:
            print(f"    {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
