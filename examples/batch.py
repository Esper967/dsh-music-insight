"""批量分析一个目录下的音频，输出汇总表（CSV + Markdown）。

用法：
    python examples/batch.py "路径/到/音乐文件夹" --out out/batch
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from music_insight import audio, features as F, structure  # noqa: E402

AUDIO_EXT = {".mp3", ".m4a", ".flac", ".wav", ".ogg", ".aac", ".wma"}

COLUMNS = ["file", "duration_sec", "bpm", "key", "key_agree", "mode_agree", "tuning",
           "key_corr", "loud_db", "crest_db", "brightness_hz", "segments", "top_chroma",
           "flip_05s", "note_density"]


def chroma_flip_rate(feat: dict, window_sec: float = 0.5) -> float:
    """0.5 秒块主音级的变化率。低 = 和声稳定（可能和弦简单）；高 = 旋律主导或和声复杂。"""
    top = np.argmax(feat["_chroma"], axis=0)
    fps = feat["sample_rate"] / F.HOP
    w = max(1, int(window_sec * fps))
    blocks = [int(np.bincount(top[i:i + w], minlength=12).argmax())
              for i in range(0, max(1, len(top) - w), w)]
    if len(blocks) < 2:
        return 0.0
    changes = sum(1 for a, b in zip(blocks, blocks[1:]) if a != b)
    return round(changes / (len(blocks) - 1), 3)


def analyze_file(path: Path) -> dict:
    samples, sr = audio.load_audio(path)
    feat = F.extract(samples, sr)
    struct = structure.analyze(feat)
    brightness = float(np.mean([t["brightness_hz"] for t in struct["timeline"]])) if struct["timeline"] else 0.0
    return {
        "file": path.name,
        "duration_sec": feat["duration_sec"],
        "bpm": feat["bpm"],
        "key": feat["key"],
        "key_agree": feat["key_agreement"],
        "mode_agree": feat["mode_agreement"],
        "tuning": feat["tuning_cents"],
        "key_corr": feat["key_correlation"],
        "loud_db": feat["loudness_db"]["mean"],
        "crest_db": feat["crest_db"],
        "brightness_hz": round(brightness),
        "segments": len(struct["segments"]),
        "top_chroma": "/".join(feat["mean_chroma_top3"]),
        "flip_05s": chroma_flip_rate(feat),
        "note_density": feat["note_density_per_sec"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--out", default="out/batch")
    args = parser.parse_args()

    folder = Path(args.folder)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXT)
    if not files:
        print(f"no audio files in {folder}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, path in enumerate(files, 1):
        try:
            row = analyze_file(path)
        except Exception as exc:  # noqa: BLE001 - 单个文件失败不应中断整批
            print(f"[{i}/{len(files)}] FAILED {path.name}: {exc}", file=sys.stderr)
            continue
        rows.append(row)
        print(f"[{i}/{len(files)}] {path.name}: {row['bpm']} BPM · {row['key']} "
              f"· {row['duration_sec']}s · flip {row['flip_05s']}")

    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    md = ["| 文件 | 时长 | BPM | 调性 | 响度 | 明暗 Hz | 段数 | 主音级 | chroma抖动 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['file'][:34]} | {r['duration_sec']:.0f}s | {r['bpm']} | {r['key']} "
                  f"| {r['loud_db']} | {r['brightness_hz']} | {r['segments']} "
                  f"| {r['top_chroma']} | {r['flip_05s']} |")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\nwrote {csv_path} and {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
