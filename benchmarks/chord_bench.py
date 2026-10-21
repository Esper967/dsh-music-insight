"""和弦识别基准：POP909 的 MIDI 真值 vs 渲染音频上的识别结果。

流程：index.mid → 合成音频 → chords.detect() → 与 chord_midi.txt 逐帧比对。
注意：渲染音频是干净的合成钢琴，准确率会偏高，只能当算法正确性测试。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from music_insight import chords, midi_render  # noqa: E402
from music_insight.dsp import PITCH_NAMES  # noqa: E402

ENHARMONIC = {"Cb": "B", "C#": "C#", "Db": "C#", "D#": "D#", "Eb": "D#", "E#": "F",
              "Fb": "E", "F#": "F#", "Gb": "F#", "G#": "G#", "Ab": "G#", "A#": "A#",
              "Bb": "A#", "B#": "C"}
QUALITY_MAP = {"maj": "", "min": "m", "maj7": "maj7", "min7": "m7", "7": "7",
               "dim": "dim", "dim7": "dim", "hdim7": "dim", "aug": "aug",
               "sus2": "sus", "sus4": "sus", "6": "6", "m6": "m6", "9": "9"}
KNOWN_QUALITIES = {"", "m", "7", "m7", "maj7", "dim"}


def split_chord(name: str) -> tuple[str, str]:
    """'C#m7' → ('C#', 'm7')；'C' → ('C', '')。根音最多两个字符。"""
    if len(name) >= 2 and name[1] in "#b":
        return name[:2], name[2:]
    return name[:1], name[1:]


def load_truth(path: Path) -> list[tuple[float, float, str | None]]:
    segs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        start, end, name = float(parts[0]), float(parts[1]), parts[2]
        if name == "N":
            segs.append((start, end, None))
            continue
        root, qual = name.split(":") if ":" in name else (name, "maj")
        root = ENHARMONIC.get(root, root)
        if root not in PITCH_NAMES:
            continue
        segs.append((start, end, f"{root}{QUALITY_MAP.get(qual, qual)}"))
    return segs


def truth_at(times: np.ndarray, truth) -> list[str | None]:
    out: list[str | None] = []
    idx = 0
    for t in times:
        while idx + 1 < len(truth) and truth[idx][1] <= t:
            idx += 1
        start, end, name = truth[idx]
        out.append(name if start <= t < end else None)
    return out


def evaluate_song(song_dir: Path) -> dict | None:
    midi_path = song_dir / f"{song_dir.name}.mid"
    truth_path = song_dir / "chord_midi.txt"
    if not midi_path.exists() or not truth_path.exists():
        return None

    audio, _ = midi_render.render_midi(str(midi_path))
    result = chords.detect(audio, midi_render.SR_DEFAULT)
    hop = result["hop_sec"]
    pred = result["frame_labels"]
    times = np.arange(len(pred)) * hop

    truth = load_truth(truth_path)
    ref = truth_at(times, truth)

    root_hit = root_total = full_hit = full_total = 0
    for p, r in zip(pred, ref):
        if r is None:
            continue
        root_total += 1
        p_root, p_qual = split_chord(p)
        r_root, r_qual = split_chord(r)
        if p_root == r_root:
            root_hit += 1
        # 只比较双方都在词表里的和弦性质
        if p_qual in KNOWN_QUALITIES and r_qual in KNOWN_QUALITIES:
            full_total += 1
            if p_root == r_root and p_qual == r_qual:
                full_hit += 1

    return {
        "song": song_dir.name,
        "duration": round(len(audio) / midi_render.SR_DEFAULT, 1),
        "frames_with_truth": root_total,
        "root_acc": round(root_hit / root_total, 3) if root_total else 0.0,
        "chord_acc": round(full_hit / full_total, 3) if full_total else 0.0,
        "pred_segments": result["n_segments"],
        "truth_segments": len(truth),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/POP909")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--out", default="out/bench")
    args = parser.parse_args()

    root = Path(args.data)
    songs = sorted(p for p in root.iterdir() if p.is_dir())[: args.limit]
    rows = []
    for song in songs:
        try:
            row = evaluate_song(song)
        except Exception as exc:  # noqa: BLE001
            print(f"{song.name}: FAILED {exc}", file=sys.stderr)
            continue
        if row:
            rows.append(row)
            print(f"{row['song']}: root {row['root_acc']:.0%} · chord {row['chord_acc']:.0%} "
                  f"· 段 {row['pred_segments']} vs 真值 {row['truth_segments']}")

    if not rows:
        print("no songs evaluated", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "chord_bench.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    root_mean = float(np.mean([r["root_acc"] for r in rows]))
    chord_mean = float(np.mean([r["chord_acc"] for r in rows]))
    print(f"\n{len(rows)} 首平均：根音准确率 {root_mean:.1%} · 和弦准确率 {chord_mean:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
