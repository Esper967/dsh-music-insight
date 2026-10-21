"""和弦识别基准：逐帧 vs 按拍聚合（POP909，渲染音频）。

用法：
    python benchmarks/beat_sync_bench.py --data ../data/POP909 --limit 20
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import benchmarks.chord_bench as B  # noqa: E402
from music_insight import chords, midi_render  # noqa: E402


def load_beats(path: Path, end_time: float) -> np.ndarray:
    times = sorted(float(line.split()[0]) for line in
                   path.read_text(encoding="utf-8").splitlines() if line.strip())
    times = np.array(times)
    return np.append(times, max(end_time, times[-1] + 0.5))


def score(pred, times, truth):
    ref = B.truth_at(np.asarray(times), truth)
    rh = rt = ch = ct = 0
    for p, r in zip(pred, ref):
        if r is None:
            continue
        rt += 1
        pr, pq = B.split_chord(p)
        rr, rq = B.split_chord(r)
        rh += int(pr == rr)
        if pq in B.KNOWN_QUALITIES and rq in B.KNOWN_QUALITIES:
            ct += 1
            ch += int(pr == rr and pq == rq)
    return (rh / rt if rt else 0.0), (ch / ct if ct else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/POP909")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", default="out/bench")
    args = ap.parse_args()

    songs = sorted(p for p in Path(args.data).iterdir() if p.is_dir())[: args.limit]
    rows = []
    for song in songs:
        midi = song / f"{song.name}.mid"
        truth_path = song / "chord_midi.txt"
        beat_path = song / "beat_midi.txt"
        if not (midi.exists() and truth_path.exists() and beat_path.exists()):
            continue
        audio, duration = midi_render.render_midi(str(midi))
        truth = B.load_truth(truth_path)

        frame = chords.detect(audio, midi_render.SR_DEFAULT)
        f_root, f_chord = score(frame["frame_labels"],
                                np.arange(len(frame["frame_labels"])) * frame["hop_sec"], truth)

        beats = load_beats(beat_path, duration)
        sync = chords.detect_beat_sync(audio, midi_render.SR_DEFAULT, beats)
        b_root, b_chord = score(sync["beat_labels"], sync["beat_centers"], truth)

        rows.append({"song": song.name, "beats": len(beats) - 1,
                     "frame_root": round(f_root, 3), "frame_chord": round(f_chord, 3),
                     "beat_root": round(b_root, 3), "beat_chord": round(b_chord, 3)})
        print(f"{song.name}: 帧 {f_root:.0%}/{f_chord:.0%} · 拍 {b_root:.0%}/{b_chord:.0%}")

    if not rows:
        print("no songs", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "beat_sync_bench.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for prefix, label in (("frame", "逐帧"), ("beat", "按拍")):
        r = np.mean([row[f"{prefix}_root"] for row in rows])
        c = np.mean([row[f"{prefix}_chord"] for row in rows])
        print(f"\n{label}（{len(rows)} 首）：根音 {r:.1%} · 和弦 {c:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
