"""节拍跟踪 + 和弦识别基准（POP909，渲染音频）。

回答两个问题：
1. 自己的节拍跟踪器离真值拍点多远（F-measure，容差 70 ms）
2. 用跟踪拍点做和弦识别，比用真值拍点掉多少
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
from music_insight import beats as beat_mod  # noqa: E402
from music_insight import chords, midi_render  # noqa: E402


def load_truth_beats(path: Path, duration: float) -> np.ndarray:
    times = sorted(float(line.split()[0]) for line in
                   path.read_text(encoding="utf-8").splitlines() if line.strip())
    return np.append(np.array(times), max(duration, times[-1] + 0.5))


def f_measure(ref: np.ndarray, est: np.ndarray, tol: float = 0.07) -> float:
    if len(ref) == 0 or len(est) == 0:
        return 0.0
    used = np.zeros(len(ref), bool)
    tp = 0
    for t in est:
        d = np.abs(ref - t)
        j = int(np.argmin(d))
        if d[j] <= tol and not used[j]:
            used[j] = True
            tp += 1
    p = tp / len(est)
    r = tp / len(ref)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def chord_accuracy(pred, times, truth) -> tuple[float, float]:
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
        midi, truth_path = song / f"{song.name}.mid", song / "chord_midi.txt"
        beat_path = song / "beat_midi.txt"
        if not (midi.exists() and truth_path.exists() and beat_path.exists()):
            continue
        audio, duration = midi_render.render_midi(str(midi))
        truth = B.load_truth(truth_path)
        truth_beats = load_truth_beats(beat_path, duration)

        tracked = beat_mod.track_beats(audio, midi_render.SR_DEFAULT)
        if len(tracked) < 4:
            print(f"{song.name}: 跟踪失败", file=sys.stderr)
            continue
        tracked = np.append(tracked, duration)

        fmeas = f_measure(truth_beats[:-1], tracked[:-1])

        syn_truth = chords.detect_beat_sync(audio, midi_render.SR_DEFAULT, truth_beats)
        syn_tracked = chords.detect_beat_sync(audio, midi_render.SR_DEFAULT, tracked)
        rt_root, rt_chord = chord_accuracy(syn_truth["beat_labels"], syn_truth["beat_centers"], truth)
        rk_root, rk_chord = chord_accuracy(syn_tracked["beat_labels"], syn_tracked["beat_centers"], truth)

        rows.append({"song": song.name, "f_measure": round(fmeas, 3),
                     "truth_root": round(rt_root, 3), "truth_chord": round(rt_chord, 3),
                     "tracked_root": round(rk_root, 3), "tracked_chord": round(rk_chord, 3)})
        print(f"{song.name}: 节拍 F {fmeas:.2f} · 和弦(真值拍) {rt_root:.0%}/{rt_chord:.0%} "
              f"· 和弦(跟踪拍) {rk_root:.0%}/{rk_chord:.0%}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "beat_track_bench.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} 首：节拍 F-measure {np.mean([r['f_measure'] for r in rows]):.3f}")
    print(f"和弦（真值拍）：根音 {np.mean([r['truth_root'] for r in rows]):.1%}"
          f" · {np.mean([r['truth_chord'] for r in rows]):.1%}")
    print(f"和弦（跟踪拍）：根音 {np.mean([r['tracked_root'] for r in rows]):.1%}"
          f" · {np.mean([r['tracked_chord'] for r in rows]):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
