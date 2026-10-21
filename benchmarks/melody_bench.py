"""旋律提取基准：POP909 的 MELODY 轨 vs 从渲染音频里跟踪出的主旋律。

指标：
- 有声帧占比（我们判成有声的比例）
- RPA：双方都有声的帧里，音高误差在 50 音分内的比例
- 八度容忍准确率（误差落在 1200k ± 50 音分）
- 方向一致率：连续有声帧的音高变化方向是否一致
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from music_insight import melody, midi_render  # noqa: E402


def melody_truth(midi_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """返回 MELODY 轨的音符 (起点, 终点, midi)。找不到 MELODY 轨就退化为最高音轨。"""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    track = None
    for inst in pm.instruments:
        if "melody" in (inst.name or "").lower():
            track = inst
            break
    if track is None or not track.notes:
        candidates = [i for i in pm.instruments if i.notes]
        if not candidates:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        track = max(candidates, key=lambda i: np.mean([n.pitch for n in i.notes]))
    starts = np.array([n.start for n in track.notes])
    ends = np.array([n.end for n in track.notes])
    pitches = np.array([n.pitch for n in track.notes], float)
    return starts, ends, pitches


def truth_at(times: np.ndarray, starts, ends, pitches) -> np.ndarray:
    out = np.zeros(len(times))
    for i, t in enumerate(times):
        idx = np.where((starts <= t) & (t < ends))[0]
        if idx.size:
            out[i] = pitches[idx[0]]
    return out


def metrics(pred_f0: np.ndarray, ref_f0: np.ndarray) -> dict:
    voiced_both = (pred_f0 > 0) & (ref_f0 > 0)
    cents = np.abs(pred_f0 - ref_f0) * 100.0
    rpa = float(np.mean(cents[voiced_both] <= 50)) if voiced_both.any() else 0.0

    oct_ok = np.zeros_like(cents, dtype=bool)
    if voiced_both.any():
        for k in range(-2, 3):
            oct_ok |= np.abs(cents - 1200 * k) <= 50
    oct_acc = float(np.mean(oct_ok[voiced_both])) if voiced_both.any() else 0.0

    idx = np.where(voiced_both)[0]
    direction = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a <= 3:
            dp = pred_f0[b] - pred_f0[a]
            dr = ref_f0[b] - ref_f0[a]
            if abs(dr) > 0.3 and abs(dp) > 0.3:
                direction.append(np.sign(dp) == np.sign(dr))
    dir_acc = float(np.mean(direction)) if direction else 0.0

    return {
        "voiced_ratio": round(float(np.mean(pred_f0 > 0)), 3),
        "rpa": round(rpa, 3),
        "octave_acc": round(oct_acc, 3),
        "direction_acc": round(dir_acc, 3),
        "frames": int(len(pred_f0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/POP909")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out", default="out/bench")
    args = ap.parse_args()

    songs = sorted(p for p in Path(args.data).iterdir() if p.is_dir())[: args.limit]
    rows = []
    for song in songs:
        midi = song / f"{song.name}.mid"
        if not midi.exists():
            continue
        audio, _ = midi_render.render_midi(str(midi))
        pred = melody.track_melody(audio, midi_render.SR_DEFAULT)
        starts, ends, pitches = melody_truth(midi)
        if starts.size == 0:
            continue
        ref = truth_at(pred["times"], starts, ends, pitches)
        m = metrics(pred["f0_midi"], ref)
        rows.append({"song": song.name, **m})
        print(f"{song.name}: 有声 {m['voiced_ratio']:.0%} · RPA {m['rpa']:.0%} "
              f"· 八度容差 {m['octave_acc']:.0%} · 方向 {m['direction_acc']:.0%}")

    if not rows:
        print("no songs", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "melody_bench.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for key in ("voiced_ratio", "rpa", "octave_acc", "direction_acc"):
        print(f"{key}: {np.mean([r[key] for r in rows]):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
