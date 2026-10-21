"""生成一段可复现的测试音频：python examples/make_demo.py out/demo.wav"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from music_insight.synth import synth_demo  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/demo.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    duration = synth_demo(str(out), bpm=100.0, bars=8)
    print(f"wrote {out} ({duration:.1f}s, 100 BPM, I-V-vi-IV)")


if __name__ == "__main__":
    main()
