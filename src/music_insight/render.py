"""证据层：把测量结果渲染成 LLM 可读的图。

图不是给人看的装饰，而是给视觉模型看的“证据”。每个面板对应一组可被引用的测量量。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import numpy as np

WIDTH = 960
BG = (18, 21, 30)
FG = (230, 236, 246)
MUTE = (150, 165, 190)
ACCENT = (96, 165, 250)
PALETTE = [(96, 165, 250), (244, 114, 182), (52, 211, 153),
           (251, 191, 36), (167, 139, 250), (248, 113, 113)]

FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
)


def _font(size: int):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _cmap(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0, 1)
    stops = np.array([[13, 16, 32], [40, 60, 110], [90, 40, 130],
                      [220, 90, 60], [255, 220, 120]], np.float32)
    pos = v * (len(stops) - 1)
    i0 = np.floor(pos).astype(int).clip(0, len(stops) - 2)
    frac = (pos - i0)[..., None]
    return (stops[i0] * (1 - frac) + stops[i0 + 1] * frac).astype(np.uint8)


def _matrix_image(mat: np.ndarray, w: int, h: int) -> Image.Image:
    a = np.asarray(mat, np.float32)
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    ix = np.linspace(0, a.shape[1] - 1, w).astype(int)
    iy = np.linspace(0, a.shape[0] - 1, h).astype(int)
    return Image.fromarray(_cmap(a[np.ix_(iy, ix)]))


def figure(feat: dict, struct: dict, audio: np.ndarray, out_path: str | Path) -> str:
    sr = feat["sample_rate"]
    logmel, chroma = feat["_logmel"], feat["_chroma"]
    height = 810
    img = Image.new("RGB", (WIDTH, height), BG)
    draw = ImageDraw.Draw(img)
    f_title, f_label = _font(22), _font(14)

    draw.text((26, 18), "Music Insight · 测量面板", font=f_title, fill=FG)
    draw.text((26, 50),
              f"{feat['bpm']} BPM · {feat['key']} · 相关 {feat['key_correlation']}"
              f" · {feat['duration_sec']}s · 平均 {feat['loudness_db']['mean']} dB",
              font=f_label, fill=MUTE)

    # 波形
    wave_h = 130
    wave = Image.new("RGB", (WIDTH - 52, wave_h), (26, 30, 44))
    wd = ImageDraw.Draw(wave)
    mid = wave_h // 2
    step = max(1, len(audio) // (WIDTH - 52))
    for px in range(WIDTH - 52):
        seg = audio[px * step:(px + 1) * step]
        if seg.size == 0:
            break
        amp = int(np.max(np.abs(seg)) * (mid - 5))
        wd.line([(px, mid - amp), (px, mid + amp)], fill=ACCENT)
    img.paste(wave, (26, 78))

    img.paste(_matrix_image(logmel[::-1], WIDTH - 52, 210), (26, 226))
    draw.text((30, 230), "梅尔谱（下→上：低→高，越亮能量越强）", font=f_label, fill=(240, 240, 240))

    img.paste(_matrix_image(chroma[::-1], WIDTH - 52, 230), (26, 446))
    draw.text((30, 450), "音高类随时间（下→上：C C# D … B）", font=f_label, fill=(240, 240, 240))

    total = max(feat["duration_sec"], 1e-6)
    bar_y, bar_h = 700, 58
    draw.text((26, 676), "段落结构（2 秒块聚类，同色=同类）", font=f_label, fill=(240, 240, 240))
    for seg in struct["segments"]:
        x0 = 26 + int((WIDTH - 52) * seg["start"] / total)
        x1 = 26 + int((WIDTH - 52) * min(seg["end"], total) / total)
        idx = (ord(seg["label"]) - 65) % len(PALETTE)
        draw.rectangle([x0, bar_y, max(x1, x0 + 1), bar_y + bar_h], fill=PALETTE[idx])
        if x1 - x0 > 26:
            draw.text(((x0 + x1) / 2, bar_y + bar_h / 2), seg["label"],
                      font=f_label, fill=(20, 24, 34), anchor="mm")

    draw.text((26, 772),
              f"峰值相对电平 {feat['crest_db']} dB · 音符密度 {feat['note_density_per_sec']}/s"
              f" · 中频带能量占比 {feat['mid_band_ratio']:.0%}（音色描述，非人声判断）",
              font=f_label, fill=MUTE)

    out_path = Path(out_path)
    img.save(out_path)
    return str(out_path)
