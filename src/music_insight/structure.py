"""测量层的结构分析：2 秒块特征 → 新颖度检测边界 → 同质段聚类。

注意：没有做人声/伴奏分离时，段落边界主要来自响度与音色变化，
对“主歌/副歌”这类靠旋律区分的结构并不敏感。这是已知限制（见 docs/DESIGN.md）。
"""

from __future__ import annotations

import numpy as np

from . import features as F

BLOCK_SEC = 2.0
NOVELTY_WINDOW = 6
MIN_GAP_BLOCKS = 4


def _band_matrix(S: np.ndarray, sr: int) -> list[np.ndarray]:
    bands = ((30, 150), (150, 400), (400, 1000), (1000, 2500), (2500, 6000), (6000, 12000))
    return [F.band_energy(S, sr, lo, hi) for lo, hi in bands]


def analyze(feat: dict) -> dict:
    """输入 features.extract() 的结果，输出结构相关的数据。"""
    S, logmel, ch = feat["_S"], feat["_logmel"], feat["_chroma"]
    cent, loud, flux = feat["_centroid"], feat["_loud"], feat["_flux"]
    sr = feat["sample_rate"]
    hop = F.HOP
    dur = feat["duration_sec"]

    fps = sr / hop
    blen = max(1, int(BLOCK_SEC * fps))
    nblk = max(2, S.shape[1] // blen)

    bands = _band_matrix(S, sr)
    feats, block_loud, block_cent, chords = [], [], [], []
    for i in range(nblk):
        sl = slice(i * blen, (i + 1) * blen)
        c = ch[:, sl].mean(axis=1)
        loud_db = float(loud[sl].mean()) if sl.stop <= len(loud) else -99.0
        block_loud.append(loud_db)
        block_cent.append(float(cent[sl].mean()))
        spec = np.array([float(b[sl].mean()) for b in bands]
                        + [float(np.log1p(cent[sl].mean())), loud_db])
        # 原始特征：音高类权重 ×2，其余各维随后统一 z-score 到同一量级
        feats.append(np.concatenate([2.0 * c, spec]))
        top = np.argsort(c)[::-1][:3]
        chords.append("".join(F.PITCH_NAMES[j] for j in sorted(top)))

    feats = np.array(feats)
    if nblk > 1:  # 各维标准化，避免“响度”这种大数值维度被音级淹没
        feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-9)
    normed = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)

    # 短音频（<30 秒）用更小的窗口，否则边界检测退化成“只有一段”
    window = NOVELTY_WINDOW if nblk >= 3 * NOVELTY_WINDOW else max(2, nblk // 3)
    min_gap = MIN_GAP_BLOCKS if nblk >= 3 * MIN_GAP_BLOCKS else 2

    novelty = np.zeros(nblk, np.float32)
    for i in range(nblk):
        l0, l1, r0, r1 = max(0, i - window), i, i, min(nblk, i + window)
        if l1 <= l0 or r1 <= r0:
            continue
        left, right = normed[l0:l1].mean(axis=0), normed[r0:r1].mean(axis=0)
        novelty[i] = 1 - float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-9))

    thr = float(novelty.mean() + 0.35 * novelty.std())
    bounds, last = [0], 0
    for i in range(3, nblk - 3):
        if (novelty[i] >= thr
                and novelty[i] == novelty[max(0, i - 3):i + 4].max()
                and i - last >= min_gap):
            bounds.append(i)
            last = i
    bounds.append(nblk)

    seg_feats = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        f = normed[a:b].mean(axis=0)
        seg_feats.append(f / (np.linalg.norm(f) + 1e-9))

    labels, centroids = [], []
    for f in seg_feats:
        if centroids:
            sims = [float(f @ c) for c in centroids]
            j = int(np.argmax(sims))
            if sims[j] > 0.92 or len(centroids) >= 6:
                labels.append(chr(65 + j))
                centroids[j] = 0.5 * centroids[j] + 0.5 * f
                continue
        centroids.append(f)
        labels.append(chr(65 + len(centroids) - 1))

    segments = []
    for (a, b), label in zip(zip(bounds[:-1], bounds[1:]), labels):
        segments.append({
            "label": label,
            "start": round(a * BLOCK_SEC, 1),
            "end": round(min(b * BLOCK_SEC, dur), 1),
            "chord_guess": chords[a],
            "loud_db": round(float(np.mean(block_loud[a:b])), 1),
            "brightness_hz": round(float(np.mean(block_cent[a:b])), 0),
        })

    timeline = []
    for i in range(0, nblk, 4):
        j = min(nblk, i + 4)
        timeline.append({
            "t": round(i * BLOCK_SEC, 1),
            "chord_guess": chords[i],
            "loud_db": round(float(np.mean(block_loud[i:j])), 1),
            "brightness_hz": round(float(np.mean(block_cent[i:j])), 0),
        })

    return {
        "segments": segments,
        "timeline": timeline,
        "novelty": {
            "mean": round(float(novelty.mean()), 4),
            "max": round(float(novelty.max()), 4),
            "threshold": round(thr, 4),
        },
        "_novelty": novelty,
        "_timeline_grid_sec": BLOCK_SEC * 4,
    }
