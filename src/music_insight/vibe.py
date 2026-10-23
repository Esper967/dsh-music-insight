"""感官层：把测量值翻译成「像听过」的话。

证据层（``measurements`` / ``evidence``）负责**真**：每个数字可复现、能回溯字段。
感官层负责**好听**：把数字映射成形容词和叙事弧，下游模型就不用自己从
「-20.8 dB / 频谱重心 2682 Hz」翻译到「雨后的薄雾」。

边界
----
- 只消费 ``features`` / ``structure`` / ``chords`` 已有字段，**不新增音频计算**
  （零成本、可复现、可离线）。
- 不做乐器 / 人声 / 流派判断——证据层没有的东西，这里也不许有。
- 输出一律标记 ``method="vibe-v1"``。它是**约定俗成的映射，不是测量**；
  要严谨引用请用 ``measurements`` / ``evidence``。

一句话：这一层不发明事实，只发明措辞。
"""

from __future__ import annotations

from pathlib import Path

# --- 阈值表 ---------------------------------------------------------------
# 每一项都是「某个测量值的分段 → 一句人话」。改这里就是改解读词。
# 阈值取自本仓库 18 首真实纯音乐 + 合成样本的经验分布，不是校准过的心理量。

LOUD_STEPS = (
    (-30.0, "很轻"),
    (-24.0, "轻"),
    (-18.0, "中等"),
    (-12.0, "偏响"),
    (1e9, "很满"),
)

BRIGHT_STEPS = (
    (800.0, "暗、闷"),
    (1300.0, "偏厚"),
    (2000.0, "中性"),
    (3200.0, "亮"),
    (1e9, "很亮、薄"),
)

TEMPO_STEPS = (
    (70.0, "很慢、拖着走"),
    (95.0, "慢"),
    (125.0, "不快不慢"),
    (145.0, "偏快"),
    (1e9, "快、往前推"),
)

DENSITY_STEPS = (
    (1.2, "几乎没击打，音一直铺着"),
    (3.0, "起音正常"),
    (6.0, "起音偏密"),
    (1e9, "起音很密"),
)

LENGTH_STEPS = (
    (60.0, "短，像一段片段"),
    (150.0, "偏短"),
    (240.0, "正常长度"),
    (1e9, "长"),
)


def _step(value: float, table) -> str:
    """按上界查表；表最后一档用 1e9 兜底。"""
    for upper, word in table:
        if value <= upper:
            return word
    return table[-1][1]


# --- 元信息 ---------------------------------------------------------------

_TAG_ALIASES = {
    "title": ("title", "tit2", "\xa9nam", "tracktitle"),
    "artist": ("artist", "tpe1", "\xa9art", "albumartist", "\xa9aart"),
    "album": ("album", "talb", "\xa9alb"),
    "year": ("date", "year", "tyer", "tdrc", "\xa9day"),
    "genre": ("genre", "tcon", "\xa9gen"),
}


def _split_stem(stem: str) -> tuple[str, str]:
    """文件名兜底：「歌手 - 歌名」或「01. 歌名」。"""
    s = stem.strip()
    # 去掉音轨号前缀：01 / 01. / 01- / [01]
    for prefix_len in (3, 2):
        head = s[:prefix_len]
        if head and head[0].isdigit() and s[prefix_len:prefix_len + 1] in (".", "-", " ", "_"):
            s = s[prefix_len + 1:].strip()
            break
    if " - " in s:  # 注意是全角/半角无关的第一处
        artist, title = s.split(" - ", 1)
        return title.strip(), artist.strip()
    return s, ""


def _read_tags(path: str) -> dict:
    """读容器标签（mp3 ID3 / m4a / flac）。PyAV 缺失或失败时返回空字典。"""
    try:
        import av  # type: ignore
    except ImportError:
        return {}
    try:
        with av.open(path) as container:
            raw = dict(container.metadata or {})
            for stream in container.streams:
                if stream.type == "audio" and stream.metadata:
                    raw.update(stream.metadata)
    except Exception:  # noqa: BLE001 - 标签是锦上添花，不能拖垮主流程
        return {}

    flat = {str(k).strip().lower(): str(v).strip() for k, v in raw.items() if v}
    out: dict[str, str] = {}
    for field, keys in _TAG_ALIASES.items():
        for key in keys:
            value = flat.get(key.lower())
            if value:
                out[field] = value
                break
    return out


def read_metadata(path: str) -> dict:
    """歌曲元信息：tag 优先，文件名兜底。

    标题本身就是最强的情绪线索（「雨上がりの空」比任何频谱都直接），
    证据层不算它，所以单独从容器里取。
    """
    p = Path(path)
    tags = _read_tags(str(p))
    file_title, file_artist = _split_stem(p.stem)
    return {
        "source": str(p),
        "title": tags.get("title") or file_title,
        "artist": tags.get("artist") or file_artist,
        "album": tags.get("album", ""),
        "year": tags.get("year", "")[:4] if tags.get("year") else "",
        "genre": tags.get("genre", ""),
        "from_tags": bool(tags.get("title") or tags.get("artist")),
    }


# --- 骨架：每段在做什么 -----------------------------------------------------


def _segment_word(rel_db: float, rel_bright: float, first: bool, last: bool,
                  phase: str) -> str:
    """单段听感。参数都是**相对全曲均值**的偏差，避免不同母带的绝对电平差。"""
    if first:
        if rel_db <= -3.0 and rel_bright > 200.0:
            return "薄薄一层，还没落地"
        if rel_db <= -3.0:
            return "压得很低，像还没开始"
        return "一上来就很满" if rel_db >= 3.0 else "一上来就铺着"
    if last:
        if rel_db <= -3.0 and rel_bright > 200.0:
            return "抽空，只剩一点空气"
        if rel_db <= -3.0:
            return "慢慢退场"
        return "到最后一个音还在响"
    if phase == "满":
        return "最满，什么都推上来" if rel_db >= 2.0 else "这里是顶点，其实没比别处响多少"
    if phase == "涨":
        if rel_db >= 3.0 and rel_bright >= 0.0:
            return "涨起来，也亮了"
        if rel_db >= 3.0:
            return "变厚，压着往前"
        return "在往上走，还没到顶"
    if phase == "退":
        if rel_db <= -3.0:
            return "退下去，空一点"
        if rel_db >= 1.0:
            return "还厚着，但开始往下"
        return "开始收，慢慢空"
    if rel_db >= 3.0 and rel_bright >= 0.0:
        return "涨起来，也亮了"
    if rel_db >= 3.0:
        return "变厚，压着往前"
    if rel_db <= -3.0:
        return "退下去，空一点"
    if rel_bright >= 500.0:
        return "铺着，偏亮"
    if rel_bright <= -500.0:
        return "铺着，偏暗"
    return "平着走"


def _shape(louds: list[float]) -> str:
    """把响度序列归类成一句「弧线」。基线用中位数，见 build() 里的说明。"""
    if len(louds) < 2:
        return "一条线走到底"
    mid = _median(louds)
    span = max(louds) - min(louds)
    first, last = louds[0], louds[-1]
    peak_pos = louds.index(max(louds)) / (len(louds) - 1)

    if span < 3.5:
        return "平铺，一条线走到底"
    if first <= mid - 2.0 and last <= mid - 2.0 and 0.2 <= peak_pos <= 0.8:
        return "从空气里出来，中间涨满，最后再散回空气"
    if first <= mid - 2.0 and last >= mid + 1.0:
        return "一路往上，越走越开"
    if first >= mid + 1.0 and last <= mid - 2.0:
        return "开得高，然后一路收"
    if last <= mid - 2.0:
        return "慢慢淡出"
    if first <= mid - 2.0:
        return "起手很轻，后来才进来"
    return "中间有起伏"


def _hand(measurements: dict) -> dict:
    """手感：速度 / 明暗 / 密度 / 长度 / 温度。"""
    key = str(measurements.get("key", ""))
    warm = "偏暖" if key.endswith("大调") else ("偏冷" if key.endswith("小调") else "不好说")
    return {
        "tempo": _step(float(measurements.get("bpm") or 0.0), TEMPO_STEPS),
        "light": _step(float(measurements.get("brightness_mean_hz") or 1000.0), BRIGHT_STEPS),
        "density": _step(float(measurements.get("note_density_per_sec") or 0.0), DENSITY_STEPS),
        "length": _step(float(measurements.get("duration_sec") or 0.0), LENGTH_STEPS),
        "warmth": warm,
        "loudness": _step(float((measurements.get("loudness_db") or {}).get("mean") or -30.0),
                          LOUD_STEPS),
    }


def _tags(hand: dict, mood: dict, shape: str) -> list[str]:
    """两三个标签，聊天时可以直接用。"""
    arousal = float(mood.get("arousal") or 0.0)
    valence = float(mood.get("valence") or 0.0)
    if arousal >= 0.55 and valence >= 0.55:
        tags = ["偏亮", "有点上头"]
    elif arousal >= 0.55:
        tags = ["紧", "有压迫感"]
    elif valence >= 0.55:
        tags = ["松弛", "偏暖"]
    else:
        tags = ["沉", "往下坠"]

    if hand["warmth"] == "偏暖" and "偏暖" not in tags:
        tags.append("偏暖")
    if "平铺" in shape and "有起伏" not in tags:
        tags.append("平铺")
    elif "平铺" not in shape:
        tags.append("有起伏")
    return tags[:3]


def _anchors(segments: list[dict], measurements: dict) -> list[str]:
    """几句可以被直接引用的「听感锚点」——都带数字，但说的是感觉。"""
    out: list[str] = []
    if not segments:
        return out

    louds = [s["loud_db"] for s in segments]
    mid = _median(louds)
    span = max(louds) - min(louds)

    intro = segments[0]
    intro_dur = intro["end"] - intro["start"]
    intro_rel = intro["loud_db"] - mid
    if intro_rel <= -3.0:
        out.append(
            f"开头 {intro_dur:.0f} 秒平均 {intro['loud_db']:.0f} dB，"
            f"重心 {intro['brightness_hz']:.0f} Hz，"
            + ("像一层高频的水汽，主体还没进来。" if intro["brightness_hz"] > 1500
               else "压得很低，像还没开始。")
        )
    else:
        out.append(
            f"开头就进来了：{intro_dur:.0f} 秒平均 {intro['loud_db']:.0f} dB，"
            f"重心 {intro['brightness_hz']:.0f} Hz。"
        )

    if span >= 3.5:
        peak = segments[louds.index(max(louds))]
        out.append(
            f"最满的是 {peak['start']:.0f}–{peak['end']:.0f} 秒，"
            f"比常态响 {peak['loud_db'] - mid:+.1f} dB，"
            f"重心 {peak['brightness_hz']:.0f} Hz。"
        )
    else:
        out.append(f"全曲响度只差 {span:.1f} dB，几乎是一条直线——它不打算给你起伏。")

    last = segments[-1]
    if last["loud_db"] <= mid - 3.0:
        out.append(
            f"最后 {last['end'] - last['start']:.0f} 秒掉到 {last['loud_db']:.0f} dB"
            + (f"，亮度反而回到 {last['brightness_hz']:.0f} Hz——像水汽散了，只剩天空。"
               if last["brightness_hz"] > measurements.get("brightness_mean_hz", 0) * 1.3
               else "，慢慢被收走。")
        )

    if span >= 3.5:
        out.append(f"全曲响度跨度 {span:.1f} dB，"
                   f"从「{_step(min(louds), LOUD_STEPS)}」走到「{_step(max(louds), LOUD_STEPS)}」。")
    return out


# --- 组装 ----------------------------------------------------------------


def _units(result: dict) -> tuple[list[dict], str]:
    """取讲故事的骨架单元。

    优先用结构段；**段太少（<3）时退回 8 秒时间线**——短歌或结构检测没抓到
    边界时，只有两段会让叙事变成「一直很响」，反而是错的。
    """
    segments = result.get("structure") or []
    if len(segments) >= 3:
        return list(segments), "segments"

    timeline = result.get("timeline") or []
    if len(timeline) < 3:
        return list(segments), "segments"

    duration = float((result.get("measurements") or {}).get("duration_sec") or 0.0)
    step = 8.0
    if len(timeline) >= 2:
        step = max(1.0, float(timeline[1]["t"]) - float(timeline[0]["t"]))
    units = []
    for row in timeline:
        start = float(row["t"])
        units.append({
            "label": "·",
            "start": start,
            "end": min(start + step, duration) if duration else start + step,
            "loud_db": row["loud_db"],
            "brightness_hz": row["brightness_hz"],
        })
    return units, "timeline"


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def build(result: dict) -> dict:
    """输入已经组装好的 result（report.build 之后、chords 之后都行）。"""
    measurements = result.get("measurements") or {}
    units, grain = _units(result)
    mood = result.get("mood_priors") or {}
    meta = result.get("meta") or {}

    louds = [float(u.get("loud_db", -30.0)) for u in units]
    brights = [float(u.get("brightness_hz", 1000.0)) for u in units]
    mean_loud = sum(louds) / len(louds) if louds else -30.0
    # 用中位数而不是均值当「常态」：长尾的安静段会把均值拖走，
    # 那样开头明明是薄雾也会被判成「正常」。
    mid_loud = _median(louds)
    mid_bright = _median(brights)
    peak_loud = max(louds) if louds else mean_loud

    arc = []
    for i, seg in enumerate(units):
        loud = float(seg.get("loud_db", mid_loud))
        bright = float(seg.get("brightness_hz", mid_bright))
        first, last = i == 0, i == len(units) - 1
        if first:
            phase = "起"
        elif loud == peak_loud and len(units) > 1:
            phase = "满"
        elif last:
            phase = "收"
        elif i <= louds.index(peak_loud):
            phase = "涨"
        else:
            phase = "退"
        arc.append({
            "start": seg.get("start"),
            "end": seg.get("end"),
            "label": seg.get("label"),
            "phase": phase,
            "loud_db": seg.get("loud_db"),
            "brightness_hz": seg.get("brightness_hz"),
            "word": _segment_word(loud - mid_loud, bright - mid_bright, first, last, phase),
        })

    shape = _shape(louds)
    hand = _hand(measurements)
    tags = _tags(hand, mood, shape)
    title = meta.get("title") or Path(str(result.get("source", "这首"))).stem

    one_liner = (
        f"《{title}》是一条「{shape}」的弧："
        f"{hand['tempo']}，{hand['light']}，{hand['warmth']}，{hand['length']}。"
    )

    anchors = _anchors(units, measurements)

    chords = result.get("key_from_chords") or {}
    if chords.get("top_chords"):
        anchors.append(f"和弦主要绕着 {' / '.join(chords['top_chords'][:4])} 转。")

    return {
        "method": "vibe-v1",
        "grain": grain,
        "one_liner": one_liner,
        "shape": shape,
        "arc": arc,
        "hand": hand,
        "tags": tags,
        "anchors": anchors,
        "caveat": "把测量值翻译成人话的启发式映射，不是情绪识别；"
                  "它只负责措辞，数字仍然以 measurements 为准。",
    }
