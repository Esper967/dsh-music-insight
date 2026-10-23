"""感官层：把测量值翻译成「一句话的感受」。

v2 的方向变了：**短、直接、说情绪**。

v1 会输出骨架表、手感表、四五个「听感锚点」——信息全，但读起来像体检报告，
模型接过去也只能继续写长。v2 只保留四样东西：

    headline  一句感受（可以比喻，但只一句）
    feelings  三个词
    arc_line  一行走向
    hand_line 一行手感

其余细节留在 ``arc`` / ``hand`` 里给机器看，不往人眼前送。
证据层（``measurements`` / ``evidence``）一个字没少——**短的是措辞，不是数据**。

边界不变：只消费已有测量值，不新增音频计算，不判断乐器 / 人声 / 流派。
输出标 ``method="vibe-v2"``：约定俗成的映射，不是测量。
"""

from __future__ import annotations

from pathlib import Path

# --- 阈值表 ---------------------------------------------------------------
# 每张表 =「测量值分段 → 一句人话」。改这里就是改解读词。
# 长词用于报告/摘要，短词用于一行式的 feelings；两者同一套阈值。

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

# 一行式短词（feelings / hand_line 用）
SHORT_STEPS = {
    "light": ((800.0, "闷"), (1300.0, "厚"), (2000.0, "平"), (3200.0, "亮"), (1e9, "很亮")),
    "tempo": ((70.0, "慢"), (95.0, "偏慢"), (125.0, "稳"), (145.0, "偏快"), (1e9, "快")),
    "density": ((1.2, "空"), (3.0, "铺"), (6.0, "密"), (1e9, "很密")),
    "length": ((60.0, "短"), (150.0, "偏短"), (240.0, "正常"), (1e9, "长")),
}

# feelings 是「感受词」不是「音色词」：同样是亮，这里写「透」而不是「很亮」。
# 和 SHORT_STEPS 分开，是因为 hand_line 要描述事实，feelings 要描述情绪。
FEEL_STEPS = {
    "light": ((800.0, "闷"), (1300.0, "沉"), (2000.0, "淡"), (3200.0, "透"), (1e9, "飘")),
    "tempo": ((70.0, "拖"), (95.0, "缓"), (125.0, "稳"), (145.0, "轻快"), (1e9, "急")),
    "density": ((1.2, "空"), (3.0, "松"), (6.0, "满"), (1e9, "挤")),
}


def _step(value: float, table) -> str:
    for upper, word in table:
        if value <= upper:
            return word
    return table[-1][1]


def _step_short(key: str, value: float) -> str:
    return _step(value, SHORT_STEPS[key])


def _feel_word(key: str, value: float) -> str:
    return _step(value, FEEL_STEPS[key])


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
    for prefix_len in (3, 2):
        head = s[:prefix_len]
        if head and head[0].isdigit() and s[prefix_len:prefix_len + 1] in (".", "-", " ", "_"):
            s = s[prefix_len + 1:].strip()
            break
    if " - " in s:
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
    """单段听感（给 arc 细节用，不直接上摘要）。"""
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


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _shape(louds: list[float]) -> str:
    """把响度序列归类成一句「弧线」。基线用中位数：长尾的安静段会把均值拖走。"""
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
    """手感：速度 / 明暗 / 密度 / 长度 / 温度（长词版，给报告细节）。"""
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


# --- 一句话：把骨架 + 手感压成感受 ---------------------------------------------

_HEADLINES = {
    "平铺": "一条直线，不哄你也不推你。",
    "从空气里出来": "像雨停之后一个人走回家：先是空的，中间暖起来，最后散了。",
    "一路往上": "像憋着一口气往上走，越到后面越开。",
    "开得高": "一上来就给满，然后一路收着走。",
    "慢慢淡出": "慢慢淡下去，像自己走远的。",
    "起手很轻": "开头很轻，像还没决定要不要开始。",
    "中间有起伏": "中间起起伏伏，像情绪一直没坐稳。",
}


def _mood_tail(mood: dict) -> str:
    arousal = float(mood.get("arousal") or 0.0)
    valence = float(mood.get("valence") or 0.0)
    if valence >= 0.55 and arousal >= 0.55:
        return "有点上头。"
    if valence >= 0.55:
        return "听完是松的。"
    if arousal >= 0.55:
        return "有点燥，压着。"
    return "听着有点闷。"


def _headline(shape: str, hand: dict, mood: dict) -> str:
    body = "中间起起伏伏，像情绪一直没坐稳。"
    for key, text in _HEADLINES.items():
        if key in shape:
            body = text
            break
    if "偏冷" in hand.get("warmth", "") and "雨停" in body:
        body = "像傍晚天一点点暗下去：先是空的，中间热了一下，最后散了。"
    return body + _mood_tail(mood)


def _feelings(hand: dict) -> list[str]:
    """三个感受词：明暗 / 速度 / 密度（走 FEEL_STEPS，不是音色描述）。"""
    return [
        _feel_word("light", float(hand.get("brightness_hz") or 1500.0)),
        _feel_word("tempo", float(hand.get("bpm") or 0.0)),
        _feel_word("density", float(hand.get("density_per_sec") or 0.0)),
    ]


def _mmss(seconds: float) -> str:
    total = max(0, int(round(float(seconds or 0))))
    return f"{total // 60}:{total % 60:02d}"


def _arc_line(arc: list[dict]) -> str:
    """一行走向：最多四个点，别把整张表倒出来。"""
    if not arc:
        return ""
    picks = {0, len(arc) - 1}
    peak = max(range(len(arc)), key=lambda i: float(arc[i].get("loud_db") or -99))
    picks.add(peak)
    if len(arc) >= 5:
        picks.add(round((peak + len(arc) - 1) / 2))
    chosen = [arc[i] for i in sorted(picks)]
    return " → ".join(f"{_mmss(seg['start'])} {seg['phase']}" for seg in chosen)


def _hand_line(measurements: dict) -> str:
    bpm = float(measurements.get("bpm") or 0.0)
    light = _step_short("light", float(measurements.get("brightness_mean_hz") or 1500.0))
    density = _step_short("density", float(measurements.get("note_density_per_sec") or 0.0))
    key = str(measurements.get("key", ""))
    warm = "暖" if key.endswith("大调") else ("凉" if key.endswith("小调") else "平")
    return f"{bpm:g} BPM · {light} · {warm} · {density}"


# --- 组装 ----------------------------------------------------------------


def _units(result: dict) -> tuple[list[dict], str]:
    """取讲故事的骨架单元。段太少（<3）时退回 8 秒时间线。"""
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


def build(result: dict) -> dict:
    """输入已经组装好的 result（report.build 之后、chords 之后都行）。"""
    measurements = result.get("measurements") or {}
    units, grain = _units(result)
    mood = result.get("mood_priors") or {}
    meta = result.get("meta") or {}

    louds = [float(u.get("loud_db", -30.0)) for u in units]
    brights = [float(u.get("brightness_hz", 1000.0)) for u in units]
    mid_loud = _median(louds)
    mid_bright = _median(brights)
    peak_loud = max(louds) if louds else mid_loud

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
    headline = _headline(shape, hand, mood)
    hand_values = {
        "bpm": measurements.get("bpm"),
        "brightness_hz": measurements.get("brightness_mean_hz"),
        "density_per_sec": measurements.get("note_density_per_sec"),
    }
    title = meta.get("title") or Path(str(result.get("source", "这首"))).stem

    return {
        "method": "vibe-v2",
        "grain": grain,
        "headline": headline,
        "feelings": _feelings(hand_values),
        "arc_line": _arc_line(arc),
        "hand_line": _hand_line(measurements),
        "title": title,
        "artist": meta.get("artist", ""),
        # 以下给机器看，不进摘要
        "shape": shape,
        "arc": arc,
        "hand": hand,
        "caveat": "把测量值翻译成人话的启发式映射，不是情绪识别；它只负责措辞，数字仍以 measurements 为准。",
    }
