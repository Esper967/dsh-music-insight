"""证据层 + 报告：把测量结果组织成 JSON / CSV / Markdown。

设计原则：报告里每一条事实性结论都必须能指回一个计算出来的字段。
感官层（``vibe.py``）负责把数字翻译成听感，它和测量值**分开放**：
``mode="vibe"`` 的报告先说感觉、数字压到最后；``mode="strict"`` 的报告
仍是一份可引用的证据清单。两种模式共用同一份 ``measurements``。

``LIMITS.md`` 单独写一个文件：限制不会消失，但不该一打开正文就撞上它。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from . import features as F
from . import vibe as vibe_mod


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


BRIGHTNESS_LOW_HZ = 1000.0
BRIGHTNESS_SPAN_HZ = 3000.0


def mood_priors(feat: dict) -> dict:
    """**不是测量**。基于统计先验把客观量线性映射到唤醒/效价。

    校准范围：频谱重心 1000–4000 Hz、每分钟起音 0–12 次、BPM 60–130、平均响度 -30~-8 dB。
    超出范围时数值会被裁剪，此时明确标记“先验失效”，不给情绪标签。
    """
    bpm = feat["bpm"]
    loud = feat["loudness_db"]["mean"]
    density = feat["note_density_per_sec"]
    brightness = feat.get("brightness_mean_hz", BRIGHTNESS_LOW_HZ)
    major = feat["key"].endswith("大调")

    bpm_term = (bpm - 60) / 70
    loud_term = (loud + 30) / 22
    density_term = density / 12.0
    bright_term = (brightness - BRIGHTNESS_LOW_HZ) / BRIGHTNESS_SPAN_HZ

    raw_arousal = 0.5 * bpm_term + 0.3 * loud_term + 0.2 * density_term
    raw_valence = 0.35 * bright_term + 0.3 * (1.0 if major else 0.15) + 0.35 * bpm_term
    saturated = any(v < -0.05 or v > 1.05 for v in
                    (bpm_term, loud_term, density_term, bright_term)) or \
        raw_arousal < 0 or raw_arousal > 1 or raw_valence < 0 or raw_valence > 1

    arousal = _clip(raw_arousal)
    valence = _clip(raw_valence)
    return {
        "method": "heuristic-v0",
        "confidence": "invalid" if saturated else "low",
        "saturated": saturated,
        "arousal": round(arousal, 2),
        "valence": round(valence, 2),
        "labels": "先验失效（输入超出校准范围）" if saturated else _quadrant(arousal, valence),
        "caveat": "线性先验映射，未经验证；只在校准范围内有意义。"
                  "小调+快节奏、亮失真吉他等都会打破这个映射，不要把它当测量值使用。",
    }


def _quadrant(arousal: float, valence: float) -> str:
    if arousal >= 0.5 and valence >= 0.5:
        return "高唤醒 / 偏正面"
    if arousal >= 0.5:
        return "高唤醒 / 偏负面"
    if valence >= 0.5:
        return "低唤醒 / 偏正面"
    return "低唤醒 / 偏负面"


def evidence_index(feat: dict, struct: dict) -> list[dict]:
    """把“结论 → 字段 → 值”列成表，供解读层逐条引用。"""
    items = [
        ("速度", "bpm", f"{feat['bpm']} BPM"),
        ("调性", "key", f"{feat['key']}（相关 {feat['key_correlation']}）"),
        ("调性置信", "key_margin", f"候选间隔 {feat.get('key_margin', '-')}"),
        ("时长", "duration_sec", f"{feat['duration_sec']} 秒"),
        ("平均响度", "loudness_db.mean", f"{feat['loudness_db']['mean']} dB"),
        ("响度范围", "loudness_db.min/max",
         f"{feat['loudness_db']['min']} ~ {feat['loudness_db']['max']} dB"),
        ("峰值相对电平", "crest_db", f"{feat['crest_db']} dB"),
        ("音符密度", "note_density_per_sec", f"{feat['note_density_per_sec']}/s"),
        ("中频带能量占比", "mid_band_ratio",
         f"{feat['mid_band_ratio']:.0%}（音色描述，与人声无关）"),
        ("主要音级", "mean_chroma_top3", " / ".join(feat["mean_chroma_top3"])),
        ("段落数", "segments", f"{len(struct['segments'])} 段"),
    ]
    return [{"claim": c, "field": f, "value": v} for c, f, v in items]


def build(feat: dict, struct: dict, source: str, image: str | None = None) -> dict:
    brightness = struct["timeline"][0]["brightness_hz"] if struct["timeline"] else 1000
    if struct["timeline"]:
        brightness = sum(t["brightness_hz"] for t in struct["timeline"]) / len(struct["timeline"])
    feat_with_bright = {**feat, "brightness_mean_hz": brightness}

    result = {
        "source": source,
        "image": image,
        "meta": vibe_mod.read_metadata(source),
        "measurements": {k: v for k, v in feat_with_bright.items() if not k.startswith("_")},
        "structure": struct["segments"],
        "timeline": struct["timeline"],
        "novelty": struct["novelty"],
        "mood_priors": mood_priors(feat_with_bright),
        "evidence": evidence_index(feat_with_bright, struct),
    }
    # 和弦还没跑，先给一版；cli 在 --chords 之后会重算一次。
    result["vibe"] = vibe_mod.build(result)
    return result


def to_json(result: dict, path: str | Path) -> str:
    Path(path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    return str(path)


def to_timeline_csv(result: dict, path: str | Path) -> str:
    rows = result["timeline"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t_sec", "chord_guess", "loud_db", "brightness_hz"])
        for r in rows:
            writer.writerow([r["t"], r["chord_guess"], r["loud_db"], r["brightness_hz"]])
    return str(path)


# --- 两套解读提示词 --------------------------------------------------------
# strict：拿去核对、拿去评估、拿去给严谨场景。
# vibe：给「听完想说点什么」的场景，允许比喻，不要求引用字段。

LLM_PROMPT = """你是音乐分析助手。下面是一首歌的**测量证据**（来自确定性算法，不是听力印象）。
规则：
1. 每条结论后面用方括号引用字段名，例如 [bpm]、[segments]。
2. 只使用证据里出现过的数据，不要补充证据之外的细节（乐器、歌词、年代）。
3. 情绪部分只能引用 mood_priors，并明确标注它是低置信度的启发式猜测。
4. 证据不足时直接说“无法从当前证据判断”。
5. 证据里没有任何人声或乐器判断：mid_band_ratio 只是能量分布，纯器乐也能很高，
   禁止据此推断“有人声演唱”“人声频段占满”之类结论。
"""

LLM_PROMPT_FUN = """你刚听完这首歌。下面是「一句感受 + 三个情绪词 + 一行走向」，以及几行可引用的数据
（调性、和弦、关键时间点的响度）。

规则：
1. **歌名、歌手、专辑只是文件标签，不可信**（可能是错的、被改过的、同名不同曲）。
   只依据下面给的数据说话；**不要搜索、不要调用任何联网工具、不要引用网络上的
   评论/歌词/歌手背景**——那些不是你的感受。
2. 2–4 句，总共不超过 100 字。第一句直接说感觉，像普通人；**但要有 1 处具体数据**：
   时间点（1:34 那段）、数字（最满 -20 dB）、调性（C 大调 / A 小调）或和弦（一直绕着 C-F-G）。
   数据要顺口带过，像懂行的朋友聊天，不是报参数。
3. 最多用 2 个数据点、1 个术语（和弦 / 调性 / 频段），不列清单、不写“根据测量 / 频谱显示”。
4. 情绪直接说出来（闷 / 空 / 松 / 想哭 / 上头 / 没感觉），再补具体的：
   哪一段、什么时候听的、身体反应。
5. 不许总结、不许分点、不许排比；不要写“总的来说 / 氛围 / 张力 / 层次”这类词。
6. 不确定就说「好像 / 说不上来」，不要编歌词、歌手、年代、乐器。
7. 对方要完整数据时，再切 strict 报告。
"""


LIMITS_MD = """# 已知限制（失效条件）

这份文件回答一个问题：**它什么时候是错的。**
报告里的每条结论都有对应的失效条件；拿本项目的结论去做严肃判断前，
先确认没踩到下面这些线。

## 和弦识别有误差

HPCP + 三/七和弦模板 + Viterbi + 类别先验。合成钢琴音频上根音约 74%，真实录音更低，
转位与复杂和声容易判错。

## 调性只能到关系大小调二选一

关系大调与小调共享全部调内和弦，仅凭和弦序列无法区分（例如 F 大调 / D 小调）。
要定到绝对调名，需要人耳、旋律，或结尾和弦 / 主和弦位置这类额外信息。

## 旋律提取不可用

复调遮蔽下 RPA 约 20%，本项目不做旋律。因此“像不像某个旋律”“有没有动机重复”
这类结论不要从本项目的输出里推。

## 结构检测依赖响度与音色变化

靠旋律区分的主歌 / 副歌，可能识别不出来。段落字母（A/B/C）只表示“这两段像”，
不表示“这是主歌 / 副歌”。

## 不判断有没有人声、也不识别乐器

中频带能量占比（300~3000 Hz）只是亮度描述：在本仓库 18 首**纯器乐**样本上取值为
27.7%~82.3%，与频谱重心相关 r=-0.89——纯钢琴曲同样能到 80%+。
**它不能作为人声证据**，任何“这里有人声”的说法都不来自本项目。

## 音符密度是相对起音率

不是音符数，不同编配风格之间不能直接比较。

## 情绪相关字段都不是测量

`mood_priors` 是线性启发式先验，`vibe` 是把测量值翻译成人话的映射表。
两者都只负责措辞，超出校准范围会标 `invalid` / 给兜底词。
真实情绪依赖文化、歌词、个人经历，本项目没有能力判断。

## 其他空白

无歌词转写（`--lyrics` 默认关闭且需检查 `no_speech_prob`）、无年代 / 流派判断、
无音乐语义理解。本项目不是音乐理解模型。
**全程离线**：不联网、不取评论、不调用任何第三方接口。

---

*这份文件由 `music_insight.report.LIMITS_MD` 生成，改文案请改源码。*
"""


def to_limits(path: str | Path) -> str:
    Path(path).write_text(LIMITS_MD, encoding="utf-8", newline="\n")
    return str(path)


def _mmss(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    return f"{total // 60}:{total % 60:02d}"


def _header(result: dict) -> tuple[str, str]:
    meta = result.get("meta") or {}
    title = meta.get("title") or Path(result["source"]).stem
    artist = meta.get("artist") or ""
    return title, artist


def _to_markdown_strict(result: dict, path: str | Path) -> str:
    m = result["measurements"]
    md = result["mood_priors"]
    lines = [
        f"# 音乐测量报告：{Path(result['source']).stem}",
        "",
        f"> 全部数值由确定性算法测量得到，可复现。情绪部分为启发式先验，置信度低。"
        f"失效条件见 LIMITS.md。",
        "",
        "## 1. 客观指标",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 时长 | {m['duration_sec']} 秒 |",
        f"| 速度 | {m['bpm']} BPM |",
        f"| 调性 | {m['key']}（多源一致度 {m.get('key_agreement', '-')}，间隔 {m.get('key_margin', '-')}） |",
        f"| 平均响度 | {m['loudness_db']['mean']} dB（{m['loudness_db']['min']} ~ {m['loudness_db']['max']}） |",
        f"| 峰值相对电平 | {m['crest_db']} dB |",
        f"| 音符密度 | {m['note_density_per_sec']}/s |",
        f"| 主要音级 | {' / '.join(m['mean_chroma_top3'])} |",
        f"| 中频带能量占比 | {m['mid_band_ratio']:.0%}（300~3000 Hz，音色描述，与人声无关） |",
        "",
    ]
    votes = m.get("key_votes") or []
    if votes:
        lines += ["多源调性投票（一致度 %s，大小调一致度 %s，调音偏移 %s 音分）："
                  % (m.get("key_agreement"), m.get("mode_agreement"), m.get("tuning_cents")),
                  "", "| 信息源 | 结论 | 相关度 | 候选间隔 |", "|---|---|---|---|"]
        for v in votes:
            lines.append(f"| {v['source']} | {v['key']} | {v['score']} | {v.get('margin', '-')} |")
        lines.append("")
    if (m.get("key_agreement") is not None and m["key_agreement"] < 1.0) or \
            (m.get("mode_agreement") is not None and m["mode_agreement"] < 1.0) or \
            (m.get("key_margin") is not None and m["key_margin"] < 0.06):
        lines += [
            "> ⚠️ 三个信息源没有完全一致，或候选间隔过小：这条调性结论不可靠。"
            "最常见的混淆是关系大小调（如 F 大调 / D 小调、D 大调 / B 小调）。",
            "",
        ]
    lines += [
        "## 2. 结构",
        "",
        "| 段 | 起止（秒） | 响度 dB | 频谱重心 Hz |",
        "|---|---|---|---|",
    ]
    for s in result["structure"]:
        lines.append(
            f"| {s['label']} | {s['start']} – {s['end']} | {s['loud_db']} "
            f"| {s['brightness_hz']:.0f} |"
        )
    lines += ["", "（每段的和弦见第 4 节。）"]

    lines += [
        "",
        "## 3. 时间线（8 秒粒度）",
        "",
        "| 时间 | 主要音级 | 响度 dB | 频谱重心 Hz |",
        "|---|---|---|---|",
    ]
    for r in result["timeline"]:
        lines.append(f"| {r['t']}s | {r['chord_guess']} | {r['loud_db']} | {r['brightness_hz']:.0f} |")

    if result.get("chords") and "segments" in result["chords"]:
        kfc = result.get("key_from_chords")
        lines += ["", "## 4. 和弦（节拍同步）", ""]
        if kfc:
            if kfc.get("ambiguous_relative"):
                pair = kfc.get("relative_pair") or [kfc["key"], kfc["runner_up"]]
                lines += [
                    f"- ⚠️ 关系大小调二选一：**{pair[0]}** / {pair[1]}",
                    "  两者共享全部调内和弦，仅凭和弦序列无法区分，需要人耳或旋律确认。",
                ]
            else:
                lines += [
                    f"- 由和弦反推调性：**{kfc['key']}**（margin {kfc['margin']}，"
                    f"次选 {kfc['runner_up']}）",
                ]
            lines += [f"- 高频和弦：{' / '.join(kfc['top_chords'])}", ""]
        lines += ["| 起止（秒） | 和弦 |", "|---|---|"]
        for seg in result["chords"]["segments"][:40]:
            lines.append(f"| {seg['start']} – {seg['end']} | {seg['chord']} |")
        if len(result["chords"]["segments"]) > 40:
            lines.append(f"| … | 共 {len(result['chords']['segments'])} 段 |")

    lines += [
        "",
        "## 5. 情绪先验（启发式）",
        "",
        f"- 方法：`{md['method']}`，置信度 `{md['confidence']}`"
        + ("，**输出已判为失效**" if md.get("saturated") else ""),
        f"- 唤醒度 {md['arousal']}，效价 {md['valence']} → {md['labels']}",
        f"- ⚠️ {md['caveat']}",
        "",
        "## 6. 证据索引",
        "",
        "| 结论 | 字段 | 值 |",
        "|---|---|---|",
    ]
    for e in result["evidence"]:
        lines.append(f"| {e['claim']} | `{e['field']}` | {e['value']} |")

    lines += [
        "",
        "## 7. 已知限制",
        "",
        "- **和弦识别有误差**：合成钢琴上根音约 74%，真实录音更低，转位与复杂和声容易判错。",
        "- **调性只能到关系大小调二选一**：两者共享全部调内和弦，需人耳或旋律确认。",
        "- **旋律提取不可用**：复调遮蔽下 RPA 约 20%，本项目不做旋律。",
        "- **结构检测依赖响度与音色变化**：靠旋律区分的主歌/副歌可能识别不出来。",
        "- **不判断有没有人声、也不识别乐器**：中频带能量占比只是能量分布，纯器乐同样能到 80%+。",
        "- **音符密度是相对起音率**，不是音符数。",
        "- **情绪先验是启发式**：只在校准范围内有意义，超出时已标记失效。",
        "- 完整失效条件见 LIMITS.md。",
        "",
    ]

    lines += [
        "## 8. 给解读模型的提示词（strict）",
        "",
        "```text",
        LLM_PROMPT.strip(),
        "```",
        "",
        "> 想要「听完有感觉」的版本，用 `--mode vibe` 生成另一份报告，"
        "提示词是 `LLM_PROMPT_FUN`。",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return str(path)


def _to_markdown_vibe(result: dict, path: str | Path) -> str:
    """vibe 报告：一屏看完。

    结构固定三块：感觉 → 走向 → 数据。没有表格，没有逐段列表，
    没有免责声明大礼包——那些都在 strict 报告和 LIMITS.md 里，需要时再看。
    """
    m = result["measurements"]
    v = result.get("vibe") or vibe_mod.build(result)
    title, artist = _header(result)
    dur_text = _mmss(float(m["duration_sec"]))
    byline = " · ".join(x for x in (artist, dur_text, f"{m['bpm']} BPM", m["key"]) if x)

    lines = [
        f"# 听完《{title}》",
        "",
        f"> {byline}",
        "",
        "## 感觉",
        "",
        v["headline"],
        "",
        " / ".join(v["feelings"]),
        "",
        "## 走向",
        "",
        v.get("arc_line") or "—",
        "",
        v.get("hand_line", ""),
        "",
        "## 数据",
        "",
        f"{dur_text} · {m['bpm']} BPM · {m['key']} · {m['loudness_db']['mean']} dB · "
        f"{m['note_density_per_sec']}/s · 中频 {m['mid_band_ratio']:.0%}",
        "",
    ]

    kfc = result.get("key_from_chords") or {}
    if kfc.get("top_chords"):
        amb = (f"（{kfc['key']} / {kfc['relative_pair'][1]} 二选一）"
               if kfc.get("ambiguous_relative") and kfc.get("relative_pair")
               else f"（{kfc.get('key', '?')}）")
        lines += [f"和弦：{' / '.join(kfc['top_chords'][:4])}{amb}", ""]

    lines += [
        "完整数值见 `analysis.json`；要引用请用 `--mode strict`。",
        "",
        "## 改写提示词（给模型）",
        "",
        "```text",
        LLM_PROMPT_FUN.strip(),
        "```",
        "",
        f"> `{v['method']}` · 出处与限制见 LIMITS.md",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return str(path)

REPORT_MODES = ("vibe", "strict")


def to_markdown(result: dict, path: str | Path, mode: str = "strict") -> str:
    """写 report.md。

    ``mode="strict"``：证据清单，可引用、可核对（默认，向后兼容）。
    ``mode="vibe"``：先讲听感、数字压后；仍是同一份 measurements。
    """
    if mode not in REPORT_MODES:
        raise ValueError(f"unknown report mode: {mode!r}（可选 {REPORT_MODES}）")
    if mode == "vibe":
        return _to_markdown_vibe(result, path)
    return _to_markdown_strict(result, path)
