"""证据层 + 报告：把测量结果组织成 JSON / CSV / Markdown。

设计原则：报告里每一条事实性结论都必须能指回一个计算出来的字段。
情绪部分单独标成 heuristic，绝不给它和测量值同等的地位。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from . import features as F


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
        ("人声频段占比", "vocal_band_ratio", f"{feat['vocal_band_ratio']:.0%}（弱证据）"),
        ("主要音级", "mean_chroma_top3", " / ".join(feat["mean_chroma_top3"])),
        ("段落数", "segments", f"{len(struct['segments'])} 段"),
    ]
    return [{"claim": c, "field": f, "value": v} for c, f, v in items]


def build(feat: dict, struct: dict, source: str, image: str | None = None) -> dict:
    brightness = struct["timeline"][0]["brightness_hz"] if struct["timeline"] else 1000
    if struct["timeline"]:
        brightness = sum(t["brightness_hz"] for t in struct["timeline"]) / len(struct["timeline"])
    feat_with_bright = {**feat, "brightness_mean_hz": brightness}

    return {
        "source": source,
        "image": image,
        "measurements": {k: v for k, v in feat_with_bright.items() if not k.startswith("_")},
        "structure": struct["segments"],
        "timeline": struct["timeline"],
        "novelty": struct["novelty"],
        "mood_priors": mood_priors(feat_with_bright),
        "evidence": evidence_index(feat_with_bright, struct),
    }


def to_json(result: dict, path: str | Path) -> str:
    Path(path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def to_timeline_csv(result: dict, path: str | Path) -> str:
    rows = result["timeline"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t_sec", "chord_guess", "loud_db", "brightness_hz"])
        for r in rows:
            writer.writerow([r["t"], r["chord_guess"], r["loud_db"], r["brightness_hz"]])
    return str(path)


LLM_PROMPT = """你是音乐分析助手。下面是一首歌的**测量证据**（来自确定性算法，不是听力印象）。
规则：
1. 每条结论后面用方括号引用字段名，例如 [bpm]、[segments]。
2. 只使用证据里出现过的数据，不要补充证据之外的细节（乐器、歌词、年代）。
3. 情绪部分只能引用 mood_priors，并明确标注它是低置信度的启发式猜测。
4. 证据不足时直接说“无法从当前证据判断”。
"""


def to_markdown(result: dict, path: str | Path) -> str:
    m = result["measurements"]
    md = result["mood_priors"]
    lines = [
        f"# 音乐测量报告：{Path(result['source']).stem}",
        "",
        f"> 全部数值由确定性算法测量得到，可复现。情绪部分为启发式先验，置信度低。",
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
        f"| 人声频段占比 | {m['vocal_band_ratio']:.0%}（弱证据） |",
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
        "- **和弦识别有误差**：HPCP + 三/七和弦模板 + Viterbi + 类别先验；"
        "合成钢琴音频上根音约 74%，真实录音更低，转位与复杂和声容易判错。",
        "- **调性只能到关系大小调二选一**：两者共享全部调内和弦，需人耳或旋律确认。",
        "- **旋律提取不可用**：复调遮蔽下 RPA 约 20%，本项目不做旋律。",
        "- **结构检测依赖响度与音色变化**：靠旋律区分的主歌/副歌可能识别不出来。",
        "- **人声频段占比是弱证据**：乐器也占这个频段。",
        "- **音符密度是相对起音率**，不是音符数；不同编配风格之间不能直接比较。",
        "- **情绪先验是启发式**：只在校准范围内有意义，超出时已标记失效，见第 5 节。",
        "- 无歌词、无乐器识别、无年代/流派判断。",
        "",
        "## 8. 给解读模型的提示词",
        "",
        "```text",
        LLM_PROMPT.strip(),
        "```",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return str(path)
