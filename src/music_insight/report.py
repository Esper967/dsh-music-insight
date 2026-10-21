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


def mood_priors(feat: dict) -> dict:
    """**不是测量**。基于公开研究里常见的统计先验，把客观量线性映射到唤醒/效价。

    任何使用方都应把它当“猜测”，置信度低，且不能用于情绪标注或心理评估。
    """
    bpm = feat["bpm"]
    loud = feat["loudness_db"]["mean"]
    density = feat["note_density_per_sec"]
    brightness = feat.get("brightness_mean_hz", 1000.0)
    major = feat["key"].endswith("大调")

    arousal = _clip(0.5 * (bpm - 60) / 70 + 0.3 * (loud + 30) / 22 + 0.2 * density / 12)
    valence = _clip(0.35 * (brightness - 700) / 900 + 0.3 * (1.0 if major else 0.15)
                    + 0.35 * (bpm - 60) / 70)
    return {
        "method": "heuristic-v0",
        "confidence": "low",
        "arousal": round(arousal, 2),
        "valence": round(valence, 2),
        "labels": _quadrant(arousal, valence),
        "caveat": "线性先验映射，未经验证；小调+快节奏、亮失真吉他等都会打破这个映射。"
                  "不要把它当测量值使用。",
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
        f"| 调性 | {m['key']}（相关 {m['key_correlation']}） |",
        f"| 平均响度 | {m['loudness_db']['mean']} dB（{m['loudness_db']['min']} ~ {m['loudness_db']['max']}） |",
        f"| 峰值相对电平 | {m['crest_db']} dB |",
        f"| 音符密度 | {m['note_density_per_sec']}/s |",
        f"| 主要音级 | {' / '.join(m['mean_chroma_top3'])} |",
        f"| 人声频段占比 | {m['vocal_band_ratio']:.0%}（弱证据） |",
        "",
        "## 2. 结构",
        "",
        "| 段 | 起止（秒） | 响度 dB | 频谱重心 Hz | 和弦猜测 |",
        "|---|---|---|---|---|",
    ]
    for s in result["structure"]:
        lines.append(
            f"| {s['label']} | {s['start']} – {s['end']} | {s['loud_db']} "
            f"| {s['brightness_hz']:.0f} | {s['chord_guess']} |"
        )

    lines += [
        "",
        "## 3. 时间线（8 秒粒度）",
        "",
        "| 时间 | 和弦猜测 | 响度 dB | 频谱重心 Hz |",
        "|---|---|---|---|",
    ]
    for r in result["timeline"]:
        lines.append(f"| {r['t']}s | {r['chord_guess']} | {r['loud_db']} | {r['brightness_hz']:.0f} |")

    lines += [
        "",
        "## 4. 情绪先验（启发式，低置信度）",
        "",
        f"- 方法：`{md['method']}`，置信度 `{md['confidence']}`",
        f"- 唤醒度 {md['arousal']}，效价 {md['valence']} → {md['labels']}",
        f"- ⚠️ {md['caveat']}",
        "",
        "## 5. 证据索引",
        "",
        "| 结论 | 字段 | 值 |",
        "|---|---|---|",
    ]
    for e in result["evidence"]:
        lines.append(f"| {e['claim']} | `{e['field']}` | {e['value']} |")

    lines += [
        "",
        "## 6. 已知限制",
        "",
        "- **和弦猜测不可靠**：用 2 秒块音级能量猜三和弦，不分转位、不分大小调色彩。",
        "- **结构检测依赖响度与音色变化**：没有做人声/伴奏分离时，靠旋律区分的主歌/副歌可能识别不出来。",
        "- **人声频段占比是弱证据**：乐器也占这个频段。",
        "- **情绪是猜测**：见第 4 节。",
        "- 无歌词、无乐器识别、无年代/流派判断。",
        "",
        "## 7. 给解读模型的提示词",
        "",
        "```text",
        LLM_PROMPT.strip(),
        "```",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return str(path)
