#!/usr/bin/env node
/**
 * 预览插件会返回给模型的摘要，不用启动 DSH。
 *
 * 改完 `index.js` 里的措辞后跑一下，确认模型读到的到底是哪句话：
 *
 *   node dsh-plugin/preview-summary.mjs <一次分析产出的 analysis.json> [vibe|strict]
 *
 * 例：
 *   music-insight song.mp3 --out out/song
 *   node dsh-plugin/preview-summary.mjs out/song/analysis.json vibe
 */
import { readFileSync } from "node:fs";

import { summarizeAnalysis } from "./index.js";

const [analysisPath, mode = "vibe"] = process.argv.slice(2);
if (!analysisPath) {
  console.error("usage: node dsh-plugin/preview-summary.mjs <analysis.json> [vibe|strict]");
  process.exit(2);
}

const analysis = JSON.parse(readFileSync(analysisPath, "utf8"));
// 输出目录只是摘要末尾的一行提示，预览时用占位符即可。
console.log(summarizeAnalysis(analysis, "<output_dir>", mode));
