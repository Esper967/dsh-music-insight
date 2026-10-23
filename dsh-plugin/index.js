/**
 * music_insight — 听完一首歌的「感官底稿」。
 *
 * The measurement code lives in a separate Python checkout (`repoRoot`), so this
 * Host plugin stays a thin wrapper: resolve the file against the Session working
 * directory, run the CLI once, then return a compact summary plus the paths of
 * the full report and the measurement panel.
 *
 * 两种 mode 决定「返回什么给模型」：
 *   vibe（默认）—— 骨架 + 听感 + 元信息，模型可以直接接着说感受；
 *   strict      —— 数字证据包，给核对/评估用。
 *
 * This package is installed as a linked local directory, so it must not import
 * `@deepseek-ai/*` packages — plain tool objects are the supported form here
 * (same pattern as the profile's dsh-local-llm plugin).
 */
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const name = "music-insight";
export const inject = ["tools", "subprocess", "workingDirectory"];

const DEFAULTS = {
  pythonPath: "",
  repoRoot: "",
  timeoutMs: 180000,
  defaultChords: true,
  defaultMode: "vibe",
  outputSuffix: "-分析",
};

function renderText(_args, value) {
  const text = typeof value === "string" ? value : value?.summary ?? JSON.stringify(value);
  return [{ type: "text", text }];
}

function fmtTime(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

/** strict：老的行为——紧凑的数字证据包。 */
function summarizeStrict(analysis, outDir) {
  const m = analysis.measurements ?? {};
  const lines = [`音频：${analysis.source ?? "?"}`];
  lines.push(`${m.duration_sec ?? "?"} 秒 · ${m.bpm ?? "?"} BPM · ${m.key ?? "?"}` +
    (m.loudness_db ? ` · 平均 ${m.loudness_db.mean} dB` : ""));

  const kfc = analysis.key_from_chords;
  if (kfc) {
    const pair = kfc.ambiguous_relative
      ? `（${kfc.relative_pair[0]} / ${kfc.relative_pair[1]} 二选一）`
      : "";
    lines.push(`由和弦反推调性：${kfc.key}${pair}`);
    if (kfc.top_chords?.length) lines.push(`高频和弦：${kfc.top_chords.join(" / ")}`);
  } else if (m.mean_chroma_top3?.length) {
    lines.push(`主要音级：${m.mean_chroma_top3.join(" / ")}`);
  }

  if (analysis.chords?.n_segments) lines.push(`和弦段数：${analysis.chords.n_segments}`);
  if (outDir) lines.push(`输出目录：${outDir}`);
  return lines.join("\n");
}

/**
 * vibe：模型拿到就能说感受的底稿。
 *
 * 顺序是刻意的——先元信息、再一句话、再骨架、最后才是数字。
 * 模型读到的第一样东西决定它的语气。
 */
function summarizeVibe(analysis, outDir) {
  const m = analysis.measurements ?? {};
  const meta = analysis.meta ?? {};
  const v = analysis.vibe ?? {};
  const dur = Number(m.duration_sec) || 0;
  const durText = dur < 60
    ? `${Math.round(dur)} 秒`
    : `${Math.floor(dur / 60)} 分 ${Math.round(dur % 60)} 秒`;
  const facts = [durText, `${m.bpm ?? "?"} BPM`, m.key ?? "?"];
  if (meta.album) facts.push(`专辑 ${meta.album}`);
  if (meta.year) facts.push(meta.year);
  if (meta.genre) facts.push(`流派标签 ${meta.genre}`);

  const lines = [];
  if (meta.title) {
    lines.push(meta.artist ? `《${meta.title}》— ${meta.artist}` : `《${meta.title}》`);
  }
  lines.push(facts.join(" · "));
  if (v.one_liner) lines.push(`一句话：${v.one_liner}`);

  if (v.arc?.length) {
    lines.push("");
    lines.push("它怎么走的：");
    for (const seg of v.arc) {
      const t = `${fmtTime(seg.start)}–${fmtTime(seg.end)}`;
      lines.push(`  ${t.padEnd(11)} ${seg.phase ?? ""}  ${seg.loud_db}dB  ` +
        `${Math.round(seg.brightness_hz)}Hz  ${seg.word ?? ""}`);
    }
    if (v.grain === "timeline") lines.push("  （结构检测没抓到边界，按 8 秒时间线讲）");
  }

  if (v.hand) {
    lines.push("");
    lines.push(`手感：速度 ${v.hand.tempo} ｜ 明暗 ${v.hand.light} ｜ ` +
      `密度 ${v.hand.density} ｜ 温度 ${v.hand.warmth} ｜ 长度 ${v.hand.length}`);
  }
  if (v.tags?.length) lines.push(`标签：${v.tags.join(" / ")}`);

  if (v.anchors?.length) {
    lines.push("");
    lines.push("可以拿去用的锚点（带数字，说的是感觉）：");
    for (const a of v.anchors) lines.push(`  · ${a}`);
  }

  // 和弦：给一句，别把时间线全倒给模型。
  const kfc = analysis.key_from_chords;
  lines.push("");
  if (kfc?.top_chords?.length) {
    lines.push(`和弦：${kfc.top_chords.join(" / ")}` +
      (kfc.ambiguous_relative ? `（调性 ${kfc.key} / ${kfc.relative_pair?.[1] ?? "?"} 二选一）` : `（${kfc.key}）`));
  } else if (m.mean_chroma_top3?.length) {
    lines.push(`主要音级：${m.mean_chroma_top3.join(" / ")}`);
  }
  if (analysis.chords?.n_segments) lines.push(`和弦段数：${analysis.chords.n_segments}`);

  if (outDir) lines.push(`输出目录：${outDir}`);
  return lines.join("\n");
}

/**
 * 选一种摘要风格。导出是为了能脱离 DSH 直接测：
 *
 *   node --input-type=module -e "import {summarizeAnalysis} from './index.js'; ..."
 */
export function summarizeAnalysis(analysis, outDir, mode = "vibe") {
  return mode === "strict"
    ? summarizeStrict(analysis, outDir)
    : summarizeVibe(analysis, outDir);
}

/**
 * Resolve the output directory for one run.
 *
 * Priority: explicit `output_dir` argument (resolved against the Session cwd) →
 * `<audio name>-分析` beside the audio file → a fresh temp directory as fallback
 * when the audio's directory is not writable.
 */
async function resolveOutDir(target, requested, cwd, suffix) {
  if (requested) {
    const dir = path.isAbsolute(requested) ? requested : path.resolve(cwd, requested);
    await mkdir(dir, { recursive: true });
    return { dir, cleanup: false };
  }
  const beside = path.join(path.dirname(target), `${path.parse(target).name}${suffix}`);
  try {
    await mkdir(beside, { recursive: true });
    return { dir: beside, cleanup: false };
  } catch {
    const dir = await mkdtemp(path.join(tmpdir(), "music-insight-"));
    return { dir, cleanup: true };
  }
}

export function apply(ctx, rawConfig) {
  const config = { ...DEFAULTS, ...(rawConfig ?? {}) };
  // 插件放在仓库内（<repo>/dsh-plugin）时，repoRoot 默认取上一层；
  // 单独安装时可以显式配置。
  if (!config.repoRoot) {
    config.repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  }
  if (!config.repoRoot) {
    throw new Error("dsh-music-insight: config.repoRoot must point at the music-insight checkout");
  }
  if (!existsSync(path.join(config.repoRoot, "src", "music_insight", "cli.py"))) {
    throw new Error(`dsh-music-insight: repoRoot has no src/music_insight/cli.py: ${config.repoRoot}`);
  }
  const python = config.pythonPath || "python";

  const tool = {
    name: "music_insight",
    description:
      "听完一首歌的「感官底稿」：把本地音频测一遍（时长、BPM、调性、响度起伏、明暗、结构，可选和弦），" +
      "再翻译成骨架和听感——哪里薄、哪里涨满、怎么散掉，附上歌名/歌手/专辑。 " +
      "拿到结果后请**先用第一人称说你听完的感受**（可以比喻、可以联想歌名和画面），" +
      "数字是给你撑感觉用的，不必逐个报、更不要说“根据测量数据”。 " +
      "同时写出 report.md / panel.png / LIMITS.md 等文件并返回路径。 " +
      "mode=\"strict\" 返回可引用的证据清单（核对/评估时用），默认 vibe。 " +
      "这是确定性测量，不是音乐理解模型：旋律、人声、乐器都不在证据里，要猜请标明是猜。" +
      "Runs offline on the Host; WAV decodes with the standard library, and mp3/m4a/flac need PyAV in the configured Python.",
    parameters: {
      type: "object",
      properties: {
        file_path: {
          type: "string",
          description: "Audio file path. Relative paths resolve against the current Session working directory.",
        },
        mode: {
          type: "string",
          enum: ["vibe", "strict"],
          description:
            "vibe（默认）：先说听感、数字压后，给“听完想说点什么”的场景。" +
            "strict：可引用的证据清单，给核对/评估用。",
        },
        chords: {
          type: "boolean",
          description: "Run beat tracking + chord recognition and infer the key from chords. Defaults to the plugin config (true).",
        },
        output_dir: {
          type: "string",
          description:
            "Optional output directory for report.md / analysis.json / timeline.csv / panel.png / LIMITS.md. " +
            "Relative paths resolve against the Session working directory; existing files there are overwritten. " +
            "Omit to use '<audio name>-分析' beside the audio file.",
        },
      },
      required: ["file_path"],
    },
    output: { schema: {}, render: renderText },
    execute: async (args, exec) => {
      const filePath = typeof args?.file_path === "string" ? args.file_path.trim() : "";
      if (!filePath) throw new Error("music_insight: file_path must be a non-empty string");

      const cwd = exec?.agent
        ? await ctx.workingDirectory.ensure(exec.agent, exec.signal)
        : process.cwd();
      const target = path.isAbsolute(filePath) ? filePath : path.resolve(cwd, filePath);
      if (!existsSync(target)) throw new Error(`music_insight: file not found: ${target}`);

      const requestedMode = args?.mode ?? config.defaultMode;
      const mode = requestedMode === "strict" ? "strict" : "vibe";
      const withChords = args?.chords ?? config.defaultChords;
      const requestedDir = typeof args?.output_dir === "string" ? args.output_dir.trim() : "";
      const { dir: outDir, cleanup } = await resolveOutDir(target, requestedDir, cwd, config.outputSuffix);

      const argv = [python, "-m", "music_insight.cli", target, "--out", outDir, "--quiet", "--mode", mode];
      if (withChords) argv.push("--chords");

      let outcome;
      let stdout = "";
      let stderr = "";
      try {
        const handle = ctx.subprocess.spawn({
          argv,
          cwd: config.repoRoot,
          stdio: {
            stdin: "ignore",
            stdout: { maxBytes: 2_000_000 },
            stderr: { maxBytes: 2_000_000 },
          },
          graceMs: 5000,
          signal: AbortSignal.any([exec.signal, AbortSignal.timeout(config.timeoutMs)]),
          env: {
            ...process.env,
            PYTHONPATH: path.join(config.repoRoot, "src"),
            PYTHONIOENCODING: "utf-8",
          },
        });
        outcome = await handle.done;
        stdout = handle.collected.stdout?.readFrom(0)?.text ?? "";
        stderr = handle.collected.stderr?.readFrom(0)?.text ?? "";
      } catch (error) {
        if (cleanup) await rm(outDir, { recursive: true, force: true }).catch(() => {});
        throw new Error(`music_insight: failed to start the analysis CLI: ${error?.message ?? error}`);
      }

      if (outcome.exitCode !== 0) {
        if (cleanup) {
          const left = await readdir(outDir).catch(() => []);
          if (left.length === 0) await rm(outDir, { recursive: true, force: true }).catch(() => {});
        }
        const tail = (stderr || stdout).trim().split("\n").slice(-6).join("\n");
        throw new Error(`music_insight: analysis failed (exit ${outcome.exitCode}): ${tail}`);
      }

      const analysisPath = path.join(outDir, "analysis.json");
      const reportPath = path.join(outDir, "report.md");
      const panelPath = path.join(outDir, "panel.png");
      const analysis = JSON.parse(await readFile(analysisPath, "utf8"));
      const summary = summarizeAnalysis(analysis, outDir, mode);
      return {
        summary,
        mode,
        output_dir: outDir,
        report_path: reportPath,
        analysis_path: analysisPath,
        ...(analysis.meta?.title ? { title: analysis.meta.title } : {}),
        ...(analysis.vibe ? { vibe: analysis.vibe } : {}),
        ...(existsSync(panelPath) ? { panel_path: panelPath } : {}),
      };
    },
  };

  ctx.effect(() => ctx.tools.register(tool));
}
