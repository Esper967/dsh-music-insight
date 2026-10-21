/**
 * music_insight — analyze one local audio file into a bounded evidence pack.
 *
 * The measurement code lives in a separate Python checkout (`repoRoot`), so this
 * Host plugin stays a thin wrapper: resolve the file against the Session working
 * directory, run the CLI once, then return a compact summary plus the paths of
 * the full report and the measurement panel.
 *
 * This package is installed as a linked local directory, so it must not import
 * `@deepseek-ai/*` packages — plain tool objects are the supported form here
 * (same pattern as the profile's dsh-local-llm plugin).
 */
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
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
};

function renderText(_args, value) {
  const text = typeof value === "string" ? value : value?.summary ?? JSON.stringify(value);
  return [{ type: "text", text }];
}

function summarize(analysis) {
  const m = analysis.measurements ?? {};
  const head = [`音频：${analysis.source ?? "?"}`];
  head.push(`${m.duration_sec ?? "?"} 秒 · ${m.bpm ?? "?"} BPM · ${m.key ?? "?"}` +
    (m.loudness_db ? ` · 平均 ${m.loudness_db.mean} dB` : ""));
  const lines = head;

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

  const chords = analysis.chords;
  if (chords?.n_segments) lines.push(`和弦段数：${chords.n_segments}`);
  return lines.join("\n");
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
      "Analyze one local audio file with the local music-insight pipeline and return a compact evidence pack: " +
      "duration, BPM, key, loudness and structure, plus (with chords) a beat-synchronised chord timeline whose key is inferred from the chord sequence. " +
      "Writes a full Markdown report and a measurement panel image next to the analysis and returns their paths. " +
      "This is a deterministic measurement tool, not a music-understanding model: melody extraction is not supported. " +
      "Runs offline on the Host; WAV decodes with the standard library, and mp3/m4a/flac need PyAV in the configured Python.",
    parameters: {
      type: "object",
      properties: {
        file_path: {
          type: "string",
          description: "Audio file path. Relative paths resolve against the current Session working directory.",
        },
        chords: {
          type: "boolean",
          description: "Run beat tracking + chord recognition and infer the key from chords. Defaults to the plugin config (true).",
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

      const withChords = args?.chords ?? config.defaultChords;
      const outDir = await mkdtemp(path.join(tmpdir(), "music-insight-"));

      const argv = [python, "-m", "music_insight.cli", target, "--out", outDir, "--quiet"];
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
        await rm(outDir, { recursive: true, force: true }).catch(() => {});
        throw new Error(`music_insight: failed to start the analysis CLI: ${error?.message ?? error}`);
      }

      if (outcome.exitCode !== 0) {
        await rm(outDir, { recursive: true, force: true }).catch(() => {});
        const tail = (stderr || stdout).trim().split("\n").slice(-6).join("\n");
        throw new Error(`music_insight: analysis failed (exit ${outcome.exitCode}): ${tail}`);
      }

      const analysisPath = path.join(outDir, "analysis.json");
      const reportPath = path.join(outDir, "report.md");
      const panelPath = path.join(outDir, "panel.png");
      const analysis = JSON.parse(await readFile(analysisPath, "utf8"));
      return {
        summary: `${summarize(analysis)}\n报告：${reportPath}`,
        report_path: reportPath,
        analysis_path: analysisPath,
        ...(existsSync(panelPath) ? { panel_path: panelPath } : {}),
      };
    },
  };

  ctx.effect(() => ctx.tools.register(tool));
}
