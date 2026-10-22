/**
 * Rank open-weight models for local inference on this machine.
 *
 *   bun src/cli.ts [--min-tps 20] [--ctx 32768] [--top 15]
 *     [--mem-gb N] [--bandwidth GBps] [--efficiency 0.4] [--authors a,b]
 *     [--scores scores.json] [--json]
 *
 * Steps: hardware budget → HF candidates → fit filter → bandwidth speed
 * estimate → Artificial Analysis score → rank. Estimates only; benchmark the
 * top picks with llama-bench / mlx_lm before trusting them.
 *
 * Env: ARTIFICIAL_ANALYSIS_API_KEY (scores; without it ranking falls back to a
 * labeled size proxy), HF_TOKEN (optional, lets gated configs load).
 * --scores: JSON {"org/repo": number} that overrides/extends AA scores.
 * --efficiency: measured tok/s ÷ (bandwidth ÷ active bytes). On an M4 Max
 * (546 GB/s), llama-bench tg128 at Q4_K_M gave 0.31 for Qwen3.5-35B-A3B (92
 * tok/s) and 0.45 for gemma-4-26B-A4B (101 tok/s); small-active MoE models pay
 * fixed per-token overhead. Recalibrate per machine with llama-bench.
 */
import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";
import {
  DEFAULT_AUTHORS, appleBandwidth, filterCandidates, parseAaModels, plausible, rankModels,
  type AaEntry, type Candidate, type Hardware, type Options, type SnapshotModel,
} from "./core";

function sh(cmd: string): string {
  try { return execSync(cmd, { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim(); } catch { return ""; }
}

function detectHardware(args: Map<string, string>): Hardware {
  const ramBytes = Number(sh("sysctl -n hw.memsize")) || 0;
  const ramGB = ramBytes / 1e9;
  const chip = sh("sysctl -n machdep.cpu.brand_string");
  const gpuCores = Number(/Total Number of Cores:\s*(\d+)/.exec(sh("system_profiler SPDisplaysDataType"))?.[1] ?? 0);
  const wiredMB = Number(sh("sysctl -n iogpu.wired_limit_mb")) || 0;
  // macOS default GPU cap: ~2/3 of RAM up to 36 GB, ~3/4 above.
  const memBudgetGB = Number(args.get("mem-gb")) || (wiredMB > 0 ? (wiredMB * 1048576) / 1e9 : ramGB * (ramGB <= 36 ? 2 / 3 : 0.75));
  const bandwidthGBs = Number(args.get("bandwidth")) || appleBandwidth(chip, gpuCores);
  if (!memBudgetGB || !bandwidthGBs) throw new Error(`Unknown hardware (${chip || "non-macOS"}): pass --mem-gb and --bandwidth`);
  return { memBudgetGB, bandwidthGBs, label: `${chip} ${gpuCores}-core GPU, ${(ramBytes / 2 ** 30).toFixed(0)} GiB RAM` };
}

const hfHeaders: Record<string, string> = process.env.HF_TOKEN ? { authorization: `Bearer ${process.env.HF_TOKEN}` } : {};
async function getJson(url: string, headers = hfHeaders): Promise<any> {
  const r = await fetch(url, { headers });
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
}

async function pool<T, R>(items: T[], n: number, fn: (x: T) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length);
  let i = 0;
  await Promise.all(Array.from({ length: n }, async () => { while (i < items.length) { const j = i++; out[j] = await fn(items[j]); } }));
  return out;
}

async function listCandidates(authors: string[]): Promise<Candidate[]> {
  const urls = authors.flatMap((a) => ["text-generation", "image-text-to-text"].flatMap((p) =>
    ["sort=downloads", "sort=createdAt&direction=-1"].map((s) =>
      `https://huggingface.co/api/models?author=${a}&pipeline_tag=${p}&${s}&limit=40&expand[]=safetensors&expand[]=tags&expand[]=createdAt`)));
  return filterCandidates(await pool(urls, 8, (u) => getJson(u).catch(() => [])));
}

async function aaScores(): Promise<AaEntry[]> {
  const key = process.env.ARTIFICIAL_ANALYSIS_API_KEY;
  if (!key) return [];
  return parseAaModels(await getJson("https://artificialanalysis.ai/api/v2/data/llms/models", { "x-api-key": key }));
}

function parseArgs(argv: string[]): Map<string, string> {
  const m = new Map<string, string>();
  for (let i = 0; i < argv.length; i++) {
    if (!argv[i].startsWith("--")) continue;
    const k = argv[i].slice(2);
    if (argv[i + 1] && !argv[i + 1].startsWith("--")) m.set(k, argv[++i]); else m.set(k, "true");
  }
  return m;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const hw = detectHardware(args);
  const o: Options = { ctx: Number(args.get("ctx") ?? 32768), minTps: Number(args.get("min-tps") ?? 20), efficiency: Number(args.get("efficiency") ?? 0.4) };
  const top = Number(args.get("top") ?? 15);
  const authors = args.get("authors")?.split(",") ?? DEFAULT_AUTHORS;
  const overrides: Record<string, number> = args.has("scores") ? JSON.parse(readFileSync(args.get("scores")!, "utf8")) : {};

  const [cands, aa] = await Promise.all([listCandidates(authors), aaScores().catch((e) => { console.error(`AA: ${e.message}`); return []; })]);
  // Cheapest quant must fit before we spend a config fetch on it.
  const fits = cands.filter((c) => plausible(c.total, hw));
  // "gated" = HTTP 401/403: accept the model's license on its HF page (HF_TOKEN alone is not enough).
  let gated = 0, noConfig = 0;
  const loaded = (await pool(fits, 8, async (c): Promise<SnapshotModel | null> => {
    const cfg = await getJson(`https://huggingface.co/${c.id}/resolve/main/config.json`).catch((e: Error) => e);
    if (cfg instanceof Error) { /^40[13] /.test(cfg.message) ? gated++ : noConfig++; return null; }
    return { ...c, cfg };
  })).filter((m) => m !== null);

  const { rows, rejects: r } = rankModels(loaded, aa, hw, o, overrides);
  const rejects = { "too big": cands.length - fits.length + r["too big"], "too slow": r["too slow"], gated, "no config": noConfig };
  const shown = rows.slice(0, top);

  if (args.has("json")) { console.log(JSON.stringify({ hw, options: o, rejects, models: shown }, null, 2)); return; }
  console.log(`${hw.label}\nbudget ${hw.memBudgetGB.toFixed(0)} GB · ${hw.bandwidthGBs} GB/s × ${o.efficiency} · ctx ${o.ctx} · floor ${o.minTps} tok/s`);
  console.log(`${cands.length} candidates from ${authors.length} orgs · rejected: ${Object.entries(rejects).map(([k, v]) => `${v} ${k}`).join(", ")}`);
  if (!aa.length) console.log("No ARTIFICIAL_ANALYSIS_API_KEY: rows without --scores are ordered by size, NOT intelligence.");
  console.log();
  const fmt = (n: number) => (n / 1e9 >= 10 ? (n / 1e9).toFixed(0) : (n / 1e9).toFixed(1)) + "B";
  console.table(shown.map((r) => ({
    model: r.id, params: `${fmt(r.total)} (${fmt(r.active)} act)`, quant: r.quant.name,
    "mem GB": +r.memGB.toFixed(1), "est tok/s": Math.round(r.tps),
    score: r.aa ? `${r.aa.score.toFixed(1)} (${r.aa.name})` : "—", released: r.created,
  })));
}

if (import.meta.main) main().catch((e) => { console.error(e.message); process.exit(1); });
