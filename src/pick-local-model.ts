/**
 * Rank open-weight models for local inference on this machine.
 *
 *   bun src/pick-local-model.ts [--min-tps 20] [--ctx 32768] [--top 15]
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

// ---------- pure estimation (tested in pick-local-model.test.ts) ----------

/** Effective bits per weight for common llama.cpp quants, best first. */
export const QUANTS = [
  { name: "Q8_0", bpw: 8.5, quality: 1.0 },
  { name: "Q6_K", bpw: 6.56, quality: 0.995 },
  { name: "Q5_K_M", bpw: 5.69, quality: 0.99 },
  { name: "Q4_K_M", bpw: 4.85, quality: 0.97 },
  { name: "Q3_K_M", bpw: 3.91, quality: 0.9 },
] as const;
export type Quant = (typeof QUANTS)[number];

export type ModelConfig = Record<string, any>;

/** The decoder config; multimodal repos nest it. */
export function textConfig(cfg: ModelConfig): ModelConfig {
  return { ...cfg, ...(cfg.text_config ?? cfg.llm_config ?? {}) };
}

/**
 * Parameters read from memory per generated token. The "-A3B" name suffix
 * wins when present (labs state it directly); otherwise derive it from the
 * MoE fields in config.json. Dense models return total.
 */
export function activeParams(total: number, rawCfg: ModelConfig, name: string): number {
  const m = /[-_]A(\d+(?:\.\d+)?)B\b/i.exec(name);
  if (m) return Number(m[1]) * 1e9;
  const c = textConfig(rawCfg);
  const nExp = c.num_experts ?? c.n_routed_experts ?? c.num_local_experts ?? 0;
  const k = c.num_experts_per_tok ?? c.moe_topk ?? c.top_k_experts ?? 0;
  if (nExp <= 1 || !k || !c.hidden_size || !c.num_hidden_layers) return total;
  const expertInter = c.moe_intermediate_size ?? c.intermediate_size;
  const moeLayers = (c.num_hidden_layers - (c.first_k_dense_replace ?? 0)) / (c.decoder_sparse_step ?? 1);
  const inactive = moeLayers * (nExp - k) * 3 * c.hidden_size * expertInter;
  return Math.max(total - inactive, total * 0.02);
}

/**
 * fp16 KV-cache bytes for `ctx` tokens. Counts only attention layers;
 * sliding-window layers cap at the window; linear/conv layers add nothing.
 * MLA (kv_lora_rank) stores the compressed latent.
 */
export function kvCacheBytes(rawCfg: ModelConfig, ctx: number): number {
  const c = textConfig(rawCfg);
  const layers = c.num_hidden_layers ?? 0;
  if (c.kv_lora_rank) return layers * (c.kv_lora_rank + (c.qk_rope_head_dim ?? 64)) * 2 * ctx;
  const heads = c.num_attention_heads ?? 1;
  const headDim = c.head_dim ?? (c.hidden_size ?? 0) / heads;
  const perLayerPerTok = 2 * (c.num_key_value_heads ?? heads) * headDim * 2;
  const types: string[] | undefined = c.layer_types;
  if (!types) return layers * perLayerPerTok * Math.min(ctx, c.sliding_window && c.use_sliding_window ? c.sliding_window : ctx);
  let toks = 0;
  for (const t of types) {
    if (t === "full_attention" || t === "attention") toks += ctx;
    else if (t.includes("sliding")) toks += Math.min(ctx, c.sliding_window ?? ctx);
  }
  return toks * perLayerPerTok;
}

export interface Hardware { memBudgetGB: number; bandwidthGBs: number; label: string }
export interface Options { ctx: number; minTps: number; efficiency: number }
export interface Plan { quant: Quant; memGB: number; tps: number }

/**
 * Highest-precision quant that fits the memory budget and clears the speed
 * floor. Decode is bandwidth-bound: each token reads the active weights plus
 * the KV cache once.
 */
export function bestPlan(total: number, active: number, kvBytes: number, hw: Hardware, o: Options): Plan | { reject: "too big" | "too slow" } {
  let fitAny = false;
  for (const quant of QUANTS) {
    const weights = (total * quant.bpw) / 8;
    const memGB = (weights * 1.05 + kvBytes) / 1e9 + 1;
    if (memGB > hw.memBudgetGB) continue;
    fitAny = true;
    const bytesPerTok = (active * quant.bpw) / 8 + kvBytes;
    const tps = (hw.bandwidthGBs * 1e9 * o.efficiency) / bytesPerTok;
    if (tps >= o.minTps) return { quant, memGB, tps };
  }
  return { reject: fitAny ? "too slow" : "too big" };
}

const SUFFIX = new Set(["instruct", "it", "chat", "hf", "base", "thinking", "reasoning", "non", "preview", "fp8", "bf16"]);
/** Name tokens for fuzzy matching HF repos to AA model slugs. */
export function nameTokens(s: string): string[] {
  const base = s.split("/").pop()!.toLowerCase().replace(/[()]/g, " ");
  return base.replace(/(\d)\.(\d)/g, "$1p$2").split(/[^a-z0-9]+/).filter((t) => t && !SUFFIX.has(t));
}

/**
 * Best AA entry for an HF repo: the size tokens (e.g. "30b", "a3b") must
 * match exactly and all remaining tokens must overlap ≥ 0.75 (Jaccard).
 * Variants (reasoning / non-reasoning) collapse to the max score.
 */
export function matchScore(repo: string, aa: { name: string; score: number }[]): { name: string; score: number } | undefined {
  const t = new Set(nameTokens(repo));
  const sizes = (s: Set<string>) => [...s].filter((x) => /^a?\d+(p\d+)?[bmt]$/.test(x)).sort().join(",");
  let best: { name: string; score: number; sim: number } | undefined;
  for (const e of aa) {
    const u = new Set(nameTokens(e.name));
    if (sizes(t) !== sizes(u)) continue;
    const inter = [...t].filter((x) => u.has(x)).length;
    const sim = inter / new Set([...t, ...u]).size;
    if (sim < 0.75) continue;
    if (!best || sim > best.sim || (sim === best.sim && e.score > best.score)) best = { ...e, sim };
  }
  return best && { name: best.name, score: best.score };
}

// ---------- IO ----------

const DEFAULT_AUTHORS = [
  "Qwen", "google", "meta-llama", "mistralai", "openai", "deepseek-ai", "zai-org", "moonshotai",
  "MiniMaxAI", "nvidia", "microsoft", "ibm-granite", "LiquidAI", "allenai", "tencent", "XiaomiMiMo",
  "inclusionAI", "baidu", "stepfun-ai",
];
// Quantized re-uploads, speculative-decoding drafts, base checkpoints, and
// diffusion LMs (llama.cpp / MLX can't serve them as chat models).
const DERIVED = /(gguf|mlx|awq|gptq|fp8|fp4|int4|int8|exl\d|bnb|-4bit|-8bit|qat|onnx|eagle|dflash|dspark|(^|[-_/])base([-_]|$)|llada|diffusion)/i;

/** GB/s by chip; Max/Ultra bandwidth depends on GPU core count. */
function appleBandwidth(chip: string, gpuCores: number): number | undefined {
  const t: Record<string, number | [number, number, number]> = {
    "M1": 68, "M1 Pro": 200, "M1 Max": 400, "M1 Ultra": 800,
    "M2": 100, "M2 Pro": 200, "M2 Max": 400, "M2 Ultra": 800,
    "M3": 100, "M3 Pro": 150, "M3 Max": [30, 300, 400], "M3 Ultra": 819,
    "M4": 120, "M4 Pro": 273, "M4 Max": [32, 410, 546],
    "M5": 153,
  };
  const v = t[chip.replace(/^Apple\s+/, "").trim()];
  return Array.isArray(v) ? (gpuCores > v[0] ? v[2] : v[1]) : v;
}

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

interface Candidate { id: string; total: number; created: string }

async function listCandidates(authors: string[]): Promise<Candidate[]> {
  const urls = authors.flatMap((a) => ["text-generation", "image-text-to-text"].flatMap((p) =>
    ["sort=downloads", "sort=createdAt&direction=-1"].map((s) =>
      `https://huggingface.co/api/models?author=${a}&pipeline_tag=${p}&${s}&limit=40&expand[]=safetensors&expand[]=tags&expand[]=createdAt`)));
  const pages = await pool(urls, 8, (u) => getJson(u).catch(() => []));
  const seen = new Map<string, Candidate>();
  for (const m of pages.flat()) {
    const total = m.safetensors?.total;
    if (!total || seen.has(m.id) || DERIVED.test(m.id)) continue;
    const tags: string[] = m.tags ?? [];
    if (!tags.includes("conversational") || tags.some((t) => /^base_model:(quantized|adapter|merge):/.test(t))) continue;
    seen.set(m.id, { id: m.id, total, created: (m.createdAt ?? "").slice(0, 10) });
  }
  return [...seen.values()];
}

async function aaScores(): Promise<{ name: string; score: number }[]> {
  const key = process.env.ARTIFICIAL_ANALYSIS_API_KEY;
  if (!key) return [];
  const body = await getJson("https://artificialanalysis.ai/api/v2/data/llms/models", { "x-api-key": key });
  const out: { name: string; score: number }[] = [];
  for (const m of body.data ?? []) {
    const ev = m.evaluations ?? {};
    const score = ev.artificial_analysis_intelligence_index ?? Object.entries(ev).find(([k]) => k.includes("intelligence_index"))?.[1];
    if (typeof score === "number") for (const n of [m.slug, m.name].filter(Boolean)) out.push({ name: n, score });
  }
  return out;
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
  const plausible = cands.filter((c) => (c.total * QUANTS.at(-1)!.bpw) / 8 / 1e9 < hw.memBudgetGB);
  // "gated" = HTTP 401/403: accept the model's license on its HF page (HF_TOKEN alone is not enough).
  const rejects = { "too big": cands.length - plausible.length, "too slow": 0, gated: 0, "no config": 0 };

  const rows = (await pool(plausible, 8, async (c) => {
    const cfg = await getJson(`https://huggingface.co/${c.id}/resolve/main/config.json`).catch((e: Error) => e);
    if (cfg instanceof Error) { rejects[/^40[13] /.test(cfg.message) ? "gated" : "no config"]++; return null; }
    const active = activeParams(c.total, cfg, c.id);
    const plan = bestPlan(c.total, active, kvCacheBytes(cfg, o.ctx), hw, o);
    if ("reject" in plan) { rejects[plan.reject]++; return null; }
    const hit = overrides[c.id] !== undefined ? { name: "--scores", score: overrides[c.id] } : matchScore(c.id, aa);
    return { ...c, active, ...plan, aa: hit, rank: hit ? hit.score * plan.quant.quality : undefined };
  })).filter((r) => r !== null);

  // Scored rows by quant-adjusted score; unscored rows after, by total size (a weak proxy).
  rows.sort((a, b) => (b.rank ?? -1) - (a.rank ?? -1) || b.total * b.quant.quality - a.total * a.quant.quality);
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
