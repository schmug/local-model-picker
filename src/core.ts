/**
 * Pure estimation and ranking for local inference. No I/O, no dependencies:
 * the same module runs in Bun (CLI), workerd (cron) and the browser (page).
 */

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


export const DEFAULT_AUTHORS = [
  "Qwen", "google", "meta-llama", "mistralai", "openai", "deepseek-ai", "zai-org", "moonshotai",
  "MiniMaxAI", "nvidia", "microsoft", "ibm-granite", "LiquidAI", "allenai", "tencent", "XiaomiMiMo",
  "inclusionAI", "baidu", "stepfun-ai",
];
// Quantized re-uploads, speculative-decoding drafts, base checkpoints, and
// diffusion LMs (llama.cpp / MLX can't serve them as chat models).
export const DERIVED = /(gguf|mlx|awq|gptq|fp8|fp4|int4|int8|exl\d|bnb|-4bit|-8bit|qat|onnx|eagle|dflash|dspark|(^|[-_/])base([-_]|$)|llada|diffusion)/i;

/** GB/s by Apple chip; Max bandwidth depends on GPU core count: [cores above which, binned, full]. */
export const APPLE_BANDWIDTH: Record<string, number | [number, number, number]> = {
  "M1": 68, "M1 Pro": 200, "M1 Max": 400, "M1 Ultra": 800,
  "M2": 100, "M2 Pro": 200, "M2 Max": 400, "M2 Ultra": 800,
  "M3": 100, "M3 Pro": 150, "M3 Max": [30, 300, 400], "M3 Ultra": 819,
  "M4": 120, "M4 Pro": 273, "M4 Max": [32, 410, 546],
  "M5": 153,
};

export function appleBandwidth(chip: string, gpuCores: number): number | undefined {
  const v = APPLE_BANDWIDTH[chip.replace(/^Apple\s+/, "").trim()];
  return Array.isArray(v) ? (gpuCores > v[0] ? v[2] : v[1]) : v;
}

/** Hardware presets for the page: one per chip, two for Max chips (binned and full GPU). */
export function applePresets(): { label: string; bandwidthGBs: number }[] {
  return Object.entries(APPLE_BANDWIDTH).flatMap(([chip, v]) => Array.isArray(v)
    ? [{ label: `${chip} (binned GPU)`, bandwidthGBs: v[1] }, { label: `${chip} (full GPU)`, bandwidthGBs: v[2] }]
    : [{ label: chip, bandwidthGBs: v }]);
}

// ---------- HF / AA response parsing ----------

export interface Candidate { id: string; total: number; created: string }

/** HF `/api/models` listing URLs: per org, chat and VLM pipelines, top downloads and newest. */
export function candidateUrls(authors: string[]): string[] {
  return authors.flatMap((a) => ["text-generation", "image-text-to-text"].flatMap((p) =>
    ["sort=downloads", "sort=createdAt&direction=-1"].map((s) =>
      `https://huggingface.co/api/models?author=${a}&pipeline_tag=${p}&${s}&limit=40&expand[]=safetensors&expand[]=tags&expand[]=createdAt`)));
}

/** Map with at most `n` calls of `fn` in flight; keeps input order. */
export async function pool<T, R>(items: T[], n: number, fn: (x: T) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length);
  let i = 0;
  await Promise.all(Array.from({ length: n }, async () => { while (i < items.length) { const j = i++; out[j] = await fn(items[j]); } }));
  return out;
}

/** Chat models from HF `/api/models` pages: dedupe, drop derived and adapter/merge repos. */
export function filterCandidates(pages: any[][]): Candidate[] {
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

export interface AaEntry { name: string; score: number }

/** Intelligence Index per AA slug and display name from `/api/v2/data/llms/models`. */
export function parseAaModels(body: any): AaEntry[] {
  const out: AaEntry[] = [];
  for (const m of body?.data ?? []) {
    const ev = m.evaluations ?? {};
    const score = ev.artificial_analysis_intelligence_index ?? Object.entries(ev).find(([k]) => k.includes("intelligence_index"))?.[1];
    if (typeof score === "number") for (const n of [m.slug, m.name].filter(Boolean)) out.push({ name: n, score });
  }
  return out;
}

// Every config.json field activeParams and kvCacheBytes read.
const CFG_KEYS = [
  "num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim",
  "intermediate_size", "moe_intermediate_size", "num_experts", "n_routed_experts", "num_local_experts",
  "num_experts_per_tok", "moe_topk", "top_k_experts", "first_k_dense_replace", "decoder_sparse_step",
  "kv_lora_rank", "qk_rope_head_dim", "sliding_window", "use_sliding_window", "layer_types",
];
/** The slice of config.json the estimators need, so a snapshot stays small. */
export function configSubset(cfg: ModelConfig): ModelConfig {
  const pick = (c: ModelConfig) => Object.fromEntries(CFG_KEYS.filter((k) => c[k] !== undefined).map((k) => [k, c[k]]));
  const out = pick(cfg);
  for (const k of ["text_config", "llm_config"]) if (cfg[k]) out[k] = pick(cfg[k]);
  return out;
}

// ---------- ranking ----------

/** A candidate with its config loaded: one row of the cron snapshot. */
export interface SnapshotModel extends Candidate { cfg: ModelConfig }
export interface Snapshot { fetchedAt: string; models: SnapshotModel[]; aa: AaEntry[]; gated: string[]; errors: string[] }

export interface Ranked extends Candidate, Plan { active: number; aa?: AaEntry; rank?: number }

/** Cheapest quant must fit before a config is worth fetching. */
export function plausible(total: number, hw: Hardware): boolean {
  return (total * QUANTS.at(-1)!.bpw) / 8 / 1e9 < hw.memBudgetGB;
}

/**
 * Plan every model for this hardware and order the survivors: scored rows by
 * quant-adjusted AA score, unscored rows after by total size (a weak proxy).
 * `overrides` ({"org/repo": score}) beat AA matches.
 */
export function rankModels(models: SnapshotModel[], aa: AaEntry[], hw: Hardware, o: Options, overrides: Record<string, number> = {}) {
  const rejects = { "too big": 0, "too slow": 0 };
  const rows: Ranked[] = [];
  for (const m of models) {
    if (!plausible(m.total, hw)) { rejects["too big"]++; continue; }
    const active = activeParams(m.total, m.cfg, m.id);
    const plan = bestPlan(m.total, active, kvCacheBytes(m.cfg, o.ctx), hw, o);
    if ("reject" in plan) { rejects[plan.reject]++; continue; }
    const hit = overrides[m.id] !== undefined ? { name: "--scores", score: overrides[m.id] } : matchScore(m.id, aa);
    rows.push({ id: m.id, total: m.total, created: m.created, active, ...plan, aa: hit, rank: hit ? hit.score * plan.quant.quality : undefined });
  }
  rows.sort((a, b) => (b.rank ?? -1) - (a.rank ?? -1) || b.total * b.quant.quality - a.total * a.quant.quality);
  return { rows, rejects };
}

/**
 * Efficiency from a llama-bench run: measured tok/s ÷ the bandwidth-bound
 * ceiling (bandwidth ÷ bytes read per token). Inverse of bestPlan's estimate.
 */
export function calibratedEfficiency(measuredTps: number, bandwidthGBs: number, active: number, bpw: number, kvBytes = 0): number {
  const bytesPerTok = (active * bpw) / 8 + kvBytes;
  return measuredTps / ((bandwidthGBs * 1e9) / bytesPerTok);
}
