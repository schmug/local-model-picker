import { expect, test } from "bun:test";
import {
  activeParams, bestPlan, calibratedEfficiency, configSubset, filterCandidates, kvCacheBytes, matchScore,
  parseAaModels, rankModels, type SnapshotModel,
} from "./core";

// Trimmed from the real config.json files on Hugging Face.
const gptOss120b = { num_hidden_layers: 36, hidden_size: 2880, num_attention_heads: 64, num_key_value_heads: 8, head_dim: 64, intermediate_size: 2880, num_local_experts: 128, num_experts_per_tok: 4, sliding_window: 128, layer_types: [...Array(18).fill("full_attention"), ...Array(18).fill("sliding_attention")] };
const lfm2Moe = { num_hidden_layers: 40, hidden_size: 2048, num_attention_heads: 32, num_key_value_heads: 8, layer_types: [...Array(30).fill("conv"), ...Array(10).fill("full_attention")] };
const m4Max = { memBudgetGB: 103, bandwidthGBs: 546, label: "M4 Max" };

test("active params: name suffix wins, else derived from MoE config, dense stays total", () => {
  expect(activeParams(24e9, lfm2Moe, "LiquidAI/LFM2.5-24B-A2B")).toBe(2e9);
  // gpt-oss-120b publishes 5.1B active; the config-only derivation must land near it.
  const a = activeParams(116.8e9, gptOss120b, "openai/gpt-oss-120b");
  expect(a).toBeGreaterThan(4.5e9);
  expect(a).toBeLessThan(6e9);
  expect(activeParams(8e9, { num_hidden_layers: 32, hidden_size: 4096 }, "meta-llama/Llama-3.1-8B-Instruct")).toBe(8e9);
});

test("KV cache counts attention layers only and caps sliding layers at the window", () => {
  const perTok = 2 * 8 * 64 * 2;
  expect(kvCacheBytes(gptOss120b, 32768)).toBe(18 * perTok * 32768 + 18 * perTok * 128);
  expect(kvCacheBytes(lfm2Moe, 1000)).toBe(10 * 2 * 8 * 64 * 2 * 1000);
  expect(kvCacheBytes({ num_hidden_layers: 61, kv_lora_rank: 512, qk_rope_head_dim: 64 }, 10)).toBe(61 * 576 * 2 * 10);
});

test("plan picks the best quant that fits and is fast enough, else says why not", () => {
  const o = { ctx: 32768, minTps: 20, efficiency: 0.4 };
  // 120B MoE: Q8 is ~124 GB (too big); Q6 ~96 GB fits and is fast at ~5B active.
  const p = bestPlan(116.8e9, 5.1e9, 1e9, m4Max, o);
  expect("quant" in p && p.quant.name).toBe("Q6_K");
  // Dense 70B fits at every quant ≤ Q8 but stays well under 20 tok/s.
  expect(bestPlan(70e9, 70e9, 1e9, m4Max, o)).toEqual({ reject: "too slow" });
  expect(bestPlan(700e9, 37e9, 1e9, m4Max, o)).toEqual({ reject: "too big" });
});

test("AA matching requires equal size tokens and takes the best variant", () => {
  // Synthetic scores: real AA data must not be committed (Data Platform Terms §2.4).
  const aa = [
    { name: "Qwen3 30B A3B 2507 (Reasoning)", score: 12 },
    { name: "Qwen3 30B A3B 2507 Instruct", score: 7 },
    { name: "Qwen3 235B A22B 2507 (Reasoning)", score: 20 },
    { name: "gpt-oss-120B (high)", score: 25 },
  ];
  expect(matchScore("Qwen/Qwen3-30B-A3B-Instruct-2507", aa)?.score).toBe(12);
  expect(matchScore("openai/gpt-oss-20b", aa)).toBeUndefined();
  expect(matchScore("Qwen/Qwen3-32B", aa)).toBeUndefined();
});

test("config subset keeps every field the estimators read, including nested text_config", () => {
  const full = { architectures: ["X"], vocab_size: 1, text_config: { ...gptOss120b, rope_theta: 1e6 } };
  const sub = configSubset(full);
  expect(sub.architectures).toBeUndefined();
  expect(sub.text_config.rope_theta).toBeUndefined();
  expect(activeParams(116.8e9, sub, "openai/gpt-oss-120b")).toBe(activeParams(116.8e9, full, "openai/gpt-oss-120b"));
  expect(kvCacheBytes(sub, 32768)).toBe(kvCacheBytes(full, 32768));
});

test("HF listing keeps original conversational repos once; AA parsing indexes slug and name", () => {
  const m = (id: string, tags = ["conversational"]) => ({ id, safetensors: { total: 8e9 }, tags, createdAt: "2026-01-02T00:00:00Z" });
  const c = filterCandidates([[m("org/Chat-8B"), m("org/Chat-8B-GGUF"), m("org/Chat-8B-Base"), m("org/Plain-8B", [])],
    [m("org/Chat-8B"), m("org/Tune-8B", ["conversational", "base_model:adapter:org/Chat-8B"])]]);
  expect(c).toEqual([{ id: "org/Chat-8B", total: 8e9, created: "2026-01-02" }]);
  const aa = parseAaModels({ data: [{ slug: "chat-8b", name: "Chat 8B", evaluations: { artificial_analysis_intelligence_index: 9 } }, { slug: "none", evaluations: {} }] });
  expect(aa).toEqual([{ name: "chat-8b", score: 9 }, { name: "Chat 8B", score: 9 }]);
});

test("snapshot ranking: scored rows by quant-adjusted score, unscored after by size, rejects counted", () => {
  const dense = (layers: number, hidden: number) => ({ num_hidden_layers: layers, hidden_size: hidden, num_attention_heads: 32, num_key_value_heads: 8 });
  const models: SnapshotModel[] = [
    { id: "org/Small-8B", total: 8e9, created: "", cfg: dense(32, 4096) },
    { id: "org/Mid-14B", total: 14e9, created: "", cfg: dense(40, 5120) },
    { id: "org/Unscored-12B", total: 12e9, created: "", cfg: dense(40, 5120) },
    { id: "org/Dense-70B", total: 70e9, created: "", cfg: dense(80, 8192) },
    { id: "org/Huge-700B", total: 700e9, created: "", cfg: dense(61, 7168) },
  ];
  const aa = [{ name: "Small 8B", score: 10 }, { name: "Mid 14B", score: 30 }, { name: "Dense 70B", score: 50 }];
  const { rows, rejects } = rankModels(models, aa, m4Max, { ctx: 8192, minTps: 20, efficiency: 0.4 }, { "org/Small-8B": 40 });
  expect(rows.map((r) => r.id)).toEqual(["org/Small-8B", "org/Mid-14B", "org/Unscored-12B"]);
  expect(rows[0].aa?.name).toBe("--scores");
  expect(rows[1].rank).toBe(30 * rows[1].quant.quality);
  expect(rejects).toEqual({ "too big": 1, "too slow": 1 });
});

test("calibrated efficiency inverts the bandwidth estimate", () => {
  // 3B active at Q4_K_M on 546 GB/s: ceiling ≈ 300 tok/s, so 92 measured ≈ 0.31.
  expect(calibratedEfficiency(92, 546, 3e9, 4.85)).toBeCloseTo(0.306, 3);
  const o = { ctx: 32768, minTps: 1, efficiency: 0.37 };
  const p = bestPlan(30e9, 3e9, 2e9, m4Max, o);
  if (!("quant" in p)) throw new Error("expected a plan");
  expect(calibratedEfficiency(p.tps, m4Max.bandwidthGBs, 3e9, p.quant.bpw, 2e9)).toBeCloseTo(0.37, 10);
});
