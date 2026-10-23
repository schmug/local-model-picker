import { afterEach, expect, test } from "bun:test";
import { buildSnapshot } from "./snapshot";

const realFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = realFetch; });

test("snapshot crawl trims configs, sorts gated from errors, and reads AA once", async () => {
  const calls: string[] = [];
  globalThis.fetch = (async (input: string | URL | Request) => {
    const url = String(input);
    calls.push(url);
    const ok = (b: unknown) => new Response(JSON.stringify(b));
    if (url.startsWith("https://huggingface.co/api/models?author=Qwen&pipeline_tag=text-generation&sort=downloads"))
      return ok(["Qwen/Chat-8B", "Qwen/Gated-8B", "Qwen/Broken-8B"].map((id) => ({ id, safetensors: { total: 8e9 }, tags: ["conversational"] })));
    if (url.startsWith("https://huggingface.co/api/models")) return ok([]);
    if (url.endsWith("Qwen/Chat-8B/resolve/main/config.json")) return ok({ num_hidden_layers: 32, hidden_size: 4096, vocab_size: 1 });
    if (url.endsWith("Qwen/Gated-8B/resolve/main/config.json")) return new Response("", { status: 401 });
    if (url.includes("artificialanalysis.ai")) return ok({ data: [{ slug: "chat-8b", evaluations: { artificial_analysis_intelligence_index: 5 } }] });
    return new Response("", { status: 404 });
  }) as typeof fetch;

  const s = await buildSnapshot({ ARTIFICIAL_ANALYSIS_API_KEY: "test" }, new Date("2026-01-01T00:00:00Z"));
  expect(s.fetchedAt).toBe("2026-01-01T00:00:00.000Z");
  expect(s.models).toEqual([{ id: "Qwen/Chat-8B", total: 8e9, created: "", cfg: { num_hidden_layers: 32, hidden_size: 4096 } }]);
  expect(s.gated).toEqual(["Qwen/Gated-8B"]);
  expect(s.errors).toEqual(["404 https://huggingface.co/Qwen/Broken-8B/resolve/main/config.json"]);
  expect(s.aa).toEqual([{ name: "chat-8b", score: 5 }]);
  expect(calls.filter((u) => u.includes("artificialanalysis.ai"))).toHaveLength(1);
});
