import { DEFAULT_AUTHORS, candidateUrls, configSubset, filterCandidates, parseAaModels, pool, type Snapshot, type SnapshotModel } from "../src/core";

export interface CrawlEnv { ARTIFICIAL_ANALYSIS_API_KEY?: string; HF_TOKEN?: string }

/**
 * One daily crawl: HF candidates from the 19 orgs, each config.json trimmed
 * to the estimator fields, and the AA model list (1 of the 1,000 daily
 * requests). ~80 list calls + ~600 configs + 1 AA fits Workers Paid's
 * subrequest limit in a single run.
 */
export async function buildSnapshot(env: CrawlEnv, now = new Date()): Promise<Snapshot> {
  const hf: Record<string, string> = env.HF_TOKEN ? { authorization: `Bearer ${env.HF_TOKEN}` } : {};
  const getJson = async (url: string, headers = hf) => {
    const r = await fetch(url, { headers });
    if (!r.ok) throw new Error(`${r.status} ${url}`);
    return r.json() as Promise<any>;
  };
  const errors: string[] = [];
  const pages = await pool(candidateUrls(DEFAULT_AUTHORS), 8, (u) => getJson(u).catch((e: Error) => { errors.push(e.message); return []; }));
  const aa = env.ARTIFICIAL_ANALYSIS_API_KEY
    ? await getJson("https://artificialanalysis.ai/api/v2/data/llms/models", { "x-api-key": env.ARTIFICIAL_ANALYSIS_API_KEY })
      .then(parseAaModels).catch((e: Error) => { errors.push(`AA: ${e.message}`); return []; })
    : [];
  const gated: string[] = [];
  const models = (await pool(filterCandidates(pages), 8, async (c): Promise<SnapshotModel | null> => {
    try { return { ...c, cfg: configSubset(await getJson(`https://huggingface.co/${c.id}/resolve/main/config.json`)) }; }
    catch (e) { /^40[13] /.test((e as Error).message) ? gated.push(c.id) : errors.push((e as Error).message); return null; }
  })).filter((m) => m !== null);
  return { fetchedAt: now.toISOString(), models, aa, gated, errors };
}
