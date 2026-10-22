/**
 * picker.cortech.online: daily cron writes the snapshot to KV; every request
 * (API and static page) must carry a valid Cloudflare Access assertion.
 */
import { verifyAccessJwt } from "./access";
import { buildSnapshot } from "./snapshot";

interface Env {
  SNAPSHOT: KVNamespace;
  ASSETS: Fetcher;
  ACCESS_TEAM_DOMAIN: string;
  ACCESS_AUD: string;
  ARTIFICIAL_ANALYSIS_API_KEY?: string;
  HF_TOKEN?: string;
}

const KEY = "snapshot";

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const auth = await verifyAccessJwt(req.headers.get("cf-access-jwt-assertion"), { teamDomain: env.ACCESS_TEAM_DOMAIN, aud: env.ACCESS_AUD });
    if (!auth.ok) return new Response("Forbidden", { status: 403 });
    const { pathname } = new URL(req.url);
    if (pathname === "/api/snapshot" && req.method === "GET") {
      const body = await env.SNAPSHOT.get(KEY);
      if (!body) return new Response("No snapshot yet", { status: 503 });
      return new Response(body, { headers: { "content-type": "application/json", "cache-control": "private, max-age=300" } });
    }
    if (pathname.startsWith("/api/")) return new Response("Not found", { status: 404 });
    return env.ASSETS.fetch(req);
  },

  async scheduled(_: ScheduledController, env: Env): Promise<void> {
    const snap = await buildSnapshot(env);
    // Keep yesterday's snapshot if HF was unreachable rather than serving an empty one.
    if (snap.models.length === 0) throw new Error(`empty crawl: ${snap.errors.slice(0, 3).join("; ")}`);
    await env.SNAPSHOT.put(KEY, JSON.stringify(snap));
    console.log(`snapshot: ${snap.models.length} models, ${snap.aa.length} AA entries, ${snap.gated.length} gated, ${snap.errors.length} errors`);
  },
} satisfies ExportedHandler<Env>;
