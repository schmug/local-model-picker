/**
 * picker.cortech.online: daily cron writes the snapshot to KV; every request
 * (API and static page) must carry a valid Cloudflare Access assertion.
 * Calibration uploads additionally require the calibration service token.
 */
import { parseCalibration, type Calibration } from "../src/core";
import { verifyAccessJwt } from "./access";
import { buildSnapshot } from "./snapshot";

interface Env {
  SNAPSHOT: KVNamespace;
  ASSETS: Fetcher;
  ACCESS_TEAM_DOMAIN: string;
  ACCESS_AUD: string;
  /** Client ID of the service token allowed to upload calibrations. */
  ACCESS_SERVICE_CLIENT_ID: string;
  ARTIFICIAL_ANALYSIS_API_KEY?: string;
  HF_TOKEN?: string;
}

const KEY = "snapshot";
const CAL_KEY = "calibration";
const CAL_MAX = 500;
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

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
    if (pathname === "/api/calibration" && req.method === "GET") return json(await env.SNAPSHOT.get<Calibration[]>(CAL_KEY, "json") ?? []);
    if (pathname === "/api/calibration" && req.method === "POST") {
      // Uploads only from the calibration service token (its assertion carries common_name = client ID), not a browser session.
      if (!env.ACCESS_SERVICE_CLIENT_ID || auth.claims.common_name !== env.ACCESS_SERVICE_CLIENT_ID) return new Response("Forbidden", { status: 403 });
      let cal: Calibration;
      try { cal = parseCalibration(await req.json()); } catch (e) { return new Response((e as Error).message, { status: 400 }); }
      const cals = [...(await env.SNAPSHOT.get<Calibration[]>(CAL_KEY, "json") ?? []), cal].slice(-CAL_MAX);
      await env.SNAPSHOT.put(CAL_KEY, JSON.stringify(cals));
      return json(cal, 201);
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
