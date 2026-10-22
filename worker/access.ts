/**
 * Cloudflare Access JWT check. Access sits in front of the custom domain; the
 * Worker verifies the assertion too, so a request that reaches it some other
 * way (a route added later, Access app misconfigured) is still refused.
 * Fails closed when the team domain or audience is not configured.
 */

export interface AccessConfig { teamDomain: string; aud: string }
type Jwk = JsonWebKey & { kid?: string };
export type KeySource = (teamDomain: string) => Promise<Jwk[]>;

const b64url = (s: string) => Uint8Array.from(atob(s.replace(/-/g, "+").replace(/_/g, "/")), (c) => c.charCodeAt(0));
const json = (s: string) => JSON.parse(new TextDecoder().decode(b64url(s)));

let cached: { team: string; keys: Jwk[]; at: number } | undefined;
/** Access signing keys, cached for an hour (Access rotates them every ~6 weeks). */
export const fetchCerts: KeySource = async (teamDomain) => {
  if (cached?.team === teamDomain && Date.now() - cached.at < 3_600_000) return cached.keys;
  const r = await fetch(`https://${teamDomain}/cdn-cgi/access/certs`);
  if (!r.ok) throw new Error(`Access certs ${r.status}`);
  const keys: Jwk[] = (await r.json() as { keys: Jwk[] }).keys;
  cached = { team: teamDomain, keys, at: Date.now() };
  return keys;
};

/** Claims of a valid assertion, or a reason it was refused. */
export async function verifyAccessJwt(token: string | null, cfg: AccessConfig, keys: KeySource = fetchCerts, now = Date.now() / 1000): Promise<{ ok: true; claims: Record<string, any> } | { ok: false; reason: string }> {
  if (!cfg.teamDomain || !cfg.aud) return { ok: false, reason: "access not configured" };
  if (!token) return { ok: false, reason: "no assertion" };
  const parts = token.split(".");
  if (parts.length !== 3) return { ok: false, reason: "malformed" };
  let header: any, claims: any;
  try { header = json(parts[0]); claims = json(parts[1]); } catch { return { ok: false, reason: "malformed" }; }
  if (header.alg !== "RS256") return { ok: false, reason: "alg" };
  const jwk = (await keys(cfg.teamDomain)).find((k) => k.kid === header.kid);
  if (!jwk) return { ok: false, reason: "unknown kid" };
  const key = await crypto.subtle.importKey("jwk", jwk, { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" }, false, ["verify"]);
  const signed = new TextEncoder().encode(`${parts[0]}.${parts[1]}`);
  if (!(await crypto.subtle.verify("RSASSA-PKCS1-v1_5", key, b64url(parts[2]), signed))) return { ok: false, reason: "signature" };
  if (claims.iss !== `https://${cfg.teamDomain}`) return { ok: false, reason: "issuer" };
  if (!(Array.isArray(claims.aud) ? claims.aud : [claims.aud]).includes(cfg.aud)) return { ok: false, reason: "audience" };
  if (typeof claims.exp !== "number" || claims.exp <= now) return { ok: false, reason: "expired" };
  if (typeof claims.nbf === "number" && claims.nbf > now + 60) return { ok: false, reason: "not yet valid" };
  return { ok: true, claims };
}
