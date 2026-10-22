import { expect, test } from "bun:test";
import { verifyAccessJwt } from "./access";

const team = "example.cloudflareaccess.com";
const cfg = { teamDomain: team, aud: "aud-tag" };
const now = 1_800_000_000;
const pair = await crypto.subtle.generateKey({ name: "RSASSA-PKCS1-v1_5", modulusLength: 2048, publicExponent: new Uint8Array([1, 0, 1]), hash: "SHA-256" }, true, ["sign", "verify"]);
const jwk = { ...(await crypto.subtle.exportKey("jwk", pair.publicKey)), kid: "k1" };
const keys = async () => [jwk];

const enc = (o: object) => Buffer.from(JSON.stringify(o)).toString("base64url");
async function sign(claims: object, header: object = { alg: "RS256", kid: "k1" }) {
  const body = `${enc(header)}.${enc(claims)}`;
  const sig = await crypto.subtle.sign("RSASSA-PKCS1-v1_5", pair.privateKey, new TextEncoder().encode(body));
  return `${body}.${Buffer.from(sig).toString("base64url")}`;
}
const good = { iss: `https://${team}`, aud: ["aud-tag"], exp: now + 60, nbf: now - 60, email: "owner@example.com" };

test("valid Access assertion passes and returns its claims", async () => {
  const r = await verifyAccessJwt(await sign(good), cfg, keys, now);
  expect(r.ok && r.claims.email).toBe("owner@example.com");
});

test("Access check fails closed", async () => {
  const reason = async (t: string | null, c = cfg) => { const r = await verifyAccessJwt(t, c, keys, now); return r.ok ? "ok" : r.reason; };
  expect(await reason(await sign(good), { teamDomain: "", aud: "" })).toBe("access not configured");
  expect(await reason(null)).toBe("no assertion");
  expect(await reason("a.b")).toBe("malformed");
  expect(await reason(await sign({ ...good, aud: ["other"] }))).toBe("audience");
  expect(await reason(await sign({ ...good, iss: "https://evil.cloudflareaccess.com" }))).toBe("issuer");
  expect(await reason(await sign({ ...good, exp: now - 1 }))).toBe("expired");
  expect(await reason(await sign(good, { alg: "none", kid: "k1" }))).toBe("alg");
  expect(await reason(await sign(good, { alg: "RS256", kid: "k2" }))).toBe("unknown kid");
  const [h, , s] = (await sign(good)).split(".");
  expect(await reason(`${h}.${enc({ ...good, email: "attacker@example.com" })}.${s}`)).toBe("signature");
});
