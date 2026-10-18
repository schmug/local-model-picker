# Spec: local model picker → own repo + live dashboard

Status: APPROVED 2026-10-18. Repo schmug/local-model-picker (public); dashboard is a Worker behind Cloudflare Access, owner only, with AA scores; ruleset with required checks approved.
Source: `harness/pick-local-model.ts` + `.test.ts` on `claude/best-local-model-selection-2149b6` in `~/ai-newtab` (commits 7699722, e9137ec). Known model gaps: schmug/agent-notes#205.

## 1. Problem

The picker answers "which open-weight model is smartest that still runs fast enough on this machine", but it lives in a fork of someone else's repo and only runs as a local CLI with two API keys. There is no way to look at the answer from another machine, try a different hardware budget without re-running the full HF crawl, or see the score-vs-speed tradeoff at a glance.

## 2. Prior art (librarian, `~/librarian`)

| Tool | Covers | Gap vs this picker |
|---|---|---|
| `wiki/sources/alexsjones-llmfit.md` (AlexsJones/llmfit, Rust) | Hardware detection, "hundreds of models", TUI + web dashboard, REST `/api/v1/system` `/api/v1/models` | Vault note shows fit-only claims; no live intelligence ranking and no bandwidth-derived tok/s estimate noted. (Deep pass is thin: key claims are README bullets. Unverified beyond that.) |
| `wiki/sources/huggingface-hf-agents.md` (huggingface/hf-agents) | HF CLI extension: calls llmfit to pick a model, starts llama-server, launches a coding agent | Inherits llmfit's ranking; it is a launcher, not a ranker |
| canirun.ai, LM Studio fit indicator, HF hardware-compatibility badge | **Not in the vault** (body + catalog grep both empty). From general knowledge, all three answer "does it fit in memory" per model; none ranks by a third-party intelligence score with a speed floor. Not verified this session. |
| Schmug-owned: `wiki/sources/schmug-benchburner.md` | Public LLM benchmark with a daily aggregator building a static leaderboard | Different domain (orchestration benchmark); reusable pattern: cron aggregate → static page. No overlap in function. |

Gap this repo fills: rank by **live Artificial Analysis Intelligence Index**, filter by **memory fit (weights per llama.cpp quant + KV cache from config.json)** and a **decode tok/s floor estimated from memory bandwidth × calibrated efficiency**, picking the highest quant that clears both.

## 3. Artificial Analysis terms — decides visibility

Sources read 2026-10-18:
- API docs https://artificialanalysis.ai/documentation — free tier 1,000 req/day; "Attribution is required for all use of our free API."; keys must not go in client-side code.
- Data Platform Terms v1.1 (rev. 2026-08-19) https://artificialanalysiscdn.com/legal/ProDataPlatformTerms.pdf — **Scope explicitly includes the free API tier.** "Data" (§1.4) names the Artificial Analysis Intelligence Index and individual scores.

Relevant clauses:
- §2.3 All tiers may: Internal Use; share charts publicly with attribution; "brief citations of individual Data points" that do not reproduce Data "in a structured, tabular, or machine-readable format".
- §2.4(c) restriction: "Embed or otherwise make raw Data available through any customer-facing product, API, dashboard". (quote under 15 words of the clause)
- §2.5 anti-competitive: no product for third parties whose primary purpose is "model/provider selection guidance" without AA's written consent.
- §1.10 a ranking that keeps scores individually identifiable is not Derived Data; external Derived Data is Pro/Commercial only anyway.
- §5.1 attribution mandatory wherever Data leaves internal systems; §5.2 non-endorsement statement.

**Verdict (not legal advice):** a public dashboard showing or ranking by AA intelligence scores is **disallowed** on the free tier — it is a customer-facing dashboard of raw Data (§2.4(c)) and its primary purpose is model-selection guidance (§2.5). Allowed: **private, single-user** use (Internal Use; §2.2 makes free/Pro access one named user). A per-visitor-key mode would make each visitor their own licensee, but AA says keys stay out of client code, so it would need a pass-through proxy holding visitors' keys — more risk than it is worth; treated as out of scope unless Schmug wants it.

Consequence for the design: the deployed dashboard with AA scores sits behind Cloudflare Access (Schmug only). A public variant is possible only with AA scores removed (size proxy or a different, redistributable score source) or AA's written consent.

## 4. Scope

**In**
1. New Bun repo seeded from the two files at e9137ec (history via `git log --follow` is not carried; commit message cites the source SHAs).
2. Split the single file into:
   - `src/core.ts` — pure math, no I/O: `QUANTS`, `textConfig`, `activeParams`, `kvCacheBytes`, `bestPlan`, `nameTokens`, `matchScore`, `appleBandwidth` table. Runs in Bun, Workers and the browser.
   - `src/cli.ts` — current CLI behaviour unchanged (sysctl detection, flags, `--json`), importing core.
   - `worker/` — Cloudflare Worker: cron + JSON API + static assets.
   - `web/` — dashboard page; bundled with `bun build` into Workers static assets.
3. Worker cron (daily): fetch HF candidate lists (19 orgs) + each `config.json` + AA `/api/v2/data/llms/models` (1 request/day of the 1,000 limit); store one snapshot `{fetchedAt, models:[{id, total, active?, cfgSubset, aaScore?, aaName?}], gated[], errors[]}` in **KV** (single key, ~580 rows; D1 is unnecessary until history is wanted). Secrets `ARTIFICIAL_ANALYSIS_API_KEY`, `HF_TOKEN` via `wrangler secret`, never in the bundle.
4. Worker `GET /api/snapshot` returns the snapshot; only reachable through Access.
5. Page: hardware presets (from the Apple chip table + manual), sliders for mem budget GB, bandwidth GB/s, ctx, tok/s floor, efficiency; runs `core.ts` client-side on the snapshot; table + score-vs-tok/s scatter; AA attribution line + logo on the chart (§5.1).
6. CLI `--calibrate` path: run `llama-bench` on a chosen GGUF, compute efficiency = measured tok/s ÷ (bandwidth ÷ active bytes), print it, and optionally `POST /api/calibration` (Access service token) so the page can default to measured efficiency per chip.
7. CI: GitHub Actions running `bun test` + `tsc --noEmit` + worker build; repo ruleset requiring those checks on `main` before any merge.

**Out**
- Public, unauthenticated display of AA scores (see §3).
- Per-visitor AA keys.
- Fixing #205 model gaps (native 4-bit overestimate, efficiency calibration breadth) — tracked there; separate issues.
- Non-Apple hardware detection in the CLI (manual `--mem-gb/--bandwidth` already covers it).
- Score history / trends (would move storage to D1).
- Removing the picker from the ai-newtab fork branch (leave the branch as is).

## 5. Constraints

- `.env` / `.dev.vars` / wrangler secrets never staged; repo `.gitignore` covers them from the first commit. Do not commit AA data (fixtures, snapshots) to the repo — committing scores to a public repo is redistribution (§2.4(a)). Test fixtures for `matchScore` use synthetic names/scores.
- Keys server-side only (AA docs + §3).
- Access policy for humans: Allow Schmug's identity only. Calibration upload uses a service token with `Action=Service Auth`, not `Allow` (global CLAUDE.md).
- Workers plan: free plan allows 50 external subrequests per invocation; the cron needs ~19 list calls + ~580 config fetches + 1 AA call. **Paid plan (10,000 default) does it in one run; free plan needs batching via Queues or chunked cron runs.** Which plan Schmug's account is on: unverified.
- Cron wall time limit 15 min; HF unauthenticated rate limits apply — keep the existing `pool()` concurrency.
- Core stays dependency-free so the same module runs in Bun, workerd and the browser.
- Existing 4 tests in `pick-local-model.test.ts` must pass unchanged against `core.ts` (import path change only).
- AA attribution: "Source: Artificial Analysis (artificialanalysis.ai)" with link on page; AA logo visible on the chart; non-endorsement line.
- Issue → PR loop: test first, PR with test output, merge only after the ruleset with required checks exists (global CLAUDE.md). Ruleset creation is a guardrail change → needs Schmug's go-ahead.
- Second-opinion gate fires at implementation time for the Access policy (authorization boundary).

## 6. Decisions that need Schmug

1. **Repo name.** Free under `schmug/`: `local-model-picker`, `localfit`, `fitpick` (checked with `gh repo view`, all not found). Recommendation: `local-model-picker` (plain; avoids confusion with llmfit).
2. **Repo visibility.** Code contains no AA data, so public is allowed by AA terms. Recommendation: public (free required-checks rulesets; README is human-facing product output). Private works too, but rulesets on private repos need GitHub Pro or higher (Schmug's plan: unverified).
3. **Hosting / dashboard visibility.** Options:
   - (a) **Recommended:** Worker on `workers.dev` or a Schmug subdomain behind Cloudflare Access, Schmug only, with AA scores.
   - (b) Public page without AA scores (size proxy only) + private Access view with them. Size proxy is a weak ranking; adds a second mode to maintain.
   - (c) Ask AA for written consent for a public page (§2.5), then public.
   - Also: which Cloudflare account/zone, and whether it is on Workers Paid (decides single cron run vs Queues batching).
4. **Approval to create the ruleset** (required checks: test, typecheck, build) on the new repo — guardrail change.

## 7. Acceptance criteria

1. `bun test` in the new repo: the 4 seeded tests pass plus one test per new pure function (snapshot → ranked rows; calibration efficiency formula); 0 failing; `tsc --noEmit` clean.
2. `bun src/cli.ts --json` on the M4 Max produces the same top-15 model ids and quants as `bun harness/pick-local-model.ts --json` at e9137ec with the same flags and same day's data.
3. After one cron run, `GET /api/snapshot` through Access returns ≥ 500 models with `fetchedAt` within 24 h; the same URL without Access credentials returns 302/403 (checked with curl).
4. In a browser, changing the mem-budget slider from 64 to 128 GB changes the ranked table and the scatter without a network request (DevTools network panel shows only the initial snapshot fetch); the AA attribution text and logo are visible.
5. `git grep -nE 'ARTIFICIAL_ANALYSIS_API_KEY=|hf_[A-Za-z0-9]{20,}'` and a search of the built `web/` bundle for the key name return nothing; `main` has a ruleset requiring the CI checks (shown by `gh api repos/schmug/<name>/rulesets`).

## 8. Open questions

- Does AA's v2 models endpoint allow server-to-server use from Cloudflare IPs without extra headers? (Settled by the first cron run.)
- Should the calibration table be per chip (M4 Max) or per chip × model architecture (dense vs small-active MoE, per #205's 0.31 vs 0.45 spread)? Recommendation: store raw runs, let the page pick nearest match; decide when #205 is worked.
- llmfit's actual ranking method — the vault note is thin. Worth a 10-minute read of its README before the README for this repo claims a gap. (Did not do web research per librarian rules.)
- canirun.ai / LM Studio / HF hardware badge are not in the vault; if Schmug wants them compared, drop the links to the Librarian bot.
