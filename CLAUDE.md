# local-model-picker

Ranks open-weight models for local inference: memory fit (weights per llama.cpp
quant + KV cache from config.json), decode tok/s from memory bandwidth x
efficiency, ordered by the Artificial Analysis Intelligence Index.

Plan of record: `.claude/plans/local-model-picker.md` (repo split into
core/cli/worker/web, Worker + cron behind Cloudflare Access).

## Commands

- `bun test` — unit tests for the pure estimation functions
- `bun run typecheck` — `tsc --noEmit`
- `bun src/pick-local-model.ts` — the CLI; flags are documented in its header comment

## Invariants

- CI job names `test` and `typecheck` are required status checks in the
  `main` ruleset. Renaming a job requires updating the ruleset in the same PR.
- A PR that touches `.github/workflows/` stops for Schmug: open it, do not merge
  it or enable auto-merge, and ask in chat. Reason: the ruleset's required checks
  run the workflow from the PR head, so a PR can edit `ci.yml` to make its own
  checks pass. Repo-level rulesets rejected a `workflows` rule pinning `ci.yml`
  to main (HTTP 422, 2026-10-18), so this rule is the only gate on that path.
- Never commit Artificial Analysis data (scores, snapshots, fixtures with real
  scores). AA Data Platform Terms §2.4 forbid redistributing raw Data and
  §2.5 forbid a public model-selection product. Test fixtures use synthetic scores.
- AA scores are shown only behind Cloudflare Access (owner only). Any page that
  displays them carries "Source: Artificial Analysis (artificialanalysis.ai)"
  with a link, the AA logo on charts, and a non-endorsement line (§5).
- API keys (`ARTIFICIAL_ANALYSIS_API_KEY`, `HF_TOKEN`) live in `.env` locally and
  in `wrangler secret` for the Worker. Never in client bundles or the repo.
- Calibration uploads authenticate with an Access service token whose policy
  action is Service Auth, not Allow.
- Known estimation gaps: schmug/agent-notes#205.
