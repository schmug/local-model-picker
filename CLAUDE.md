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
