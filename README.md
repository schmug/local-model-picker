# local-model-picker

Which open-weight model is the smartest one that still runs fast enough on your machine?

```bash
bun install
bun src/cli.ts --min-tps 20 --ctx 32768 --top 15
```

The CLI detects your Apple Silicon memory budget and bandwidth, pulls chat models
from 19 lab orgs on Hugging Face, picks the highest-precision llama.cpp quant that
fits in memory and clears the tok/s floor, and ranks the survivors by the
Artificial Analysis Intelligence Index.

- `ARTIFICIAL_ANALYSIS_API_KEY` (optional): get a free key at
  [artificialanalysis.ai](https://artificialanalysis.ai/). Without it, rows are
  ordered by size, not intelligence.
- `HF_TOKEN` (optional): lets gated model configs load.

Speeds are estimates (bandwidth x efficiency 0.4, calibrated with llama-bench on
an M4 Max). Benchmark your top picks before trusting them.

## Dashboard (owner only)

`worker/` runs on Cloudflare at picker.cortech.online behind Cloudflare Access.
A daily cron crawls Hugging Face and the AA model list into KV; `GET /api/snapshot`
serves it, and the Worker re-checks the Access assertion on every request.

```bash
bunx wrangler secret put ARTIFICIAL_ANALYSIS_API_KEY
bunx wrangler secret put HF_TOKEN
# fill ACCESS_TEAM_DOMAIN and ACCESS_AUD in wrangler.jsonc from the Access app, then
bun run deploy
```

Calibration: measure real decode speed with llama-bench and send it to the
dashboard, which then defaults the efficiency slider to the median for that chip.

```bash
bun src/cli.ts --calibrate ~/models/Qwen3-30B-A3B-Q4_K_M.gguf --active-b 3 \
  --upload https://picker.cortech.online   # needs ACCESS_CLIENT_ID / ACCESS_CLIENT_SECRET
```

The upload uses an Access service token. Its Access policy action must be
Service Auth (not Allow), and `ACCESS_SERVICE_CLIENT_ID` in `wrangler.jsonc` must
be its Client ID; any other identity gets 403 on `POST /api/calibration`.

Intelligence scores: Source: [Artificial Analysis](https://artificialanalysis.ai/).
This project is not endorsed by Artificial Analysis.
