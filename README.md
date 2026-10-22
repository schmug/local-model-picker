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

Intelligence scores: Source: [Artificial Analysis](https://artificialanalysis.ai/).
This project is not endorsed by Artificial Analysis.
