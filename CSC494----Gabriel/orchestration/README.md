# PluralTree — In-Place Orchestration Layer

## How to read this project

**To run it** (start here):
1. `README.md` (this file) — what the layer is, where the folder goes, the run commands, and the architecture at a glance.
2. `NARVAL.md` — the operational runbook for the cluster: pre-download the model, run the sanity check, do the cheap `salloc` smoke test, then submit the full batch job.

**To understand how it works** (read the code in dataflow order — each file only references things introduced before it):
3. `bootstrap.py` — how the two agent repos are loaded by absolute path so their `utils/` folders never collide.
4. `llm_backend.py` — the `mock` / `api` / `local` switch; the single abstraction that lets one codebase run offline and on-cluster.
5. `agent_a_adapter.py` — Agent A (CCKG generator): what it consumes (a query) and produces (path dicts) via `.generate()` / `.repair()`.
6. `agent_b_engine.py` — Agent B (CulFiT critique): how it scores paths and returns approval + feedback.
7. `orchestrator.py` — how the three topologies (static / parallel / sequential) wire Agent A and Agent B together.
8. `run_ablation.py` and `run_local.py` — the harnesses that drive the topologies over the query set and write the logs (`run_ablation` = mock/api, `run_local` = on-cluster GPU).

**Reference, not sequential reading:**
- `AGENT_CONTRACT.md` — the interface and data contract a new Agent A or Agent B must honor. Read this only when swapping in a different agent.

**Quick start:** From inside `CSC494/` run `python -m orchestration.check_setup` to verify the layout, and `python -m orchestration.run_ablation --mode mock` to see the pipeline work offline.

## Structure of things

```
CSC494/
├── CulFiT/                              # Agent B 
│   └── utils/{prompt_utils,llm_utils,json_fields}.py
├── Cultural_Commonsense_Knowledge_Graph/  # Agent A 
│   ├── src/{config,main}.py
│   └── src/utils/{llm_query,prompt_templates,response_parser}.py
└── orchestration/        
    ├── __init__.py
    ├── bootstrap.py          # resolves the 3x "utils" collision via importlib
    ├── llm_backend.py        # make_backend("mock"|"api"|"local")
    ├── agent_a_adapter.py    # AgentA.generate()/.repair()
    ├── agent_b_engine.py     # CulFiT critique + get_static_schema()
    ├── orchestrator.py       # static / parallel / sequential
    └── run_ablation.py       # shared queries + mock responders + logging
```

## Run

```bash
cd CSC494
python -m orchestration.run_ablation --mode mock      # offline, no deps beyond pandas
python -m orchestration.run_ablation --mode api  --model gpt-4o
python -m orchestration.run_ablation --mode local --model meta-llama/Llama-3.1-8B-Instruct
```

Outputs to `CSC494/runs/`.

## How the utils collision is solved (bootstrap.py) --> now irrelevant as redundant utils is removed

There are three directories named `utils`:
`CulFiT/utils/`, `.../src/utils/`, and a CulFiT copy at `.../Cultural_..._Graph/utils/`.
Rather than put any of them on `sys.path` (where `import utils.X` would resolve
ambiguously), `bootstrap.py` loads each needed file by absolute path under a
unique module name (`culfit_prompt_utils`, `agentA_prompt_templates`, ...) and
aliases the names upstream code expects (e.g. `utils.prompt_utils`). Verified:
each name resolves to exactly one intended file.

## Dependency notes

- **mock mode** needs only `pandas` (used by Tonga's response_parser).
- **api/local modes** additionally need `tenacity`, `openai`, and for Tonga's
  `config.py`, `python-dotenv`. These load lazily, so a missing cluster-only dep
  never blocks mock development.

## Switching to the cluster

- `--mode api`: set `OPENAI_API_KEY` (or point at a vLLM server on 0.0.0.0:8001).
- need to Add GEMINI_API_KEY=... to your .env
- **Local / GPU (Narval):** use the dedicated entrypoint, which loads the HF model
  ONCE and reuses it across all runs (unlike `run_ablation`, which builds a fresh
  backend per run and refuses to reload a heavy local model):

  ```bash
  # On a LOGIN node first (compute nodes are air-gapped): pre-download weights
  huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
      --local-dir $SCRATCH/models/llama31-8b

  # Then submit the batch job from CSC494/
  cd CSC494
  sbatch orchestration/run_ablation.slurm
  ```

  Edit `run_ablation.slurm` first: set `--account` to Alliance allocation,
  confirm the GPU type, and adjust `MODEL`/`MAX_LOOPS` (overridable as env vars).
  The script loads `rust` before vllm/tokenizers and prefers Alliance wheels,
  per the Narval environment issues noted in the Agent A handoff. If you hit the
  "invalid wheel" corruption again, the venv is built in `$SLURM_TMPDIR` (node-
  local, fresh each job), which sidesteps stale site-packages metadata.

  To run local mode interactively (e.g. in a `salloc` session) instead of batch:
  ```bash
  python -m orchestration.run_local --model meta-llama/Llama-3.1-8B-Instruct --max_loops 3
  ```


## Metrics

The harness logs raw signal (approval, per-iteration precision, per-dimension
verdicts, loops, repairs, per-agent LLM call counts) so headline metrics can be
chosen later.
