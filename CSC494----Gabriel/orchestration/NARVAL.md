# Running on Narval (def-enaskt)

Project lives at `~/projects/def-enaskt/hhpfiona/CSC494`, pulled from GitHub.
Allocation is **Opportunistic Use** (low priority, no guaranteed resources),
so keep jobs modest and expect variable queue times.

> Compute nodes have **no internet**. Anything downloaded from the web (model
> weights, pip packages from PyPI, HF datasets) must be fetched on a **login
> node** first, then read from disk inside the job.

> **Windows editors add CRLF line endings that break bash.** After editing any
> `.slurm` or `.sh` on Windows, run `dos2unix <file>` on Narval (or set
> `git config --global core.autocrlf input` once, locally, so git strips CRLF
> on commit). Verify a script with `bash -n <file>` before submitting.

---

## Step 0 (once) — Pre-download model + SBERT on a LOGIN node

Use a **venv in $SCRATCH**, not `pip install --user`. The `--user` site
(`~/.local`) previously got corrupted (broke `import yaml`, caused `Errno 5` I/O
errors); a scratch venv sidesteps it and doesn't touch home quota.

```bash
cd ~/projects/def-enaskt/hhpfiona/CSC494
module load python/3.11

# One-time login-node venv for downloads/pre-caching (lives in scratch).
python -m venv $SCRATCH/precache_env
source $SCRATCH/precache_env/bin/activate
pip install --no-index --upgrade pip
# IMPORTANT: install `requests` in the SAME command as sentence-transformers.
# The Alliance sentence-transformers wheel does NOT pull requests automatically,
# and the pre-cache line below imports it — omit requests and it crashes with
# "ModuleNotFoundError: No module named 'requests'" BEFORE caching anything.
pip install --no-index huggingface_hub sentence-transformers requests

export HF_HOME=$SCRATCH/hf_cache       # big files belong in scratch, not project

# --- Pre-cache SBERT (all-MiniLM-L6-v2) so the air-gapped job can load it ---
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('SBERT cached OK')"

# Verify it landed on disk (must return a path; empty => NOT cached, do not submit):
find $SCRATCH/hf_cache -iname "*minilm*" -type d | head

# Verify it loads in OFFLINE mode — this is exactly what the batch job does.
# 'offline load OK' with no traceback == the job's SBERT check will pass.
HF_HUB_OFFLINE=1 python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('offline load OK')"

# --- Download the Llama weights (gated model, needs auth) ---
hf auth login                 # needed for gated models like Llama
hf download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir $SCRATCH/models/llama31-8b
    # only need to download once

deactivate   # done with the login-node precache venv
```

Then point `--model` at `$SCRATCH/models/llama31-8b` (a local dir is the most
reliable offline). If using a different model, e.g. qwen25-7b, point `--model`
at `$SCRATCH/models/qwen25-7b`.

If SBERT is NOT pre-cached, the template / llm-rewrite jobs will silently fall
back to Jaccard token-overlap for graph node merging (weaker). The batch script
now fails fast at minute 1 if SBERT can't load offline, so a missing cache costs
seconds, not a 12h slot.

---

## Step 1 — Verify the drop (no GPU needed, run on login node)

```bash
cd ~/projects/def-enaskt/hhpfiona/CSC494
module load python/3.11
python -m orchestration.check_setup
```

Expect `PASS — setup looks good.` If it can't find a repo folder, the
`orchestration/` folder isn't a direct child of `CSC494/`, or a repo folder was
renamed.

---

## Step 2 — Cheap smoke test via `salloc` (interactive, 1 query / 1 topology / 1 loop)

Grab a short interactive GPU session and run the smallest possible real job.
This shakes out env/model/parsing problems for pennies before a full batch job.

```bash
salloc --account=def-enaskt --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:40:00

# if the wait is long:
squeue -u $USER --start      # estimated start time, if predictable
squeue -j <jobid> -o "%.12i %.8T %.10M %.10l %.20S %R"   # state + reason

# once on the GPU node (prompt changes e.g. hhpfiona@narval3 -> hhpfiona@ng10104)
cd ~/projects/def-enaskt/hhpfiona/CSC494

module purge
module load StdEnv/2023 gcc/12.3 rust/1.76.0 python/3.11 arrow/16 cuda/12.2

# node-local venv (fast, avoids stale metadata). Note: requests IS in the list.
python -m venv $SLURM_TMPDIR/env && source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index requests torch transformers tokenizers sentence-transformers \
    accelerate openai pandas openpyxl tenacity tqdm pydantic python-dotenv

export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Stage model to node-local SSD (loading 16GB off Lustre is slow).
time cp -r $SCRATCH/models/llama31-8b $SLURM_TMPDIR/llama31-8b

# Smoke: 1 query, sequential only, max_loops=1 — loads from LOCAL disk.
python -m orchestration.run_local \
    --model $SLURM_TMPDIR/llama31-8b \
    --queries CulFiT/GlobalCultureQA/eval_set_n100.jsonl \
    --smoke

exit   # releases the salloc allocation
```

`--smoke` = 1 query, sequential topology, max_loops=1. It exercises the full
path (model load, Agent A generate, Agent B critique, one repair), so JSON
parsing of real Llama output breaks here cheaply. Check `runs/ablation_smoke_*.jsonl`.

---

## Step 3 — Full batch run (ONE ARM per job; 12h wall)

Full-100 both-arms is ~20-26 GPU-hours and does NOT fit 12h, so run one arm per
job via the `ARMS` env var. Records append per-query, so a timeout still leaves
partial results.

```bash
cd ~/projects/def-enaskt/hhpfiona/CSC494 && mkdir -p runs

# If you edited the script on Windows, normalize line endings first:
dos2unix orchestration/run_ablation.slurm
bash -n orchestration/run_ablation.slurm && echo "syntax OK"

# no-context builds no graph (SBERT irrelevant) — reuse existing data, or:
ARMS=no-context  sbatch orchestration/run_ablation.slurm
ARMS=template    sbatch orchestration/run_ablation.slurm    # needs SBERT pre-cached
ARMS=llm-rewrite sbatch orchestration/run_ablation.slurm    # needs SBERT pre-cached

squeue -u $USER          # watch the queue
```

When a template/llm-rewrite job starts, confirm SBERT loaded (NOT Jaccard):

```bash
grep -i "SBERT\|Jaccard" runs/slurm_pluraltree_ablation_<jobid>.err | head
# want "Graph reconstruction using SBERT ..."; if "falling back to Jaccard",
# cancel and re-run the Step-0 pre-cache — the model wasn't cached.
```

Logs: `runs/slurm_pluraltree_ablation_<jobid>.out` (and `.err` for INFO logs).
Results: `runs/ablation_local_*.jsonl` and `*_summary.json`.

> The batch script stages the model from `MODEL_SRC` (a /scratch dir) to
> node-local SSD before loading. `MODEL_SRC` defaults to
> `$SCRATCH/models/llama31-8b`; override if weights live elsewhere. A bare HF id
> (not a dir) skips staging and loads from `$HF_HOME`.

---

## Module versions — adjust to what Narval currently exposes

The `module load` lines are best-guess. Confirm with `module spider <name>` and
swap versions as needed. Not optional: **load `rust` before installing
`vllm`/`tokenizers`** (caused "invalid wheel" failures). Prefer `--no-index`
(Alliance wheels) over PyPI.
