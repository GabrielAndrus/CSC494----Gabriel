#!/bin/bash
#SBATCH --job-name=pluraltree_align
#SBATCH --account=def-enaskt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm_%x_%j.out
#SBATCH --error=runs/slurm_%x_%j.err

# PluralTree TASK-ALIGNMENT DIAGNOSTIC - Narval.
#
# Answers the professor's July 9 question: does Agent A's augmentation carry
# signal for Agent B's GlobalCultureQA task? Runs a paired no-context vs.
# augmented on-task F1 per query over the n=100 set, in the static topology (no
# repair loop), and writes per-query deltas + traces.
#
# Unlike the topology ablation, this runs BOTH arms in one pass (paths are
# generated once per query and reused), so it's a single job, ~11k judge calls
# for n=100 -> fits comfortably in 6h. Records append per-query, so a timeout
# still leaves partial results.
#
# Submit:
#     cd CSC494 && mkdir -p runs
#     sbatch orchestration/run_align_diag.slurm
#
# For a cheap first check use salloc + --smoke instead (see NARVAL.md Step 2):
#     python -m orchestration.task_alignment_diag \
#         --mode local --model $SLURM_TMPDIR/llama31-8b \
#         --queries CulFiT/GlobalCultureQA/eval_set_n100.jsonl --smoke
#
# If edited on Windows: run `dos2unix orchestration/run_align_diag.slurm` then
# `bash -n orchestration/run_align_diag.slurm` to confirm syntax before submit.

set -euo pipefail

echo "Job $SLURM_JOB_ID on $(hostname) starting at $(date)"
nvidia-smi || echo "WARNING: nvidia-smi unavailable"

# 1. Modules
module purge
module load StdEnv/2023
module load gcc/12.3
module load rust/1.76.0
module load python/3.11
module load arrow/16
module load cuda/12.2

# 2. Environment (node-local venv)
VENV="$SLURM_TMPDIR/pluraltree_env"
python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip
# `requests` is REQUIRED for SBERT (sentence-transformers imports it). Without it
# the augmented arm silently falls back to Jaccard token overlap for node merging.
pip install --no-index torch transformers tokenizers sentence-transformers \
    accelerate pandas openpyxl tenacity tqdm pydantic python-dotenv openai requests || {
    echo "Alliance wheels incomplete; retrying with PyPI for the gaps..."
    pip install torch transformers tokenizers sentence-transformers \
        accelerate pandas openpyxl tenacity tqdm pydantic python-dotenv openai requests
}
echo "--- installed packages ---"; pip list 2>/dev/null | grep -Ei \
    "torch|transformers|sentence|accelerate|requests|pydantic" || true

# 3. Offline caches
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL_SRC="${MODEL_SRC:-$SCRATCH/models/llama31-8b}"
QUERIES="${QUERIES:-CulFiT/GlobalCultureQA/eval_set_n100.jsonl}"
KP_MODE="${KP_MODE:-judge}"          # judge (semantic) | lexical (deterministic floor)
MAX_PATHS="${MAX_PATHS:-12}"         # cap Agent A paths per query
MAX_PATHS_SCORED="${MAX_PATHS_SCORED:-5}"   # judge calls = scored * n_points per arm
N="${N:-}"                           # optional query cap (empty = all 100)

# 3b. Stage model to node-local SSD (loading 16GB off Lustre is slow)
if [ -d "$MODEL_SRC" ]; then
    STAGED="$SLURM_TMPDIR/$(basename "$MODEL_SRC")"
    echo "Staging model: $MODEL_SRC -> $STAGED"
    time cp -r "$MODEL_SRC" "$STAGED"
    MODEL="$STAGED"
else
    echo "MODEL_SRC is not a local dir; using as-is: $MODEL_SRC"
    MODEL="$MODEL_SRC"
fi

# Fail early if the eval set isn't where we expect (don't waste the slot)
if [ ! -f "$QUERIES" ]; then
    echo "FATAL: queries file not found: $QUERIES" >&2
    exit 1
fi

# Fail fast if SBERT isn't cached — the augmented arm builds the graph, and a
# silent Jaccard fallback would weaken the very augmentation we're measuring.
python -c "import os; os.environ['HF_HUB_OFFLINE']='1'; from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('SBERT loads offline OK')" \
    || { echo "FATAL: SBERT not cached offline; run the Step-0 pre-cache on a login node first." >&2; exit 1; }

# 4. Run the diagnostic (both arms, one pass)
mkdir -p runs
echo "Align diag: MODEL=$MODEL QUERIES=$QUERIES KP_MODE=$KP_MODE MAX_PATHS=$MAX_PATHS MAX_PATHS_SCORED=$MAX_PATHS_SCORED N=${N:-all}"

N_ARG=()
if [ -n "$N" ]; then N_ARG=(--n "$N"); fi

python -m orchestration.task_alignment_diag \
    --mode local \
    --model "$MODEL" \
    --queries "$QUERIES" \
    --kp_mode "$KP_MODE" \
    --max_paths "$MAX_PATHS" \
    --max_paths_scored "$MAX_PATHS_SCORED" \
    "${N_ARG[@]}"

echo "Job $SLURM_JOB_ID finished at $(date)"
