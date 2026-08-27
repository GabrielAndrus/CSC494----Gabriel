#!/bin/bash
#SBATCH --account=def-enaskt
#SBATCH --job-name=pt_generate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=runs/slurm_generate_%j.out
#SBATCH --error=runs/slurm_generate_%j.out

# ============================================================================
# PASS 1 — generate answers for all systems on n=100. Long, GPU-bound.
# Resumable: if this times out, resubmit with RESUME set to the answers file
# it was writing (printed near the top of the log), and it skips finished items.
# ============================================================================
set -euo pipefail

cd ~/projects/def-enaskt/hhpfiona/CSC494

module purge
module load StdEnv/2023 gcc/12.3 rust/1.76.0 python/3.11 arrow/16 cuda/12.2

python -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index requests torch transformers tokenizers sentence-transformers \
    accelerate openai pandas openpyxl tenacity tqdm pydantic python-dotenv

export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Stage the generator to node-local SSD (fast load; Lustre is slow).
cp -r $SCRATCH/models/llama31-8b $SLURM_TMPDIR/llama31-8b

# Optional resume: set to a prior partial answers file to continue it.
#   sbatch --export=ALL,RESUME=runs/answers_20260713_XXXXXX.jsonl generate.slurm
RESUME_ARG=""
if [ -n "${RESUME:-}" ]; then
    RESUME_ARG="--resume ${RESUME}"
    echo "RESUMING from ${RESUME}"
fi

srun python -m orchestration.run_generate \
    --model $SLURM_TMPDIR/llama31-8b \
    --queries CulFiT/GlobalCultureQA/eval_set_n100.jsonl \
    --systems culfit_baseline,agentic_sequential,agentic_sequential_terse \
    ${RESUME_ARG}

echo "=== generation job done ==="
