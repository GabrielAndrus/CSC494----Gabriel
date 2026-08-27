#!/bin/bash
#SBATCH --account=def-enaskt
#SBATCH --job-name=pt_judge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm_judge_%j.out
#SBATCH --error=runs/slurm_judge_%j.out

# ============================================================================
# PASS 2 — judge a saved answers file with Qwen2.5-7B. Shorter than generation.
# Requires ANSWERS to be set to the answers_*.jsonl produced by pass 1:
#   sbatch --export=ALL,ANSWERS=runs/answers_20260713_XXXXXX.jsonl judge.slurm
# ============================================================================
set -euo pipefail

cd ~/projects/def-enaskt/hhpfiona/CSC494

if [ -z "${ANSWERS:-}" ]; then
    echo "ERROR: set ANSWERS=runs/answers_<ts>.jsonl via --export=ALL,ANSWERS=..."
    exit 1
fi

module purge
module load StdEnv/2023 gcc/12.3 rust/1.76.0 python/3.11 arrow/16 cuda/12.2

python -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index requests torch transformers tokenizers sentence-transformers \
    accelerate openai pandas openpyxl tenacity tqdm pydantic python-dotenv

export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Stage the judge to node-local SSD.
cp -r $SCRATCH/models/qwen25-7b $SLURM_TMPDIR/qwen25-7b

srun python -m orchestration.run_judge \
    --answers "${ANSWERS}" \
    --judge_model $SLURM_TMPDIR/qwen25-7b

echo "=== judge job done ==="
