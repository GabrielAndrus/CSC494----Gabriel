#!/bin/bash
#SBATCH --job-name=ki_debate
#SBATCH --account=def-enaskt
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
# Iteration 2 (MULTI-AGENT), pass 1: Debate-Only, Gemma-2-9B + Llama-3-8B, both variants.
# 2 GPUs = one 40GB GPU per model (safe). If the queue is too slow, try 1 GPU with a smaller
# batch:   sbatch --gres=gpu:1 --export=ALL,BATCH=4 debate.slurm
# Variants run one after another; both are resumable. Submit from ki_replication/narval/.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source env.sh
GEMMA=$(stage gemma2-9b-it); LLAMA=$(stage llama3-8b-instruct)
[ -f ../data/normad_ki.jsonl ] || { echo "FATAL: ../data/normad_ki.jsonl missing (run fetch_data.sh)"; exit 1; }
mkdir -p ../results_narval
for VARIANT in ${VARIANTS:-fixed released}; do
    echo "=== debate variant: $VARIANT  $(date) ==="
    python run_debate.py --gemma-dir "$GEMMA" --llama-dir "$LLAMA" --variant "$VARIANT" \
        --batch-size "${BATCH:-16}" --out ../results_narval/debate_${VARIANT}.jsonl
done
echo "=== debate pass done $(date) ==="
