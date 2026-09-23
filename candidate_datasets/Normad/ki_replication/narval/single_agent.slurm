#!/bin/bash
#SBATCH --job-name=ki_single
#SBATCH --account=def-enaskt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
# Iteration 1 (SINGLE AGENT): Ki et al. single-model baseline, Llama-3-8B (chat) and
# Gemma-2-9B (raw), without and with rule-of-thumb. Submit from ki_replication/narval/:
#     mkdir -p logs && sbatch single_agent.slurm
# Resumable: resubmitting skips finished items.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source env.sh
LLAMA=$(stage llama3-8b-instruct); GEMMA=$(stage gemma2-9b-it)
[ -f ../data/normad_ki.jsonl ] || { echo "FATAL: ../data/normad_ki.jsonl missing (run fetch_data.sh)"; exit 1; }

python run_single_agent.py --model-dir "$LLAMA" --model-id meta-llama/Meta-Llama-3-8B-Instruct --format chat
python run_single_agent.py --model-dir "$GEMMA" --model-id google/gemma-2-9b-it --format raw --eager-attn
python ../score.py \
    ../results_narval/meta-llama_Meta-Llama-3-8B-Instruct__without_rot.jsonl \
    ../results_narval/meta-llama_Meta-Llama-3-8B-Instruct__with_rot.jsonl \
    ../results_narval/google_gemma-2-9b-it__without_rot.jsonl \
    ../results_narval/google_gemma-2-9b-it__with_rot.jsonl \
    --out ../results_narval/summary_single_agent.json
echo "=== single agent done $(date) ==="
