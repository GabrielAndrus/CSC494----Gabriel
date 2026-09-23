#!/bin/bash
#SBATCH --job-name=ki_judge27b
#SBATCH --account=def-enaskt
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
# Gemma-2-27B (bf16 ~54GB -> needs 2 x 40GB GPUs). Two uses of the same model:
#   (a) single-agent reference row of paper Table 1 (55.8 / 79.2)
#   (b) the debate JUDGE, run on disagreements of every debate_*.jsonl that exists.
# Run AFTER debate.slurm finished (or chain with: sbatch --dependency=afterok:<jobid> ...).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source env.sh
J=$(stage gemma2-27b-it)
[ -f ../data/normad_ki.jsonl ] || { echo "FATAL: ../data/normad_ki.jsonl missing (run fetch_data.sh)"; exit 1; }

python run_single_agent.py --model-dir "$J" --model-id google/gemma-2-27b-it --format raw --eager-attn
for V in fixed released; do
    F=../results_narval/debate_${V}.jsonl
    if [ -f "$F" ]; then python run_judge.py --judge-dir "$J" --debate "$F"; else echo "skip judge: $F not found"; fi
done
for V in fixed released; do
    F=../results_narval/debate_${V}.jsonl
    [ -f "$F" ] && python score_debate.py "$F" | tee ../results_narval/summary_debate_${V}.txt
done
echo "=== judge/27B done $(date) ==="
