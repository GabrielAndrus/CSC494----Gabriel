#!/bin/bash
# Cheap end-to-end check on an INTERACTIVE 2-GPU allocation (fails in minutes, not after a 12h queue):
#     salloc --account=def-enaskt --gres=gpu:2 --cpus-per-task=6 --mem=128G --time=01:00:00
#     cd <repo>/candidate_datasets/Normad/ki_replication/narval && bash smoke.sh
# 8 items through: single agent (Llama-3, Gemma-2-9B), both debate variants, and the 27B judge.
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
LLAMA=$(stage llama3-8b-instruct); GEMMA=$(stage gemma2-9b-it)
python run_single_agent.py --model-dir "$LLAMA" --model-id meta-llama/Meta-Llama-3-8B-Instruct --format chat --smoke
python run_single_agent.py --model-dir "$GEMMA" --model-id google/gemma-2-9b-it --format raw --eager-attn --smoke
for V in fixed released; do
    python run_debate.py --gemma-dir "$GEMMA" --llama-dir "$LLAMA" --variant $V --smoke --out ../results_narval/smoke_debate_${V}.jsonl
done
echo "--- audit: the released variant must show the unfilled placeholders ---"
grep -c '{{rot}}' ../results_narval/smoke_debate_released.jsonl.prompts_sample.json || true
J=$(stage gemma2-27b-it)
python run_judge.py --judge-dir "$J" --debate ../results_narval/smoke_debate_fixed.jsonl --smoke || true
python score_debate.py ../results_narval/smoke_debate_fixed.jsonl || true
echo "=== SMOKE OK: inspect ../results_narval/*smoke* before submitting the real jobs ==="
