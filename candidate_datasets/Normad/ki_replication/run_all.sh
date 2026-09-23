#!/usr/bin/env bash
# Runs the full Ki et al. single-model replication (without_rot + with_rot) for each model.
# Requires OPENROUTER_API_KEY in the environment. Resumable: re-running skips finished items.
set -u
cd "$(dirname "$0")"
run_model () {  # $1=model id  $2=max_tokens
  local slug="${1//\//_}"
  for t in without_rot with_rot; do
    python3 run_single_model.py --model "$1" --type "$t" --max-tokens "$2" --workers "${WORKERS:-32}" \
      --out "results/${slug}__${t}.jsonl"
  done
}
run_model google/gemma-2-27b-it 32 &                    # paper: Gemma-2-9B used max_new_tokens=32
run_model meta-llama/llama-3.1-8b-instruct 256 &        # paper: Llama-3 used max_new_tokens=256
run_model qwen/qwen-2.5-7b-instruct 256 &               # extension (not in paper)
wait
