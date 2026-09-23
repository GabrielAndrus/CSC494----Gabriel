#!/bin/bash
# LOGIN NODE ONLY (compute nodes are offline). One-time, ~90GB into $SCRATCH. Resumable.
# Prerequisites (manual, in a browser, with your Hugging Face account):
#   accept the license on each model page:  meta-llama/Meta-Llama-3-8B-Instruct,
#   google/gemma-2-9b-it, google/gemma-2-27b-it  -- then `hf auth login` with a READ token.
# Uses the venv from NARVAL.md Step 0:  source $SCRATCH/precache_env/bin/activate
set -euo pipefail
mkdir -p "$SCRATCH/models"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
hf download meta-llama/Meta-Llama-3-8B-Instruct --local-dir "$SCRATCH/models/llama3-8b-instruct" \
    --exclude "original/*"
hf download google/gemma-2-9b-it  --local-dir "$SCRATCH/models/gemma2-9b-it"
hf download google/gemma-2-27b-it --local-dir "$SCRATCH/models/gemma2-27b-it"
du -sh "$SCRATCH"/models/*
