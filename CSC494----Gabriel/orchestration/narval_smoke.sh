# ============================================================================
# TWO-PASS SMOKE TEST (interactive salloc) — Qwen2.5-7B judges Llama-3.1-8B
# Run these BLOCK BY BLOCK, not as a script. Goal: 1 item through each pass.
# ============================================================================

# --- 0. Grab one GPU interactively (opportunistic; may wait) ---------------
salloc --account=def-enaskt --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:50:00
# If the wait is long, in another shell:  squeue -u $USER --start

# --- 1. Once on the GPU node (prompt changes narval3 -> ngXXXXX) ------------
cd ~/projects/def-enaskt/hhpfiona/CSC494

module purge
module load StdEnv/2023 gcc/12.3 rust/1.76.0 python/3.11 arrow/16 cuda/12.2

# Node-local venv (fast, avoids stale metadata). requests IS in the list.
python -m venv $SLURM_TMPDIR/env && source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index requests torch transformers tokenizers sentence-transformers \
    accelerate openai pandas openpyxl tenacity tqdm pydantic python-dotenv

export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# --- 2. Stage BOTH models to node-local SSD (Lustre load is slow) ----------
time cp -r $SCRATCH/models/llama31-8b $SLURM_TMPDIR/llama31-8b
time cp -r $SCRATCH/models/qwen25-7b  $SLURM_TMPDIR/qwen25-7b

# --- 3. Verify the code drop (no GPU needed) -------------------------------
python -m orchestration.check_setup      # expect: PASS — setup looks good.

# --- 4. PASS 1 SMOKE — generate answers for 1 item, both systems -----------
# --smoke = 1 item; small + fast. Generator = Llama-3.1-8B on local SSD.
python -m orchestration.run_generate \
    --model $SLURM_TMPDIR/llama31-8b \
    --queries CulFiT/GlobalCultureQA/eval_set_n100.jsonl \
    --systems culfit_baseline,agentic_sequential \
    --smoke
# -> prints the answers file path, e.g. runs/answers_<ts>.jsonl
# Eyeball it: did both systems produce a non-empty "answer"?
ls -t runs/answers_*.jsonl | head -1 | xargs cat | python -m json.tool 2>/dev/null | head -40

# --- 5. PASS 2 SMOKE — judge those answers with Qwen2.5-7B ------------------
# Replace <ts> with the file from step 4 (or use the $(ls ...) form).
ANSWERS=$(ls -t runs/answers_*.jsonl | head -1)
python -m orchestration.run_judge \
    --answers "$ANSWERS" \
    --judge_model $SLURM_TMPDIR/qwen25-7b
# -> prints the summary with P/R/F1 per system + deltas + provenance.
# Sanity: numbers in [0,1], culfit_baseline present, comparability=DELTA-ONLY.

# --- 6. RELIABILITY SMOKE — dump a few judge pairs (tiny, just to test wiring)
python -m orchestration.run_judge \
    --judge_model $SLURM_TMPDIR/qwen25-7b \
    --queries CulFiT/GlobalCultureQA/eval_set_n100.jsonl \
    --dump_judge_pairs 4 --pairs_csv runs/judge_pairs_smoke.csv
head -5 runs/judge_pairs_smoke.csv     # confirm columns + a blank human_label

exit   # releases the salloc allocation
