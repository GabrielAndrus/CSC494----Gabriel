#!/bin/bash
# Source (don't execute) from a SLURM job or an interactive `salloc` shell on a GPU node:
#     source env.sh
# Builds a node-local venv from Alliance wheels, turns on offline mode, and defines stage().
# Compute nodes have NO internet: models must already be in $SCRATCH/models (see precache_models.sh).

module purge
module load StdEnv/2023 gcc/12.3 rust/1.76.0 python/3.11 arrow/16 cuda/12.2

VENV="$SLURM_TMPDIR/ki_env"
python -m venv "$VENV" && source "$VENV/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index torch transformers tokenizers accelerate

# Fail fast (seconds, not after a queue wait): Gemma-2 needs transformers >= 4.42.
python - <<'PY'
import sys, torch, transformers
v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
print(f"torch {torch.__version__}  transformers {transformers.__version__}  cuda={torch.cuda.is_available()}  gpus={torch.cuda.device_count()}")
if v < (4, 42):
    sys.exit("transformers < 4.42 cannot load Gemma-2. Ask the cluster for a newer wheel (`pip download transformers`).")
if not torch.cuda.is_available():
    sys.exit("No CUDA device visible - are you on a GPU node?")
PY

export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# stage <dirname-under-$SCRATCH/models>  -> copies to node-local SSD, echoes the local path
stage() {
    local src="$SCRATCH/models/$1" dst="$SLURM_TMPDIR/$1"
    if [ ! -d "$src" ]; then echo "FATAL: $src missing (run precache_models.sh on a login node)" >&2; return 1; fi
    if [ ! -d "$dst" ]; then echo "staging $1 ..." >&2; cp -r "$src" "$dst"; fi
    echo "$dst"
}
