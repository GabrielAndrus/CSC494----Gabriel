#!/usr/bin/env python
"""
run_debate.py -- Iteration 2 (MULTI-AGENT), pass 1: Debate-Only between Gemma-2-9B (agent A)
and Llama-3-8B (agent B), one round of feedback, as in Ki et al. (paper Sec. 3.2 / Table 2,
row LLAMA-3 + GEMMA-2). Six generations per item: initial decision, feedback, final decision,
for each agent. The judge (Gemma-2-27B) is a separate pass: run_judge.py.

    --variant released : reproduces the released code's prompt-substitution behaviour
                         (rule-of-thumb never inserted, etc. -- see common.py)
    --variant fixed    : the prompts exactly as the paper's Appendix A describes them

Batched greedy decoding (Ki's scripts go one item at a time; see hf_gen.py). Resumable:
re-running skips items already in --out.

    python run_debate.py --gemma-dir $SLURM_TMPDIR/gemma2-9b-it \
        --llama-dir $SLURM_TMPDIR/llama3-8b-instruct --variant fixed --out ../results_narval/debate_fixed.jsonl
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time

from common import HERE, load_rows, done_ids, debate_chunk
from hf_gen import HFGenerator


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemma-dir", required=True)
    ap.add_argument("--llama-dir", required=True)
    ap.add_argument("--variant", required=True, choices=["released", "fixed"])
    ap.add_argument("--input", default=os.path.join(os.path.dirname(HERE), "data", "normad_ki.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=1024, help="Ki et al.: 1024 for every stage")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="8 items, batch 4")
    args = ap.parse_args()
    if args.smoke:
        args.limit, args.batch_size = 8, 4

    import torch
    n_gpu = torch.cuda.device_count()
    dev_a, dev_b = ({"": 0}, {"": 1}) if n_gpu >= 2 else ({"": 0}, {"": 0})
    print(f"[debate:{args.variant}] GPUs visible: {n_gpu} -> "
          f"{'one model per GPU' if n_gpu >= 2 else 'both models on GPU 0 (needs ~34GB weights)'}", flush=True)

    A = HFGenerator(args.gemma_dir, "raw", device=dev_a, eager_attention=True)   # Gemma-2-9B: raw prompt
    B = HFGenerator(args.llama_dir, "chat", device=dev_b)                        # Llama-3-8B: chat template

    rows = load_rows(args.input, args.limit)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    finished = done_ids(args.out)
    todo = [r for r in rows if r["ID"] not in finished]
    print(f"[debate:{args.variant}] {len(rows)} items, {len(finished)} done, {len(todo)} to run", flush=True)

    mnt, t0 = args.max_new_tokens, time.time()
    sampled_prompts = {}
    for s in range(0, len(todo), args.batch_size):
        chunk = todo[s:s + args.batch_size]
        bs = len(chunk)
        records, first = debate_chunk(A, B, chunk, args.variant, mnt)
        if s == 0:  # keep the exact prompts of the first item so the substitution behaviour is auditable
            sampled_prompts = first
        with open(args.out, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done = s + bs
        print(f"  {done}/{len(todo)}  ({time.time() - t0:.0f}s elapsed, "
              f"{(time.time() - t0) / done:.1f}s/item)", flush=True)

    if sampled_prompts:
        with open(args.out + ".prompts_sample.json", "w", encoding="utf-8") as f:
            json.dump(sampled_prompts, f, indent=2, ensure_ascii=False)
    print("[debate] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
