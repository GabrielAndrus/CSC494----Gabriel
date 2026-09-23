#!/usr/bin/env python
"""
run_judge.py -- Iteration 2 (MULTI-AGENT), pass 2: the Gemma-2-27B judge.

Per the paper (Sec. 3.2, Eq. 2): if the two agents' FINAL decisions are identical, that is the
aggregated decision; otherwise a judge LLM (Gemma-2-27B) reads the debate and decides. The
judge step is not in the released code; the prompt is the paper's Appendix A.3.4.

Only disagreements are sent to the judge. Gemma-2-27B in bf16 is ~54GB, so this needs
2 x 40GB GPUs (device_map="auto" splits it). Reads the pass-1 file, writes <debate>.judge.jsonl.

    python run_judge.py --judge-dir $SLURM_TMPDIR/gemma2-27b-it --debate ../results_narval/debate_fixed.jsonl
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time

from common import HERE, load_rows, done_ids, prompt_judge, paper_predict
from hf_gen import HFGenerator

LABELS = {"yes", "no", "neutral"}


def agreed_label(final_a: str, final_b: str):
    """Paper scorer's normalisation; returns the shared label, or None if they differ / are unparseable."""
    la, lb = paper_predict(final_a), paper_predict(final_b)
    return la if (la == lb and la in LABELS) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-dir", required=True)
    ap.add_argument("--debate", required=True, help="pass-1 JSONL from run_debate.py")
    ap.add_argument("--input", default=os.path.join(os.path.dirname(HERE), "data", "normad_ki.jsonl"))
    ap.add_argument("--out", default=None, help="default: <debate>.judge.jsonl")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--smoke", action="store_true", help="judge at most 8 disagreements")
    args = ap.parse_args()
    out_path = args.out or args.debate + ".judge.jsonl"

    rows = {r["ID"]: r for r in load_rows(args.input)}
    with open(args.debate, encoding="utf-8") as f:
        deb = [json.loads(l) for l in f if l.strip()]
    disagree = [d for d in deb if agreed_label(d["gemma_final"], d["llama3_final"]) is None]
    print(f"[judge] {len(deb)} debates, {len(disagree)} disagreements "
          f"({100 * len(disagree) / max(len(deb), 1):.1f}%) -> judge", flush=True)
    finished = done_ids(out_path)
    todo = [d for d in disagree if d["ID"] not in finished]
    if args.smoke:
        todo = todo[:8]
    print(f"[judge] {len(finished)} already judged, {len(todo)} to run", flush=True)
    if not todo:
        return 0

    J = HFGenerator(args.judge_dir, "raw", device="auto", eager_attention=True)
    t0 = time.time()
    for s in range(0, len(todo), args.batch_size * 4):
        chunk = todo[s:s + args.batch_size * 4]
        prompts = [prompt_judge(rows[d["ID"]], d["llama3_1"], d["gemma_1"], d["llama3_2"],
                                d["gemma_2"], d["llama3_final"], d["gemma_final"]) for d in chunk]
        outs = J.generate(prompts, args.max_new_tokens, args.batch_size)
        with open(out_path, "a", encoding="utf-8") as f:
            for d, o in zip(chunk, outs):
                f.write(json.dumps({"ID": d["ID"], "variant": d["variant"],
                                    "judge_generation": o.split("<")[0].strip(),
                                    "judge_generation_raw": o}, ensure_ascii=False) + "\n")
        print(f"  {min(s + len(chunk), len(todo))}/{len(todo)}  ({time.time() - t0:.0f}s)", flush=True)
    print("[judge] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
