#!/usr/bin/env python
"""
run_single_agent.py -- Iteration 1 (SINGLE AGENT): Ki et al. "Single Model" baseline
(paper Table 1, columns "Si (w/o)" and "Si (w/)") with local HuggingFace weights on Narval.

Mirrors single_llm/single_model/{llama3,gemma}.py: same prompts (verbatim), same data, same
per-model format (Llama-3 = chat template, Gemma-2 = raw string), greedy decoding,
max_new_tokens 256 (Llama-3) / 32 (Gemma). Output JSONL uses the same schema as the
OpenRouter runs, so `python ../score.py <files>` scores it (paper scorer + strict parser).

    python run_single_agent.py --model-dir $SLURM_TMPDIR/llama3-8b-instruct \
        --model-id meta-llama/Meta-Llama-3-8B-Instruct --format chat
    python run_single_agent.py --model-dir $SLURM_TMPDIR/gemma2-9b-it \
        --model-id google/gemma-2-9b-it --format raw --eager-attn
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time

from common import HERE, SINGLE_PROMPTS, COUNTRY_CAPITALIZED, load_rows, done_ids
from hf_gen import HFGenerator


def build(row, rot_mode):
    p = SINGLE_PROMPTS[rot_mode].replace("{{country}}", COUNTRY_CAPITALIZED[row["Country"]]) \
        .replace("{{story}}", row["Story"])
    if rot_mode == "with_rot":
        p = p.replace("{{rot}}", row["Rule-of-Thumb"])
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-id", required=True, help="HF id used as the label in outputs")
    ap.add_argument("--format", required=True, choices=["chat", "raw"])
    ap.add_argument("--types", nargs="+", default=["without_rot", "with_rot"],
                    choices=["without_rot", "with_rot"])
    ap.add_argument("--input", default=os.path.join(os.path.dirname(HERE), "data", "normad_ki.jsonl"))
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(HERE), "results_narval"))
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="default: 256 for chat (Llama-3), 32 for raw (Gemma), as in Ki's scripts")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eager-attn", action="store_true", help="recommended for Gemma-2")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="8 items, batch 4")
    args = ap.parse_args()
    if args.smoke:
        args.limit, args.batch_size = 8, 4
    mnt = args.max_new_tokens or (256 if args.format == "chat" else 32)

    rows = load_rows(args.input, args.limit)
    os.makedirs(args.out_dir, exist_ok=True)
    slug = args.model_id.replace("/", "_")
    print(f"[single] {args.model_id}: {len(rows)} items, fmt={args.format}, "
          f"max_new_tokens={mnt}, batch={args.batch_size}", flush=True)

    gen = HFGenerator(args.model_dir, args.format, device="auto", eager_attention=args.eager_attn)

    for typ in args.types:
        out_path = os.path.join(args.out_dir, f"{slug}__{typ}{'__smoke' if args.smoke else ''}.jsonl")
        finished = done_ids(out_path)
        todo = [r for r in rows if r["ID"] not in finished]
        print(f"[single] {typ}: {len(finished)} done, {len(todo)} to run", flush=True)
        t0 = time.time()
        for s in range(0, len(todo), args.batch_size * 8):  # checkpoint every 8 batches
            chunk = todo[s:s + args.batch_size * 8]
            outs = gen.generate([build(r, typ) for r in chunk], mnt, args.batch_size)
            with open(out_path, "a", encoding="utf-8") as f:
                for r, o in zip(chunk, outs):
                    text = o
                    if args.format == "raw":  # Ki's Gemma script: cut at first "<"
                        text = text.split("<")[0].strip()
                    f.write(json.dumps({
                        "ID": r["ID"], "Country": r["Country"], "Gold Label": r["Gold Label"],
                        "model": args.model_id, "type": typ, "backend": "hf-local",
                        "generation": text, "generation_raw": o,
                    }, ensure_ascii=False) + "\n")
            print(f"  {min(s + len(chunk), len(todo))}/{len(todo)}  "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)
    print("[single] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
