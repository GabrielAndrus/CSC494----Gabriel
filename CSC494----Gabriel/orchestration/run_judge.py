"""
run_judge.py — PASS 2 of the two-pass on-task evaluation.

Loads ONE (big) judge/extractor model, reads answers produced by run_generate.py,
decomposes + scores each answer with CulFiT's cultural P/R/F1, and writes the
per-item scores and a summary. The judge model is recorded in the output so the
comparability caveat travels with the numbers.

IMPORTANT — comparability:
  CulFiT's paper numbers were produced with a gpt-4o-mini judge/extractor. Here
  the judge is a local model. Absolute F1 is therefore NOT comparable to their
  Table 1. The defensible claim is the WITHIN-PROTOCOL DELTA: every system is
  judged by the SAME local judge on the SAME items, so judge bias cancels in
  (agentic - baseline). The summary records judge_model/extractor_model and a
  `comparability` flag to keep this explicit.

Usage — score a generated answers file with a big judge:
    python -m orchestration.run_judge \
        --answers runs/answers_<ts>.jsonl \
        --judge_model meta-llama/Llama-3.3-70B-Instruct \
        [--load_in_4bit]

Usage — reliability calibration (one-time, ~15 min of human labeling):
    # 1) dump ~20 real judge pairs to a CSV with a blank human_label column
    python -m orchestration.run_judge --judge_model <m> \
        --queries CulFiT/GlobalCultureQA/eval_set_n100.jsonl \
        --dump_judge_pairs 20 --pairs_csv runs/judge_pairs.csv
    # 2) fill in the human_label column by hand, then:
    python -m orchestration.run_judge \
        --score_reliability runs/judge_pairs.csv

Outputs:
    runs/taskeval_<ts>.jsonl          (per (system, item) score)
    runs/taskeval_<ts>_summary.json   (macro P/R/F1 per system + deltas + provenance)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from datetime import datetime

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from orchestration import bootstrap
bootstrap.install()

from orchestration.task_eval import (
    TaskEvaluator, dump_judge_pairs, score_judge_reliability,
    _USING_CULFIT_PROMPTS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("run_judge")


def load_judge_backend(judge_model, dtype, load_in_4bit):
    """Load the judge model. 4-bit lets a 70B fit one ~40GB GPU (quality cost)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from orchestration.llm_backend import LocalBackend

    logger.info("Loading judge tokenizer: %s", judge_model)
    tok = AutoTokenizer.from_pretrained(judge_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {"device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        logger.info("Loading judge in 4-bit NF4 (fits ~40GB; some quality cost).")
    else:
        torch_dtype = getattr(torch, dtype, torch.bfloat16)
        try:
            kwargs["dtype"] = torch_dtype
            model = AutoModelForCausalLM.from_pretrained(judge_model, **kwargs)
        except TypeError:
            kwargs.pop("dtype", None)
            kwargs["torch_dtype"] = torch_dtype
            model = AutoModelForCausalLM.from_pretrained(judge_model, **kwargs)
        model.eval()
        return LocalBackend(model_obj=model, tokenizer_obj=tok, model_name=judge_model)

    model = AutoModelForCausalLM.from_pretrained(judge_model, **kwargs)
    model.eval()
    return LocalBackend(model_obj=model, tokenizer_obj=tok, model_name=judge_model)


def load_answers(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    logger.info("Loaded %d answer records from %s", len(recs), path)
    return recs


def load_items(path, limit):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items[:limit] if limit else items


def score_answers(answer_recs, evaluator, judge_model):
    """Score each pre-generated answer; aggregate macro P/R/F1 per system."""
    by_system = defaultdict(list)
    per_item = []
    for rec in answer_recs:
        scored = evaluator.score_answer(
            rec.get("answer", ""), rec.get("verified_points", []),
            group=rec.get("location", "Unknown"),
            question=rec.get("query", ""))
        out = {
            "idx": rec.get("idx"), "system": rec.get("system"),
            "query": rec.get("query"), "location": rec.get("location"),
            "generator_model": rec.get("generator_model"),
            "judge_model": judge_model,
            "answer": rec.get("answer"),
            "precision": scored["precision"], "recall": scored["recall"],
            "f1": scored["f1"],
            "n_answer_units": scored.get("n_answer_units"),
            "n_golden_units": scored.get("n_golden_units"),
            "covered_golden": scored.get("covered_golden", []),
            "missed_golden": scored.get("missed_golden", []),
        }
        per_item.append(out)
        by_system[rec.get("system")].append(out)
        logger.info("[%s] item %s  P=%.3f R=%.3f F1=%.3f",
                    rec.get("system"), rec.get("idx"),
                    scored["precision"], scored["recall"], scored["f1"])

    summaries = {}
    for sys_name, rs in by_system.items():
        n = len(rs) or 1
        summaries[sys_name] = {
            "system": sys_name, "n_items": len(rs),
            "precision": sum(r["precision"] for r in rs) / n,
            "recall": sum(r["recall"] for r in rs) / n,
            "f1": sum(r["f1"] for r in rs) / n,
        }
    return per_item, summaries


def run_scoring(answers_path, judge_model, dtype, load_in_4bit):
    answer_recs = load_answers(answers_path)
    backend = load_judge_backend(judge_model, dtype, load_in_4bit)
    evaluator = TaskEvaluator(backend)

    per_item, summaries = score_answers(answer_recs, evaluator, judge_model)

    base = summaries.get("culfit_baseline")
    deltas = {}
    if base:
        for name, s in summaries.items():
            if name == "culfit_baseline":
                continue
            deltas[name] = {
                "d_precision": s["precision"] - base["precision"],
                "d_recall": s["recall"] - base["recall"],
                "d_f1": s["f1"] - base["f1"],
            }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("runs", exist_ok=True)
    jsonl_path = f"runs/taskeval_{ts}.jsonl"
    summary_path = f"runs/taskeval_{ts}_summary.json"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in per_item:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "answers_file": answers_path,
        "judge_model": judge_model,
        "extractor_model": judge_model,  # same model decomposes + judges
        "generator_models": sorted({r.get("generator_model") for r in answer_recs}),
        "used_culfit_prompts": _USING_CULFIT_PROMPTS,
        "comparability": (
            "DELTA-ONLY: absolute F1 is judged by a local model, NOT gpt-4o-mini, "
            "so it is not comparable to CulFiT's Table 1. Compare "
            "(agentic - baseline) within this same-judge protocol."),
        "systems": summaries,
        "deltas_vs_culfit_baseline": deltas,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s and %s", jsonl_path, summary_path)
    print("\n=== TASK EVAL SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def run_dump_pairs(judge_model, dtype, load_in_4bit, queries, n_pairs, pairs_csv, limit):
    items = load_items(queries, limit)
    backend = load_judge_backend(judge_model, dtype, load_in_4bit)
    evaluator = TaskEvaluator(backend)
    os.makedirs(os.path.dirname(pairs_csv) or "runs", exist_ok=True)
    n = dump_judge_pairs(evaluator, items, pairs_csv, n_pairs=n_pairs)
    print(f"\nWrote {n} judge pairs to {pairs_csv}")
    print("Now fill the 'human_label' column (Yes/No) by hand, then run:")
    print(f"    python -m orchestration.run_judge --score_reliability {pairs_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--answers", default=None, help="answers_*.jsonl from run_generate")
    p.add_argument("--judge_model", default="meta-llama/Llama-3.3-70B-Instruct")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--load_in_4bit", action="store_true",
                   help="4-bit NF4 so a 70B fits one ~40GB GPU (quality cost)")
    # reliability harness
    p.add_argument("--queries", default=None, help="eval set (for --dump_judge_pairs)")
    p.add_argument("--dump_judge_pairs", type=int, default=0,
                   help="dump this many (candidate,reference,verdict) pairs for labeling")
    p.add_argument("--pairs_csv", default="runs/judge_pairs.csv")
    p.add_argument("--score_reliability", default=None,
                   help="path to a human-labeled judge_pairs.csv to score agreement")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    # Mode 1: score a labeled reliability CSV (no model load needed).
    if args.score_reliability:
        rel = score_judge_reliability(args.score_reliability)
        print("\n=== JUDGE RELIABILITY ===")
        print(json.dumps(rel, indent=2, ensure_ascii=False))
        if rel["agreement"] is not None:
            print(f"\nJudge<->human agreement: {rel['n_agree']}/{rel['n_labeled']} "
                  f"= {rel['agreement']:.0%}")
        return

    # Mode 2: dump judge pairs for labeling (needs the judge model).
    if args.dump_judge_pairs > 0:
        if not args.queries:
            raise SystemExit("--dump_judge_pairs needs --queries <eval_set.jsonl>")
        run_dump_pairs(args.judge_model, args.dtype, args.load_in_4bit,
                       args.queries, args.dump_judge_pairs, args.pairs_csv, args.limit)
        return

    # Mode 3: score generated answers (the main judge pass).
    if not args.answers:
        raise SystemExit("provide --answers <file>, or --dump_judge_pairs, or --score_reliability")
    run_scoring(args.answers, args.judge_model, args.dtype, args.load_in_4bit)


if __name__ == "__main__":
    main()
