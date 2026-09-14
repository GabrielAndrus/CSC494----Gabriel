#!/usr/bin/env python
"""
run_pipeline_normad_mock.py -- Exercise the PluralTree orchestrator end-to-end
on the normalized NormAd sample (normad_eval_set.jsonl).

Default (--mode mock, no flags needed) is a STRUCTURAL / INTEGRATION check, not
a quality evaluation. It answers "does the pipeline correctly ingest
NormAd-shaped records and run all three topologies to completion" -- not "how
good are the model's answers." Reuses run_ablation.py's mock responders /
build_orchestrator UNCHANGED, so it exercises exactly the same mock heuristics
as the project's existing mock harness -- no new pipeline behavior, only a
different (file-based) query source.

--mode gemini runs an ACTUAL quality read (real Agent A/B calls via Gemini,
needs GEMINI_API_KEY -- see NORMAD_REPORT.md Sec 7 and orchestration/
smoke_test_gemini.py to verify the key first). This is the one non-mock mode
this script supports locally without the Narval GPU cluster; for a
Llama/Qwen-backed run see orchestration/NARVAL.md instead.

Usage (from the CSC494 repo root, with the project venv):
    venv/bin/python3 candidate_datasets/Normad/run_pipeline_normad_mock.py \
        candidate_datasets/Normad/normad_eval_set.jsonl

    # real run, once GEMINI_API_KEY is set:
    venv/bin/python3 candidate_datasets/Normad/run_pipeline_normad_mock.py \
        candidate_datasets/Normad/normad_eval_set.jsonl --mode gemini
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from orchestration.run_ablation import build_orchestrator  # noqa: E402


def load_items(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run(queries_path: str, max_loops: int = 3, mode: str = "mock", model: str = "n/a"):
    items = load_items(queries_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    tag = "mock" if mode == "mock" else f"{mode}_{model}".replace("/", "_")
    jsonl_path = os.path.join(out_dir, f"normad_{tag}_run_{ts}.jsonl")
    records = []

    for q in items:
        for topology in ("static", "parallel", "sequential"):
            orch, a_backend, b_backend = build_orchestrator(mode, model, q)
            if topology == "static":
                result = orch.static_integration(q["query"], q["ground_truth"])
            elif topology == "parallel":
                result = orch.parallel_debate(q["query"], q["ground_truth"])
            else:
                result = orch.sequential_debate(q["query"], q["ground_truth"], max_loops=max_loops)

            rec = {
                "location": q["location"], "sub_topic": q["sub_topic"],
                "gold_label": q.get("_normad_source", {}).get("gold_label"),
                "topology": topology,
                "n_final_paths": len(result.get("final_paths", [])),
                "trace": result["trace"],
                "llm_calls": {"agent_a": getattr(a_backend, "call_count", None),
                              "agent_b": getattr(b_backend, "call_count", None)},
            }
            records.append(rec)
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    summary = {}
    for topo in ("static", "parallel", "sequential"):
        rs = [r for r in records if r["topology"] == topo]
        n = len(rs) or 1
        summary[topo] = {
            "n_runs": len(rs),
            "approval_rate": sum(r["trace"]["final_approved"] for r in rs) / n,
            "avg_mean_precision": sum(r["trace"]["final_mean_precision"] for r in rs) / n,
            "avg_loops": sum(r["trace"]["loops"] for r in rs) / n,
            "avg_repairs": sum(r["trace"]["repairs"] for r in rs) / n,
        }
    # Break down approval by NormAd gold label. In --mode mock this should NOT
    # track yes/no/neutral at all -- the canned responder (keys off "history"/
    # "tradition"/"culinary" words) knows nothing about social-acceptability
    # judgments, so a flat approval rate across labels is the expected result,
    # not a bug (see NORMAD_REPORT.md Sec 5). In a real mode (gemini/api/local)
    # this breakdown is the actually meaningful number: does the system's
    # approval track the real yes/no/neutral verdict.
    by_label = {}
    for lbl in ("yes", "no", "neutral"):
        rs = [r for r in records if r["gold_label"] == lbl and r["topology"] == "static"]
        n = len(rs) or 1
        by_label[lbl] = {
            "n": len(rs),
            "approval_rate": sum(r["trace"]["final_approved"] for r in rs) / n,
        }

    summary_path = os.path.join(out_dir, f"normad_{tag}_run_{ts}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"by_topology": summary, "by_gold_label_static_only": by_label}, f, indent=2)

    print(f"Wrote {jsonl_path}")
    print(f"Wrote {summary_path}")
    print(json.dumps({"by_topology": summary, "by_gold_label_static_only": by_label}, indent=2))
    return jsonl_path, summary_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("queries_path")
    ap.add_argument("--max_loops", type=int, default=3)
    ap.add_argument("--mode", default="mock", choices=["mock", "api", "gemini"],
                    help="mock (default, offline) or gemini (needs GEMINI_API_KEY; "
                         "see NORMAD_REPORT.md Sec 7 / NARVAL.md for setup)")
    ap.add_argument("--model", default=None,
                    help="Defaults to gemini-2.0-flash for --mode gemini, gpt-4o for --mode api")
    args = ap.parse_args()
    default_models = {"api": "gpt-4o", "gemini": "gemini-2.0-flash"}
    model = args.model or default_models.get(args.mode, "n/a")
    run(args.queries_path, args.max_loops, mode=args.mode, model=model)
