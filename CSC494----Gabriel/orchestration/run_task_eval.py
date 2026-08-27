"""
run_task_eval.py — SINGLE-PASS on-task eval (dev / mock / same-model runs).

This runs generation AND judging with ONE model in one process. It's the right
tool for offline mock verification and for runs where the generator and judge are
the same model. For the cluster setup with a SMALL generator and a BIGGER judge,
use the TWO-PASS flow instead:

    python -m orchestration.run_generate --model <small> ...   # -> answers_*.jsonl
    python -m orchestration.run_judge    --answers answers_*.jsonl --judge_model <big>

The two-pass flow decouples cheap generation from the expensive judge, lets you
re-judge saved answers with different judges without regenerating, and records
judge provenance + the delta-only comparability caveat in its summary.

Run offline to verify plumbing:
    python run_task_eval.py --mode mock --systems culfit_baseline,agentic_sequential
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime

from task_eval import (
    TaskEvaluator, culfit_baseline_provider, agentic_answer_provider,
    evaluate_system,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("taskeval")


# --------------------------------------------------------------------------- #
# Eval set loading (same JSONL contract as run_local.load_queries).           #
# --------------------------------------------------------------------------- #
_SMOKE_ITEMS = [
    {"query": "What is a traditional Indonesian breakfast and what does it involve?",
     "location": "Indonesia", "sub_topic": "breakfast",
     "ground_truth": {"location": "Indonesia", "sub_topic": "breakfast",
         "verified_points": [
             "Bubur Ayam is a traditional rice porridge eaten in the morning.",
             "Nasi goreng is sometimes eaten at breakfast in Indonesia.",
             "Breakfast in Indonesia often includes rice-based dishes."]}},
    {"query": "How is afternoon tea observed as a social custom in England?",
     "location": "England", "sub_topic": "beverages",
     "ground_truth": {"location": "England", "sub_topic": "beverages",
         "verified_points": [
             "Tea is commonly served with milk in England.",
             "Afternoon tea is a recognised social custom.",
             "Afternoon tea is typically served in the mid-afternoon."]}},
]


def load_items(path: str | None, smoke: bool, limit: int | None):
    if not path:
        items = _SMOKE_ITEMS
    else:
        items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        logger.info("Loaded %d items from %s", len(items), path)
    if smoke:
        items = items[:1]
    elif limit:
        items = items[:limit]
    return items


# --------------------------------------------------------------------------- #
# Backend construction. Mock proves the plumbing offline; local/api run real.  #
# --------------------------------------------------------------------------- #
def _mock_backend():
    """
    Deterministic mock that exercises all three call types the scorer makes:
    QA answer generation, answer decomposition (must return a JSON list), and
    the Yes/No unit judge. Kept simple: the judge says Yes when the point shares
    a salient keyword with the reference blob.
    """
    from llm_backend import make_backend

    KEYS = ("porridge", "rice", "bubur", "tea", "milk", "afternoon", "breakfast")

    def responder(messages):
        content = messages[0]["content"].lower()
        if "return only a json array" in content:  # decomposition
            # Pull the answer block and split into 2 crude units.
            ans = content.split("answer:\n", 1)[-1]
            sents = [s.strip() for s in ans.split(".") if len(s.strip()) > 12][:3]
            return json.dumps(sents or ["a cultural fact about food"])
        if "reference cultural knowledge points" in content:  # unit judge
            point = content.split("cultural knowledge points:", 1)[-1]
            point = point.split("reference cultural knowledge points", 1)[0]
            return "Yes - matches." if any(k in point for k in KEYS) else "No - unrelated."
        # QA answer generation (baseline or augmented)
        return ("A traditional breakfast includes bubur ayam, a rice porridge. "
                "Rice-based dishes are common. Tea with milk is served in the afternoon.")

    return make_backend("mock", responder=responder, name="TASK")


def _real_backend(mode: str, model: str):
    from llm_backend import make_backend
    if mode == "api":
        return make_backend("api", model_name=model)
    if mode == "local":
        # On Narval, load the HF model once and wrap it. Mirrors run_local.
        from run_local import load_hf_model
        from llm_backend import LocalBackend
        m, tok = load_hf_model(model, "bfloat16")
        return LocalBackend(model_obj=m, tokenizer_obj=tok, model_name=model)
    raise SystemExit(f"unknown mode {mode}")


def _build_orchestrator_factory(backend, max_loops):
    """Return a build_orchestrator(item) closure for the agentic provider."""
    from agent_a_adapter import AgentA
    from agent_b_engine import AgentBCritiqueEngine
    from orchestrator import CulturalAgentOrchestrator

    def build(item):
        gt = item["ground_truth"]
        agent_a = AgentA(backend, location=gt.get("location", "Unknown"),
                         sub_topic=gt.get("sub_topic", "Unknown"),
                         reconstruct_graph=True, use_sbert=True)
        agent_b = AgentBCritiqueEngine(backend)
        return CulturalAgentOrchestrator(
            agent_a, agent_b, arbiter_backend=backend,
            use_context=True, context_mode="template")
    return build


def build_providers(systems, backend, evaluator, max_loops):
    providers = {}
    orch_factory = _build_orchestrator_factory(backend, max_loops)
    for s in systems:
        if s == "culfit_baseline":
            providers[s] = culfit_baseline_provider(evaluator)
        elif s.startswith("agentic_"):
            rest = s.split("agentic_", 1)[1] or "sequential"
            style = "verbose"
            for suf in ("_terse", "_verbose"):
                if rest.endswith(suf):
                    style = suf[1:]
                    rest = rest[: -len(suf)]
                    break
            topo = rest or "sequential"
            providers[s] = agentic_answer_provider(
                orch_factory, evaluator, topology=topo, style=style)
        else:
            raise SystemExit(f"unknown system '{s}' "
                             "(expected culfit_baseline or agentic_<static|parallel|sequential>)")
    return providers


def run(mode, model, queries, systems, max_loops, smoke, limit):
    items = load_items(queries, smoke, limit)
    backend = _mock_backend() if mode == "mock" else _real_backend(mode, model)
    evaluator = TaskEvaluator(backend)
    providers = build_providers(systems, backend, evaluator, max_loops)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("runs", exist_ok=True)
    jsonl_path = f"runs/taskeval_{ts}.jsonl"
    summary_path = f"runs/taskeval_{ts}_summary.json"

    summaries = {}
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for name, provider in providers.items():
            logger.info("=== SYSTEM: %s (n=%d) ===", name, len(items))
            out = evaluate_system(items, provider, evaluator, name)
            summaries[name] = out["summary"]
            for rec in out["per_item"]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Head-to-head deltas vs the baseline (the number the paper must beat).
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

    summary = {"n_items": len(items), "mode": mode, "model": model,
               "systems": summaries, "deltas_vs_culfit_baseline": deltas}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Wrote %s and %s", jsonl_path, summary_path)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="mock", choices=["mock", "api", "local"])
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--queries", default=None,
                   help="JSONL eval set (e.g. CulFiT/GlobalCultureQA/eval_set_n100.jsonl)")
    p.add_argument("--systems", default="culfit_baseline,agentic_sequential",
                   help="comma-separated: culfit_baseline, agentic_static/parallel/sequential")
    p.add_argument("--max_loops", type=int, default=3)
    p.add_argument("--smoke", action="store_true", help="1 item only")
    p.add_argument("--limit", type=int, default=None, help="cap number of items")
    args = p.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    summary = run(args.mode, args.model, args.queries, systems,
                  args.max_loops, args.smoke, args.limit)
    print("\n=== TASK EVAL SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
