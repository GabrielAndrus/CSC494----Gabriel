"""
Ablation harness: runs Static vs. Parallel vs. Sequential topologies over a
shared query set and logs rich per-run signal so metrics can be chosen later.

Run (mock, works offline NOW):
    cd pluraltree && python -m orchestration.run_ablation --mode mock

Run (cluster, later — no other code changes):
    python -m orchestration.run_ablation --mode api --model gpt-4o
    python -m orchestration.run_ablation --mode local --model meta-llama/Llama-3.1-8B-Instruct

Outputs:
    runs/ablation_<timestamp>.jsonl   (one record per (query, topology))
    runs/ablation_<timestamp>_summary.json
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import re
from datetime import datetime

from orchestration.llm_backend import make_backend
from orchestration.agent_a_adapter import AgentA
from orchestration.agent_b_engine import AgentBCritiqueEngine
from orchestration.orchestrator import CulturalAgentOrchestrator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ablation")


# --- Shared evaluation set ------------------------------------------------
QUERIES = [
    {"query": "breakfast", "location": "Indonesia", "sub_topic": "breakfast",
     "ground_truth": {"location": "Indonesia", "sub_topic": "breakfast",
         "verified_points": [
             "Bubur Ayam is a traditional rice porridge eaten in the morning.",
             "Breakfast in Indonesia often includes rice-based dishes."]}},
    {"query": "tea customs", "location": "England", "sub_topic": "beverages",
     "ground_truth": {"location": "England", "sub_topic": "beverages",
         "verified_points": [
             "Tea is commonly served with milk in England.",
             "Afternoon tea is a recognised social custom."]}},
]


# --- Mock responders that actually exercise each topology -----------------
def mock_agent_a_responder(messages: list[dict]) -> str:
    """Agent A: first draft is thin; repair adds historical/cultural context."""
    text = " ".join(m["content"] for m in messages)
    is_repair = "critiqued" in text.lower() or "repair" in text.lower()
    if is_repair:
        return """Here is the repaired JSON:
        [
          {"action": "eating Bubur Ayam", "knowledge": "engaging with local culinary history and tradition",
           "relation_type": "xEffect", "result": "If eating Bubur Ayam, a traditional rice porridge, then you engage with local culinary history."}
        ]
        Hope this helps!"""
    return """[
      {"action": "eating breakfast", "knowledge": "you feel full",
       "relation_type": "xEffect", "result": "If eating Bubur Ayam, then you feel full."}
    ]"""


def mock_agent_b_responder(messages: list[dict]) -> str:
    """
    Agent B: approves group/topic; for knowledge points, requires that the
    path mentions history/tradition (so a thin first draft fails, the repaired
    draft passes -> the sequential loop visibly converges).
    """
    content = messages[0]["content"]
    is_kp_check = "reference cultural knowledge" in content
    if is_kp_check:
        # Inspect ONLY the injected candidate path (the text after the final
        # "cultural knowledge points:" marker), not the static CulFiT examples
        # which themselves contain words like "history"/"tradition".
        marker = "cultural knowledge points:"
        candidate = content.rsplit(marker, 1)[-1]
        candidate = candidate.split("reference cultural knowledge", 1)[0].lower()
        if "history" in candidate or "tradition" in candidate or "culinary" in candidate:
            return "Yes - aligns with traditional culinary knowledge."
        return "No - omits historical/traditional context."
    return "Yes"


def build_orchestrator(mode: str, model: str, q: dict):
    if mode == "mock":
        a_backend = make_backend("mock", responder=mock_agent_a_responder, name="A")
        b_backend = make_backend("mock", responder=mock_agent_b_responder, name="B")
        arbiter = make_backend("mock", responder=mock_agent_a_responder, name="ARB")
    elif mode == "api":
        a_backend = make_backend("api", model_name=model)
        b_backend = make_backend("api", model_name=model)
        arbiter = make_backend("api", model_name=model)
    elif mode == "local":
        raise SystemExit("local mode needs model_obj/tokenizer_obj wired in on the cluster; "
                         "see LocalBackend. Use --mode mock offline.")
    else:
        raise SystemExit(f"unknown mode {mode}")

    agent_a = AgentA(a_backend, location=q["location"], sub_topic=q["sub_topic"])
    agent_b = AgentBCritiqueEngine(b_backend)
    return CulturalAgentOrchestrator(agent_a, agent_b, arbiter_backend=arbiter), a_backend, b_backend


def run(mode: str, model: str, max_loops: int):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = f"runs/ablation_{ts}.jsonl"
    summary_path = f"runs/ablation_{ts}_summary.json"
    records = []

    for q in QUERIES:
        for topology in ("static", "parallel", "sequential"):
            orch, a_backend, b_backend = build_orchestrator(mode, model, q)
            if topology == "static":
                result = orch.static_integration(q["query"], q["ground_truth"])
            elif topology == "parallel":
                result = orch.parallel_debate(q["query"], q["ground_truth"])
            else:
                result = orch.sequential_debate(q["query"], q["ground_truth"], max_loops=max_loops)

            rec = {
                "query": q["query"], "location": q["location"], "topology": topology,
                "n_final_paths": len(result.get("final_paths", [])),
                "trace": result["trace"],
                "llm_calls": {"agent_a": getattr(a_backend, "call_count", None),
                              "agent_b": getattr(b_backend, "call_count", None)},
            }
            records.append(rec)
            os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            logger.info("[%s | %s] approved=%s mean_precision=%.2f loops=%d repairs=%d",
                        q["query"], topology, rec["trace"]["final_approved"],
                        rec["trace"]["final_mean_precision"],
                        rec["trace"]["loops"], rec["trace"]["repairs"])

    # Aggregate summary per topology (raw signal; pick headline metrics later)
    summary = {}
    for topo in ("static", "parallel", "sequential"):
        rs = [r for r in records if r["topology"] == topo]
        n = len(rs)
        summary[topo] = {
            "n_runs": n,
            "approval_rate": sum(r["trace"]["final_approved"] for r in rs) / n,
            "avg_mean_precision": sum(r["trace"]["final_mean_precision"] for r in rs) / n,
            "avg_loops": sum(r["trace"]["loops"] for r in rs) / n,
            "avg_repairs": sum(r["trace"]["repairs"] for r in rs) / n,
            "avg_agent_b_calls": sum((r["llm_calls"]["agent_b"] or 0) for r in rs) / n,
        }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Wrote %s and %s", jsonl_path, summary_path)
    return jsonl_path, summary_path, summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="mock", choices=["mock", "api", "local"])
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--max_loops", type=int, default=3)
    args = p.parse_args()
    _, _, summary = run(args.mode, args.model, args.max_loops)
    print("\n=== ABLATION SUMMARY ===")
    print(json.dumps(summary, indent=2))
