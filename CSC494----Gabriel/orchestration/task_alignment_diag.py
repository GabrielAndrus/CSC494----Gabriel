"""
Task-alignment diagnostic (PluralTree).

WHAT QUESTION THIS ANSWERS
--------------------------
The professor's July 9 reframe: the null augmentation result may not be a bug —
it may mean the two tasks are not *aligned*, i.e. Agent A's cultural-graph
augmentation does not carry signal for Agent B's GlobalCultureQA task. Before any
more topology ablation, we need one artifact that measures exactly that, and that
lets us *read the reasoning* to see where augmentation helped, hurt, or was
ignored.

WHAT THIS RUNS
--------------
For each query, a single-variable paired comparison in the STATIC topology (no
repair loop — we are testing alignment, not repair dynamics):

    arm A  "no-context"  : Agent A generate() -> Agent B on-task scoring, context OFF
    arm B  "augmented"   : Agent A generate_with_context() -> same scoring, context ON

Both arms score the SAME generated paths against the SAME ground-truth
verified_points using the existing on-task metrics:
    - PRIMARY: CulFiT rubric mean_precision (evaluate_payload_batch). The engine
      threads the augmentation context into the Knowledge-Path check, so THIS
      metric responds to the augmentation. The per-query delta is computed on it.
    - SECONDARY: point-level kp_recall / kp_f1 (score_knowledge_points). The
      engine's point-coverage judge takes NO context argument, so kp_f1 is
      context-BLIND and is identical across arms by construction. It is recorded
      as a descriptive coverage number, NOT used for the delta.

Why rubric and not kp_f1: a first pass mistakenly put the delta on kp_f1, which
the augmentation never reaches — the delta was pinned to zero regardless of
alignment. The rubric metric is the lever the augmentation was designed to move,
so measuring the delta there is what actually answers the professor's question.

The ONLY thing that differs between the two arms is whether Agent A's
reconstructed context is shown to Agent B's knowledge-path judge. That isolates
"does the augmentation help the task" to a single lever. Paths are generated ONCE
per query and reused across both arms (a) to halve generation cost and (b) so the
delta cannot be contaminated by path-set differences — it is purely the effect of
the augmentation evidence.

WHAT IT WRITES
--------------
runs/align_<tag>_<model>_<ts>.jsonl        one record per (query, arm) with the
                                           full Agent B trace + kp block + the
                                           context string that was shown (arm B)
runs/align_<tag>_<model>_<ts>_deltas.jsonl one record per query: paired
                                           no-context vs augmented RUBRIC scores,
                                           the per-query delta (helped/hurt/same),
                                           per-dimension rubric verdicts, and the
                                           secondary context-blind kp_f1 numbers
runs/align_<tag>_<model>_<ts>_summary.json aggregate: mean rubric precision per
                                           arm, mean delta, helped/hurt/same tally

The deltas + traces are the reasoning-log material the professor wants: you can
open the jsonl and read, per query, the augmentation string and whether it moved
the score.

MODES
-----
--mode mock   : deterministic, no GPU/API — validates the harness end to end.
--mode local  : in-process HF model on the GPU (cluster). Loads ONE model, shared
                by both agents and the judge, exactly like run_local.py.
--mode api    : OpenAI-compatible endpoint (e.g. gpt-4o-mini once access is
                confirmed), so absolute F1 can later be made CulFiT-comparable.

COST / SMOKE
------------
Static topology, no repair. Per query the model calls are:
    generate (1) [+ generate again is AVOIDED: paths reused]
    Agent B batch rubric: 3 calls/path * min(n_paths, max_paths_binary)
    kp judge: n_paths_scored * n_points   (capped by --max_paths_scored)
Both arms reuse the SAME generation, so generation is paid once per query.
Use --smoke (default 3 queries, 3 paths, 3 scored) for a <30-min GPU check; scale
up with --n / --max_paths for the sbatch run.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("align_diag")


# --------------------------------------------------------------------------- #
# Scoring: run Agent B's on-task metric for one arm over a fixed path set.     #
# --------------------------------------------------------------------------- #
def _f1(precision: float, recall: float) -> float:
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


def score_arm(agent_b, paths, ground_truth, context, kp_mode, max_paths_scored,
              max_paths_binary):
    """
    Score one arm (context None => no-context arm; context str => augmented arm)
    over an ALREADY-GENERATED path set. Returns the on-task numbers + the full
    trace so the reasoning can be read later.

    Uses the existing engine methods unchanged:
      - evaluate_payload_batch : CulFiT binary rubric -> mean_precision, per_path
        verdicts (Group/Topic/Knowledge-Path). context IS threaded into the
        knowledge-path check here, so `rubric_mean_precision` is the metric that
        actually RESPONDS to the augmentation. This is the PRIMARY alignment
        signal (the delta is computed on it).
      - score_knowledge_points : point-level kp_recall / kp_precision against
        verified_points. NOTE: the engine's point-coverage judge does NOT take a
        context argument, so kp_f1 is context-BLIND by construction — it is
        recorded as a SECONDARY descriptive number (how much of the golden set
        the paths cover) but is NOT used for the delta. Threading context into
        point coverage would be a separate engine change (Option B), deliberately
        not taken here.
    """
    path_texts = [p.get("llm_result", "") for p in paths]

    batch = agent_b.evaluate_payload_batch(
        paths, ground_truth, context=context, max_paths_binary=max_paths_binary)

    # evaluate_payload_batch already computes a kp block, but with fixed args.
    # Recompute explicitly so --kp_mode / --max_paths_scored are honoured and the
    # numbers are unambiguous in the record.
    kp = agent_b.score_knowledge_points(
        path_texts, ground_truth, mode=kp_mode, max_paths_scored=max_paths_scored)

    precision = kp["kp_precision"]
    recall = kp["kp_recall"]
    return {
        "kp_precision": precision,
        "kp_recall": recall,
        "kp_f1": _f1(precision, recall),
        "rubric_mean_precision": batch["mean_precision"],
        "rubric_approved": batch["approved"],
        "n_paths_total": kp.get("n_paths_total"),
        "n_paths_scored": kp.get("n_paths_scored"),
        "n_points": kp["n_points"],
        "covered_points": kp["covered_points"],
        "missed_points": kp["missed_points"],
        "per_path": batch["per_path"],       # Group/Topic/KP verdicts + feedback
        "context_used": bool(context),
        "kp_mode": kp["kp_mode"],
    }


# --------------------------------------------------------------------------- #
# Per-query paired run: generate once, score both arms, emit delta.           #
# --------------------------------------------------------------------------- #
def run_query(agent_a, agent_b, q, kp_mode, max_paths_scored, max_paths_binary):
    """
    Generate paths ONCE (with context so we get both paths and the augmentation),
    then score the no-context and augmented arms over that same path set.
    """
    gen = agent_a.generate_with_context(q["query"])
    paths = gen["paths"]
    context = gen.get("contextualization")
    central = gen.get("central_nodes", [])

    gt = q["ground_truth"]

    if not paths:
        logger.warning("[%s] Agent A produced 0 paths; recording empty.", q["query"])
        empty = {"kp_precision": 0.0, "kp_recall": 0.0, "kp_f1": 0.0,
                 "rubric_mean_precision": 0.0, "rubric_approved": False,
                 "n_paths_total": 0, "n_paths_scored": 0, "n_points": len(gt.get("verified_points", [])),
                 "covered_points": [], "missed_points": list(gt.get("verified_points", [])),
                 "per_path": [], "context_used": False, "kp_mode": kp_mode}
        return {
            "no_context": empty,
            "augmented": {**empty, "context_used": True},
            "context_str": context, "central_nodes": central, "n_paths": 0,
        }

    arm_none = score_arm(agent_b, paths, gt, None, kp_mode,
                         max_paths_scored, max_paths_binary)
    arm_ctx = score_arm(agent_b, paths, gt, context, kp_mode,
                        max_paths_scored, max_paths_binary)
    return {
        "no_context": arm_none,
        "augmented": arm_ctx,
        "context_str": context,
        "central_nodes": central,
        "n_paths": len(paths),
    }


def classify_delta(d: float, eps: float = 1e-9) -> str:
    if d > eps:
        return "helped"
    if d < -eps:
        return "hurt"
    return "same"


# --------------------------------------------------------------------------- #
# Backends / agents.                                                           #
# --------------------------------------------------------------------------- #
def build_agents_mock():
    """Deterministic agents that exercise the alignment comparison offline."""
    from orchestration.llm_backend import make_backend
    from orchestration.agent_a_adapter import AgentA
    from orchestration.agent_b_engine import AgentBCritiqueEngine
    from orchestration.run_ablation import mock_agent_a_responder, mock_agent_b_responder

    a_backend = make_backend("mock", responder=mock_agent_a_responder, name="A")
    b_backend = make_backend("mock", responder=mock_agent_b_responder, name="B")
    return a_backend, b_backend, AgentA, AgentBCritiqueEngine


def build_backend_local(model_name, dtype):
    from orchestration.run_local import load_hf_model
    from orchestration.llm_backend import LocalBackend
    model, tok = load_hf_model(model_name, dtype)
    backend = LocalBackend(model_obj=model, tokenizer_obj=tok, model_name=model_name)
    return backend


def build_backend_api(model_name):
    from orchestration.llm_backend import make_backend
    return make_backend("api", model_name=model_name)


# --------------------------------------------------------------------------- #
# Eval set loading (reuses run_local's loader semantics).                      #
# --------------------------------------------------------------------------- #
def load_queries(queries_path, n):
    from orchestration.run_ablation import QUERIES
    if not queries_path:
        items = list(QUERIES)
    else:
        items = []
        with open(queries_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    if n is not None:
        items = items[:n]
    logger.info("Loaded %d queries.", len(items))
    return items


def run(mode, model_name, dtype, queries_path, n, kp_mode,
        max_paths, max_paths_scored, max_paths_binary,
        merge_threshold, use_sbert):
    from orchestration import bootstrap
    bootstrap.install()

    # Build backends / agent classes per mode.
    if mode == "mock":
        a_backend, b_backend, AgentA, AgentBCritiqueEngine = build_agents_mock()
    else:
        from orchestration.agent_a_adapter import AgentA
        from orchestration.agent_b_engine import AgentBCritiqueEngine
        backend = (build_backend_local(model_name, dtype) if mode == "local"
                   else build_backend_api(model_name))
        a_backend = b_backend = backend

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = (model_name or "mock").replace("/", "_")
    tag = "smoke" if (n is not None and n <= 3) else "full"
    os.makedirs("runs", exist_ok=True)
    rec_path = f"runs/align_{tag}_{safe_model}_{ts}.jsonl"
    delta_path = f"runs/align_{tag}_{safe_model}_{ts}_deltas.jsonl"
    summary_path = f"runs/align_{tag}_{safe_model}_{ts}_summary.json"

    queries = load_queries(queries_path, n)

    # PRIMARY signal = CulFiT rubric mean_precision, which context flows into.
    # kp_f1 is retained per-arm as a SECONDARY, context-blind descriptor only.
    rub_none, rub_ctx, deltas = [], [], []
    tally = {"helped": 0, "hurt": 0, "same": 0}

    for q in queries:
        # Fresh Agent A per query (location/sub_topic differ). reconstruct_graph
        # True so generate_with_context yields the augmentation string.
        agent_a = AgentA(a_backend, location=q["location"], sub_topic=q["sub_topic"],
                         max_paths=max_paths, reconstruct_graph=True,
                         merge_threshold=merge_threshold, use_sbert=use_sbert)
        agent_b = AgentBCritiqueEngine(b_backend)

        out = run_query(agent_a, agent_b, q, kp_mode, max_paths_scored, max_paths_binary)

        # Delta is on the rubric metric — the number the augmentation can move.
        n_rub = out["no_context"]["rubric_mean_precision"]
        c_rub = out["augmented"]["rubric_mean_precision"]
        delta = c_rub - n_rub
        cls = classify_delta(delta)
        tally[cls] += 1
        rub_none.append(n_rub)
        rub_ctx.append(c_rub)
        deltas.append(delta)

        # One rich record per (query, arm) for reading the reasoning later.
        for arm_name, arm in (("no-context", out["no_context"]),
                              ("augmented", out["augmented"])):
            with open(rec_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "query": q["query"], "location": q["location"],
                    "arm": arm_name, "n_paths": out["n_paths"],
                    "central_nodes": out["central_nodes"],
                    # context string only meaningful on the augmented arm
                    "context_str": out["context_str"] if arm_name == "augmented" else None,
                    **arm,
                }) + "\n")

        # One paired delta record per query — the alignment signal, readable.
        # Primary fields are rubric_*; kp_f1_* kept as secondary context-blind refs.
        with open(delta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "query": q["query"], "location": q["location"],
                "n_paths": out["n_paths"],
                # PRIMARY: rubric mean_precision, the context-sensitive signal
                "rubric_no_context": n_rub, "rubric_augmented": c_rub,
                "delta_rubric": delta, "class": cls,
                # per-dimension rubric verdicts, so you can SEE which of Group /
                # Topic / Knowledge-Path the context flipped (or didn't)
                "per_path_no_context": out["no_context"]["per_path"],
                "per_path_augmented": out["augmented"]["per_path"],
                # SECONDARY (context-blind): point coverage, for descriptive value
                "kp_f1_no_context": out["no_context"]["kp_f1"],
                "kp_f1_augmented": out["augmented"]["kp_f1"],
                "kp_recall_no_context": out["no_context"]["kp_recall"],
                "kp_recall_augmented": out["augmented"]["kp_recall"],
                "missed_no_context": out["no_context"]["missed_points"],
                "missed_augmented": out["augmented"]["missed_points"],
                "context_str": out["context_str"],
            }) + "\n")

        logger.info("[%s | %s] rubric none=%.3f aug=%.3f  delta=%+.3f (%s)  [kp_f1 none=%.3f aug=%.3f, ctx-blind]",
                    q["location"], q["query"], n_rub, c_rub, delta, cls,
                    out["no_context"]["kp_f1"], out["augmented"]["kp_f1"])

    n_q = len(queries) or 1
    mean_none = sum(rub_none) / n_q
    mean_ctx = sum(rub_ctx) / n_q
    mean_delta = sum(deltas) / n_q
    summary = {
        "n_queries": len(queries),
        "mode": mode, "model": model_name, "kp_mode": kp_mode,
        "topology": "static (no repair) — alignment diagnostic",
        "primary_metric": ("CulFiT rubric mean_precision (Group/Topic/Knowledge-"
                           "Path). Context is threaded into the Knowledge-Path "
                           "check, so this metric responds to augmentation."),
        "mean_rubric_no_context": mean_none,
        "mean_rubric_augmented": mean_ctx,
        "mean_delta_rubric": mean_delta,
        "tally": tally,
        # Honesty caveat: absolute rubric precision is judge-model-dependent and
        # NOT comparable to CulFiT Table 1 (GPT-4o-mini). Only the within-protocol
        # delta (augmented - no_context) is the defensible alignment signal.
        "comparability": ("within-protocol delta only; absolute rubric precision "
                          f"depends on the judge ({model_name or 'mock'}) and is "
                          "NOT comparable to CulFiT's reported numbers"),
        "secondary_note": ("kp_f1 fields in the records/deltas are the point-"
                           "coverage metric; the engine's coverage judge takes no "
                           "context, so kp_f1 is context-BLIND and identical across "
                           "arms by construction — descriptive only, not the delta."),
        "interpretation": (
            "delta>0 across queries => augmentation shifts Agent B's on-task "
            "rubric judgement (tasks carry mutual signal). delta~0 => augmentation "
            "is inert for the rubric (not aligned / wrong augmentation source — "
            "the professor's 'find a better RAG' branch). delta<0 => augmentation "
            "worsens the judgement (e.g. scaffolding narration)."),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Wrote:\n  %s\n  %s\n  %s", rec_path, delta_path, summary_path)
    print("\n=== TASK-ALIGNMENT DIAGNOSTIC ===")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="mock", choices=["mock", "local", "api"])
    p.add_argument("--model", default=None,
                   help="HF dir/id (local) or API model name (api). Ignored for mock.")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--queries", default=None,
                   help="JSONL eval set; default = built-in 2-item fixture.")
    p.add_argument("--n", type=int, default=None,
                   help="cap number of queries (use --n 3 for a <30-min smoke).")
    p.add_argument("--smoke", action="store_true",
                   help="shortcut for a cheap check: n=3, max_paths=3, "
                        "max_paths_scored=3 (override individually if needed).")
    p.add_argument("--kp_mode", default="judge", choices=["judge", "lexical"],
                   help="point-coverage scorer: judge (semantic, model calls) or "
                        "lexical (deterministic, zero calls — fast floor).")
    p.add_argument("--max_paths", type=int, default=None,
                   help="cap Agent A paths per query.")
    p.add_argument("--max_paths_scored", type=int, default=5,
                   help="cap paths scored in judge mode (n_paths*n_points calls).")
    p.add_argument("--max_paths_binary", type=int, default=5,
                   help="cap paths for the CulFiT binary rubric (3 calls/path).")
    p.add_argument("--merge_threshold", type=float, default=0.8)
    p.add_argument("--no_sbert", action="store_true")
    args = p.parse_args()

    # --smoke sets cheap defaults without clobbering explicit overrides.
    n = args.n
    max_paths = args.max_paths
    max_paths_scored = args.max_paths_scored
    if args.smoke:
        if n is None:
            n = 3
        if max_paths is None:
            max_paths = 3
        max_paths_scored = min(max_paths_scored, 3)

    run(args.mode, args.model, args.dtype, args.queries, n, args.kp_mode,
        max_paths, max_paths_scored, args.max_paths_binary,
        args.merge_threshold, use_sbert=not args.no_sbert)
