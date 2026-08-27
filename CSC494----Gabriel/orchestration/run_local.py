"""
Cluster entrypoint for the PluralTree ablation in LOCAL mode.

Loads ONE HuggingFace model + tokenizer onto the GPU, wraps it in LocalBackend,
and runs the three-topology ablation. This is what the SLURM script calls.

Why a separate entrypoint (not run_ablation --mode local):
  run_ablation builds a fresh backend per (query, topology) and intentionally
  refuses to instantiate a heavy local model repeatedly. Here we load the model
  ONCE and reuse the same backend across all runs, which is the only sane thing
  to do on a GPU node.

Usage (inside the SLURM job, after `module load` + venv/conda activate):
    python -m orchestration.run_local \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --max_loops 3 \
        [--dtype bfloat16] [--max_new_tokens 2048]

Outputs to CSC494/runs/ just like the mock harness.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
from datetime import datetime

from orchestration import bootstrap
bootstrap.install()

from orchestration.llm_backend import LocalBackend
from orchestration.agent_a_adapter import AgentA
from orchestration.agent_b_engine import AgentBCritiqueEngine
from orchestration.orchestrator import CulturalAgentOrchestrator
from orchestration.run_ablation import QUERIES  # reuse the shared eval set

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("run_local")


def load_hf_model(model_name: str, dtype: str = "bfloat16"):
    """Load tokenizer + causal LM onto available GPU(s). Mirrors Tonga main.py."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    torch_dtype = getattr(torch, dtype, torch.bfloat16)
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Loading model: %s (dtype=%s, device_map=auto)", model_name, dtype)
    # Newer transformers renamed `torch_dtype` -> `dtype`. 
    # Try the new name first, 
    # fall back to the old one so this works on both.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", dtype=torch_dtype
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch_dtype
        )
    model.eval()
    logger.info("Model loaded. Device map ready.")
    return model, tokenizer


def load_queries(queries_path: str | None, smoke: bool):
    """
    Eval set loader. Default: the 2-item smoke fixture from run_ablation.QUERIES.
    With --queries PATH, load a JSONL where each line is a query dict with keys
    {query, location, sub_topic, ground_truth:{location,sub_topic,verified_points}}.
    This is how the n=100 validation set and the full benchmark set are supplied
    without editing code.
    """
    if not queries_path:
        return QUERIES[:1] if smoke else QUERIES
    items = []
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    logger.info("Loaded %d queries from %s", len(items), queries_path)
    return items[:1] if smoke else items


def run(model_name: str, max_loops: int, dtype: str, smoke: bool = False,
        only_topology: str | None = None, max_paths: int | None = None,
        arms: tuple[str, ...] = ("no-context", "template"),
        queries_path: str | None = None,
        merge_threshold: float = 0.8, use_sbert: bool = True):
    model, tokenizer = load_hf_model(model_name, dtype)

    # ONE backend instance, shared by both agents and the arbiter.
    backend = LocalBackend(model_obj=model, tokenizer_obj=tokenizer, model_name=model_name)

    # Smoke mode: 1 query, 1 topology, 1 loop — cheapest possible real run.
    queries = load_queries(queries_path, smoke)
    topologies = (only_topology,) if only_topology else ("static", "parallel", "sequential")
    if smoke and not only_topology:
        topologies = ("sequential",)  # the most complex path, to shake out the loop
    if smoke:
        max_loops = 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace("/", "_")
    tag = "smoke" if smoke else "local"

    os.makedirs("runs", exist_ok=True)
    jsonl_path = f"runs/ablation_{tag}_{safe_model}_{ts}.jsonl"
    summary_path = f"runs/ablation_{tag}_{safe_model}_{ts}_summary.json"
    records = []

    for arm in arms:
        use_context = (arm != "no-context")
        context_mode = "template" if arm == "template" else ("llm" if arm == "llm-rewrite" else "template")
        logger.info("=== ARM: %s (use_context=%s) ===", arm, use_context)

        for q in queries:
            # Path cap keeps Agent B from running 3 critique calls x dozens of paths.
            # Smoke defaults to 3; otherwise use --max_paths (None = no cap).
            a_max_paths = max_paths if max_paths is not None else (3 if smoke else None)
            agent_a = AgentA(backend, location=q["location"], sub_topic=q["sub_topic"],
                             max_paths=a_max_paths,
                             reconstruct_graph=use_context,
                             merge_threshold=merge_threshold, use_sbert=use_sbert)

            agent_b = AgentBCritiqueEngine(backend)
            orch = CulturalAgentOrchestrator(agent_a, agent_b, arbiter_backend=backend,
                                             use_context=use_context, context_mode=context_mode)

            for topology in topologies:
                if topology == "static":
                    result = orch.static_integration(q["query"], q["ground_truth"])
                elif topology == "parallel":
                    result = orch.parallel_debate(q["query"], q["ground_truth"])
                else:
                    result = orch.sequential_debate(q["query"], q["ground_truth"], max_loops=max_loops)

                rec = {
                    "model": model_name, "arm": arm, "query": q["query"],
                    "location": q["location"], "topology": topology,
                    "n_final_paths": len(result.get("final_paths", [])),
                    "trace": result["trace"],
                }
                records.append(rec)
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                logger.info("[%s | %s | %s] approved=%s mean_precision=%.2f loops=%d repairs=%d",
                            arm, q["query"], topology, rec["trace"]["final_approved"],
                            rec["trace"]["final_mean_precision"],
                            rec["trace"]["loops"], rec["trace"]["repairs"])

    # Aggregate per (arm, topology) so the ablation comparison is read directly.
    summary = {}
    for arm in arms:
        for topo in ("static", "parallel", "sequential"):
            rs = [r for r in records if r["arm"] == arm and r["topology"] == topo]
            n = len(rs) or 1
            summary[f"{arm}::{topo}"] = {
                "arm": arm, "topology": topo, "n_runs": len(rs),
                "approval_rate": sum(r["trace"]["final_approved"] for r in rs) / n,
                "avg_mean_precision": sum(r["trace"]["final_mean_precision"] for r in rs) / n,
                "avg_loops": sum(r["trace"]["loops"] for r in rs) / n,
                "avg_repairs": sum(r["trace"]["repairs"] for r in rs) / n,
            }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s and %s", jsonl_path, summary_path)
    print("\n=== ABLATION SUMMARY (local, by arm x topology) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--max_loops", type=int, default=3)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--smoke", action="store_true",
                   help="1 query / 1 topology / 1 loop — cheap first-run check")
    p.add_argument("--topology", choices=["static", "parallel", "sequential"],
                   default=None, help="run only this topology")
    p.add_argument("--max_paths", type=int, default=None,
                   help="cap Agent A paths per query (keeps Agent B cost bounded). "
                        "Default: no cap for full runs, 3 for --smoke.")
    p.add_argument("--arms", nargs="+", default=["no-context", "template"],
                   choices=["no-context", "template", "llm-rewrite"],
                   help="ablation arms to run (default: no-context template)")
    p.add_argument("--queries", default=None,
                   help="JSONL eval set (one query dict per line). Default: built-in "
                        "2-item fixture. Use this to supply the n=100 / full set.")
    p.add_argument("--no_sbert", action="store_true",
                   help="force Jaccard merge instead of SBERT (for CPU/offline checks)")
    p.add_argument("--merge_threshold", type=float, default=0.8,
                   help="node-merge similarity threshold (default 0.8, matches CCKG)")
    args = p.parse_args()
    run(args.model, args.max_loops, args.dtype, smoke=args.smoke,
        only_topology=args.topology, max_paths=args.max_paths,
        arms=tuple(args.arms), queries_path=args.queries,
        merge_threshold=args.merge_threshold, use_sbert=not args.no_sbert)
