"""
run_generate.py — PASS 1 of the two-pass on-task evaluation.

Loads ONE (small) generator model, runs each system's answer provider over the
eval set, and writes the generated answers to disk. NO judging happens here, so
this pass is cheap, fits a single GPU, and never needs to be re-run when you
change or swap the judge.

Why two passes (see run_judge.py for pass 2):
  - Judging is the expensive part (~n_items * units_per_item calls per system).
    Decoupling means a judge OOM/timeout never destroys the generations.
  - You can re-score the SAME answers with different judge models (e.g. an 8B
    judge, then a 70B judge) and show the deltas are stable — a stronger
    reliability claim than a single judge alone.
  - Peak VRAM stays low; only the judge pass needs the big allocation.

Systems (each produces an answer per item; all scored identically in pass 2):
  - culfit_baseline : base model answers the QA one-shot (CulFiT-at-inference).
  - agentic_<topo>  : orchestrator produces paths+critique as augmentation, then
                      the model writes the task answer on top.

Usage (inside the SLURM job):
    python -m orchestration.run_generate \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --queries CulFiT/GlobalCultureQA/eval_set_n100.jsonl \
        --systems culfit_baseline,agentic_sequential

Output:
    runs/answers_<ts>.jsonl   (one record per (system, item), judge-agnostic)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime

import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from orchestration import bootstrap
bootstrap.install()

from orchestration.llm_backend import LocalBackend
from orchestration.run_local import load_hf_model
from orchestration.task_eval import (
    TaskEvaluator, culfit_baseline_provider, agentic_answer_provider,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("run_generate")


_SMOKE_ITEMS = [
    {"query": "What is a traditional Indonesian breakfast and what does it involve?",
     "location": "Indonesia", "sub_topic": "breakfast",
     "ground_truth": {"location": "Indonesia", "sub_topic": "breakfast",
         "verified_points": [
             "Bubur Ayam is a traditional rice porridge eaten in the morning.",
             "Nasi goreng is sometimes eaten at breakfast in Indonesia.",
             "Breakfast in Indonesia often includes rice-based dishes."]}},
]


def load_items(path, smoke, limit):
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
        return items[:1]
    if limit:
        return items[:limit]
    return items


def build_orchestrator_factory(backend, max_loops):
    from orchestration.agent_a_adapter import AgentA
    from orchestration.agent_b_engine import AgentBCritiqueEngine
    from orchestration.orchestrator import CulturalAgentOrchestrator

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
    orch_factory = build_orchestrator_factory(backend, max_loops)
    for s in systems:
        if s == "culfit_baseline":
            providers[s] = culfit_baseline_provider(evaluator)
        elif s.startswith("agentic_"):
            rest = s.split("agentic_", 1)[1] or "sequential"
            # optional "_terse" / "_verbose" suffix selects the answer style;
            # default is verbose (the original prompt).
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
            raise SystemExit(f"unknown system '{s}'")
    return providers


def _load_done(resume_path):
    """Return set of (system, idx) already completed in a prior partial run."""
    done = set()
    if resume_path and os.path.exists(resume_path):
        with open(resume_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done.add((r.get("system"), r.get("idx")))
                except json.JSONDecodeError:
                    continue  # skip a half-written final line from a kill
        logger.info("Resume: %d (system,item) pairs already done in %s",
                    len(done), resume_path)
    return done


def run(model_name, dtype, queries, systems, max_loops, smoke, limit, resume=None):
    items = load_items(queries, smoke, limit)
    model, tokenizer = load_hf_model(model_name, dtype)
    backend = LocalBackend(model_obj=model, tokenizer_obj=tokenizer, model_name=model_name)
    # The evaluator here is used ONLY for its answer-generation helpers; no
    # judging is invoked in this pass.
    evaluator = TaskEvaluator(backend)
    providers = build_providers(systems, backend, evaluator, max_loops)

    # Resume appends to the SAME file so one file holds the whole run; a fresh
    # run opens a new timestamped file.
    done = _load_done(resume)
    if resume:
        out_path = resume
        mode = "a"
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"runs/answers_{ts}.jsonl"
        mode = "w"
    os.makedirs("runs", exist_ok=True)

    n_written = 0
    with open(out_path, mode, encoding="utf-8") as f:
        for name, provider in providers.items():
            logger.info("=== GENERATE: %s (n=%d) ===", name, len(items))
            for i, item in enumerate(items):
                if (name, i) in done:
                    continue  # already generated in a prior (timed-out) run
                gt = item.get("ground_truth", item)
                try:
                    answer = provider(item)
                except Exception as e:  # noqa: BLE001
                    logger.exception("provider failed on item %d (%s): %s", i, name, e)
                    answer = ""
                rec = {
                    "idx": i,
                    "system": name,
                    "generator_model": model_name,
                    "query": item.get("query"),
                    "location": gt.get("location", item.get("location", "Unknown")),
                    "sub_topic": gt.get("sub_topic", item.get("sub_topic", "Unknown")),
                    "verified_points": gt.get("verified_points", []),
                    "answer": answer,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()          # survive a SLURM kill: don't lose buffered lines
                os.fsync(f.fileno())
                n_written += 1
                logger.info("[%s] %d/%d generated (%d chars)",
                            name, i + 1, len(items), len(answer or ""))

    logger.info("Wrote %d NEW answer records to %s", n_written, out_path)
    print(f"\nGeneration complete: {out_path}")
    print(f"Next: python -m orchestration.run_judge --answers {out_path} "
          f"--judge_model <big-model>")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                   help="generator model (small; runs the answer step)")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--queries", default=None)
    p.add_argument("--systems", default="culfit_baseline,agentic_sequential")
    p.add_argument("--max_loops", type=int, default=3)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", default=None,
                   help="append to and skip (system,idx) already in this "
                        "answers file (for resuming a timed-out run)")
    args = p.parse_args()
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    run(args.model, args.dtype, args.queries, systems,
        args.max_loops, args.smoke, args.limit, resume=args.resume)
