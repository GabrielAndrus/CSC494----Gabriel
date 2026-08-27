#!/usr/bin/env python
"""
build_eval_set.py — Construct the n=100 PluralTree validation set from
CulFiT's GlobalCultureQA.csv.

Confirmed source schema (real header)
-------------------------------------
    cultural_group, topic, source, cultural_knowledge, question,
    grounded_answer, grounded_answer_knowledge_points

  - grounded_answer is a JSON string: {"answer": "...", "cultural_group": ...,
    "language": ..., "topic": ...}
  - grounded_answer_knowledge_points is a JSON string:
    {"knowledge_points": ["...atomic fact...", "...atomic fact...", ...]}

Why this mapping is the right one
---------------------------------
GlobalCultureQA already decomposes each grounded answer into atomic, individually
checkable knowledge units (`knowledge_points`). That is EXACTLY what Agent B's
knowledge-path check scores a candidate path against — so we use the dataset's own
hand-authored decomposition as `verified_points` rather than re-splitting the prose
answer ourselves. The verified points stay fully traceable to the benchmark.

  query                        <- question
  location                     <- cultural_group
  sub_topic                    <- topic
  ground_truth.location        <- cultural_group
  ground_truth.sub_topic       <- topic
  ground_truth.verified_points <- grounded_answer_knowledge_points.knowledge_points

Dependencies: Python stdlib only (csv, json). Nothing to pip-install on Narval.

Usage
-----
    python build_eval_set.py --csv GlobalCultureQA.csv --n 100 --seed 13 \
        --out eval_set_n100.jsonl

    python -m orchestration.run_local \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --queries eval_set_n100.jsonl --max_loops 3

Determinism: seeded, stratified across cultural_group so the 100 aren't all one
region. Writes UTF-8 explicitly (GlobalCultureQA is multilingual / non-ASCII).
"""

from __future__ import annotations
import argparse
import csv
import json
import logging
import sys
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("build_eval_set")

# Some grounded answers are very long; CSV fields can exceed the default limit.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def _parse_kps(raw: str) -> list[str]:
    """grounded_answer_knowledge_points -> list[str] of atomic knowledge units."""
    if not raw or not raw.strip():
        return []
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return []
    kps = obj.get("knowledge_points", []) if isinstance(obj, dict) else []
    return [str(p).strip() for p in kps if str(p).strip()]


def _parse_answer(raw: str) -> str:
    """grounded_answer JSON -> the 'answer' prose (fallback verified point)."""
    if not raw or not raw.strip():
        return ""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    return str(obj.get("answer", "")).strip() if isinstance(obj, dict) else ""


def to_eval_record(row: dict, max_points: int | None) -> dict | None:
    location = str(row.get("cultural_group", "")).strip()
    sub_topic = str(row.get("topic", "")).strip() or "general"
    question = str(row.get("question", "")).strip()
    if not (location and question):
        return None

    verified_points = _parse_kps(row.get("grounded_answer_knowledge_points", ""))
    if not verified_points:
        # Fallback: use the grounded prose answer as a single point so a row with
        # a malformed KP field is still usable rather than silently dropped.
        ans = _parse_answer(row.get("grounded_answer", ""))
        if ans:
            verified_points = [ans]
    if not verified_points:
        return None

    if max_points is not None:
        verified_points = verified_points[:max_points]

    return {
        "query": question,
        "location": location,
        "sub_topic": sub_topic,
        "ground_truth": {
            "location": location,
            "sub_topic": sub_topic,
            "verified_points": verified_points,
        },
    }


def stratified_sample(records, n, seed):
    import random
    rng = random.Random(seed)
    by_loc = defaultdict(list)
    for r in records:
        by_loc[r["location"]].append(r)
    for v in by_loc.values():
        rng.shuffle(v)
    out, locs, i = [], sorted(by_loc), 0
    while len(out) < n and any(by_loc.values()):
        loc = locs[i % len(locs)]
        if by_loc[loc]:
            out.append(by_loc[loc].pop())
        i += 1
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to GlobalCultureQA.csv")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-points", type=int, default=None,
                    help="Optional cap on verified_points per instance "
                         "(default: keep all; some rows have 15+).")
    ap.add_argument("--out", "-o", default="eval_set_n100.jsonl")
    args = ap.parse_args()

    with open(args.csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        log.info("CSV columns: %s", reader.fieldnames)
        records = [rec for row in reader
                   if (rec := to_eval_record(row, args.max_points))]

    log.info("Mapped %d usable records from CSV.", len(records))
    if len(records) < args.n:
        log.warning("Only %d usable records (< requested %d). Writing all.",
                    len(records), args.n)

    sample = stratified_sample(records, args.n, args.seed)

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in sample:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    cultures = sorted({r["location"] for r in sample})
    pts = [len(r["ground_truth"]["verified_points"]) for r in sample]
    log.info("Wrote %d lines to %s", len(sample), args.out)
    log.info("Cultural groups (%d): %s", len(cultures), ", ".join(cultures))
    log.info("verified_points per instance: min=%d max=%d mean=%.1f",
             min(pts), max(pts), sum(pts) / len(pts))
    log.info("Sample line:")
    print(json.dumps(sample[0], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
