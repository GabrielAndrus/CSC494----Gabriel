#!/usr/bin/env python
"""
score.py -- Score run_single_model.py outputs the way Ki et al. do, and print a
comparison against their Table 1.

Two scorers, reported side by side:
  paper  : `evaluate/accuracy_single.py` logic ported VERBATIM (clean_text ->
           contains_yes -> contains_no -> contains_neither, later checks overwrite
           earlier ones only if they fire on the already-rewritten text). This is the
           number comparable to the paper. NOTE it does substring matching on the whole
           generation ('no' also matches "not", "norm", "know", ...).
  strict : first-word parser -- the label must be the first word of the answer
           (yes / no / neither). Anything else counts as unparsed (= wrong).
The gap between the two is a measure of how much answer-extraction rules move the
headline number, independent of the model.

    python score.py results/*.jsonl [--out results/summary.json]
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from collections import defaultdict

# ---- Ki et al., evaluate/accuracy_single.py (verbatim) ---------------------------------
def clean_text(text):
    return text.replace(',', '').replace('.', '').strip().lower()

def contains_yes(text):
    return 'yes' in text.lower() or 'is socially acceptable' in text.lower() or 'are socially acceptable' in text.lower()

def contains_no(text):
    return 'no' in text.lower() or 'is not socially acceptable' in text.lower() or 'are not socially acceptable' in text.lower()

def contains_neither(text):
    return 'neither' in text.lower()


def paper_predict(generation: str) -> str:
    cleaned_text = clean_text(generation)
    if contains_yes(cleaned_text):
        cleaned_text = "yes"
    if contains_no(cleaned_text):
        cleaned_text = "no"
    if contains_neither(cleaned_text):
        cleaned_text = "neutral"
    return cleaned_text
# -----------------------------------------------------------------------------------------

_STRICT = re.compile(r"^\W*(yes|no|neither)\b", re.IGNORECASE)


def strict_predict(generation: str) -> str:
    m = _STRICT.match(generation or "")
    if not m:
        return "unparsed"
    w = m.group(1).lower()
    return "neutral" if w == "neither" else w


# Ki et al. Table 1 (mean accuracy %, Si w/o RoT, Si w/ RoT, Self-Reflect)
PAPER_TABLE1 = {
    "YES-ONLY": 35.8, "NO-ONLY": 33.2, "NEITHER-ONLY": 31.0,
    "Llama-3 (8B)": (49.5, 63.7, 65.7),
    "Gemma-2 (9B)": (50.7, 68.9, 72.5),
    "Gemma-2-27B": (55.8, 79.2, 80.1),
}
# Which paper row each OpenRouter model is compared to, and how close the match is.
PAPER_ROW = {
    "meta-llama/Meta-Llama-3-8B-Instruct": ("Llama-3 (8B)", "exact model (local HF weights)"),
    "google/gemma-2-9b-it": ("Gemma-2 (9B)", "exact model (local HF weights)"),
    "google/gemma-2-27b-it": ("Gemma-2-27B", "same model family/size (judge row in paper)"),
    "meta-llama/llama-3.1-8b-instruct": ("Llama-3 (8B)", "NEAREST available: paper used Llama-3-8B-Instruct, not 3.1"),
    "qwen/qwen-2.5-7b-instruct": (None, "no Qwen row in the paper (extension)"),
}


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def score_file(rows):
    n = len(rows)
    out = {"n": n}
    for name, fn in (("paper", paper_predict), ("strict", strict_predict)):
        correct = 0
        per_c_ok, per_c_n = defaultdict(int), defaultdict(int)
        confusion = defaultdict(int)
        for r in rows:
            pred = fn(r["generation"])
            gold = r["Gold Label"].lower()
            ok = pred == gold
            correct += ok
            per_c_n[r["Country"]] += 1
            per_c_ok[r["Country"]] += ok
            confusion[f"{gold}->{pred if pred in ('yes','no','neutral','unparsed') else 'other'}"] += 1
        out[name] = {
            "accuracy": correct / n if n else 0.0,
            "correct": correct,
            "per_country": {c: per_c_ok[c] / per_c_n[c] for c in sorted(per_c_n)},
            "confusion": dict(sorted(confusion.items())),
        }
    out["strict"]["unparsed"] = sum(1 for r in rows if strict_predict(r["generation"]) == "unparsed")
    out["truncated"] = sum(1 for r in rows if r.get("finish_reason") == "length")
    out["mean_completion_tokens"] = (
        sum(r.get("completion_tokens") or 0 for r in rows) / n if n else 0.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    by_model = defaultdict(dict)
    all_rows_for_priors = None
    summary = {"runs": {}}
    for path in args.files:
        rows = load(path)
        if not rows:
            continue
        model, typ = rows[0]["model"], rows[0]["type"]
        by_model[model][typ] = score_file(rows)
        summary["runs"][f"{model}::{typ}"] = by_model[model][typ] | {
            "file": path, "providers": sorted({str(r.get("provider")) for r in rows}),
            "served_models": sorted({str(r.get("served_model")) for r in rows})}
        all_rows_for_priors = rows

    # Majority-class priors: deterministic from the data, no model involved.
    if all_rows_for_priors is not None:
        labels = [r["Gold Label"].lower() for r in all_rows_for_priors]
        n = len(labels)
        priors = {k: labels.count(k) / n * 100 for k in ("yes", "no", "neutral")}
        summary["priors_on_scored_items"] = priors

    def fmt(x):
        return f"{x*100:5.1f}" if x is not None else "  -  "

    print("\n" + "=" * 100)
    print("KI ET AL. (ACL 2025) TABLE 1 REPLICATION -- Single Model, NormAd-ETI (accuracy %)")
    print("=" * 100)
    print(f"{'Model (OpenRouter id)':38} {'scorer':7} {'w/o RoT':>8} {'w/ RoT':>8}   paper w/o -> w/    match")
    print("-" * 100)
    for model, d in by_model.items():
        row_name, note = PAPER_ROW.get(model, (None, "no paper reference"))
        ref = PAPER_TABLE1.get(row_name) if row_name else None
        ref_s = f"{ref[0]:.1f} -> {ref[1]:.1f}" if ref else "-"
        for sc in ("paper", "strict"):
            a = d.get("without_rot", {}).get(sc, {}).get("accuracy")
            b = d.get("with_rot", {}).get(sc, {}).get("accuracy")
            n_a = d.get("without_rot", {}).get("n")
            n_b = d.get("with_rot", {}).get("n")
            print(f"{model:38} {sc:7} {fmt(a):>8} {fmt(b):>8}   {ref_s if sc=='paper' else '':16} {note if sc=='paper' else ''}")
        ns = {t: d[t]["n"] for t in d}
        unparsed = {t: d[t]["strict"]["unparsed"] for t in d}
        truncated = {t: d[t]["truncated"] for t in d}
        print(f"{'':38} n={ns} strict-unparsed={unparsed} truncated={truncated}")
    if all_rows_for_priors is not None:
        print("-" * 100)
        p = summary["priors_on_scored_items"]
        print(f"Label priors on scored items  YES-ONLY {p['yes']:.1f} | NO-ONLY {p['no']:.1f} | "
              f"NEITHER-ONLY {p['neutral']:.1f}   (paper: 35.8 | 33.2 | 31.0)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
