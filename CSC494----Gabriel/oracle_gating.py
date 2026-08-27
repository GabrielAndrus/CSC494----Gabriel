"""
oracle_gating.py — does the conditional finding contain a positive result?

The n=100 run shows augmentation is net-negative on average (ΔF1 ≈ −0.030) but
helps items where the baseline is weak and hurts items where it's strong. IF that
pattern is exploitable, a policy that applies augmentation only where it helps
should beat plain baseline. This script quantifies the ceiling of that idea BEFORE
any pipeline is built.

It reports three numbers per (baseline, augmented) system pair:

  1. baseline mean F1              — always answer with the base model
  2. augmented mean F1             — always inject augmentation (current agentic)
  3. ORACLE-gated mean F1          — per item, take max(baseline, augmented) F1.
                                     Cheats (uses the true score) → upper bound on
                                     ANY competence-gating policy.
  4. best THRESHOLD-gated mean F1  — inject augmentation only when baseline_f1 < t,
                                     sweeping t. Uses only baseline competence as
                                     the signal (still peeks at baseline_f1, so it's
                                     a realistic *ceiling*, not yet deployable), and
                                     reports the best threshold + its F1.

How to read the output (this is the "do we have a paper?" test):

  * ORACLE ≫ baseline (e.g. +0.08 F1 or more), AND threshold-gate captures much of
    that gap → STRONG. The paper is "blind augmentation is net-negative; competence-
    gated augmentation recovers a large gain." Build the pipeline + a learned gate.

  * ORACLE ≈ baseline (small gap) → the wins/losses are small in magnitude even if
    real. Gating isn't a compelling contribution on its own. Better to know now.

  * In between → viable if a learned gate can capture most of the gap; the
    threshold-gate number tells you how much a simple signal already recovers.

Also prints the correlation between baseline_f1 and per-item ΔF1: the more negative,
the stronger "augmentation helps the weak, hurts the strong," which is the mechanism
the gate exploits.

USAGE
-----
Point it at a run_judge / run_ablation output. It accepts either:
  - a per-item JSONL where each line has {idx, location, system, f1, ...}
  - a *_summary.json / run dict that carries a "per_item" list

    python oracle_gating.py runs/taskeval_<ts>.jsonl
    python oracle_gating.py runs/taskeval_<ts>.jsonl \
        --baseline culfit_baseline --augmented agentic_sequential
    python oracle_gating.py runs/ablation_<ts>.jsonl --csv gating_report.csv

If --baseline/--augmented aren't given, it auto-detects: the system whose name
contains 'baseline' (or the highest-mean non-agentic system) vs each other system.
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict


# ----------------------------- loading ------------------------------------- #

def load_per_item(path: str) -> list[dict]:
    """Read per-item records from JSONL or a JSON dict with a 'per_item' list."""
    recs: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return recs

    # Try whole-file JSON first (array, or a run/summary dict). If that fails,
    # fall back to JSONL. This is robust to JSONL files whose lines start with
    # "{" (which a first-char sniff would misread as a single object).
    parsed = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        recs = parsed
    elif isinstance(parsed, dict):
        if "per_item" in parsed:
            recs = parsed["per_item"]
        else:
            for v in parsed.values():
                if isinstance(v, dict) and "per_item" in v:
                    recs.extend(v["per_item"])
    else:  # JSONL
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and "per_item" in obj:
                recs.extend(obj["per_item"])
            else:
                recs.append(obj)
    # keep only records that have the fields we need
    return [r for r in recs if "system" in r and "f1" in r and
            ("idx" in r or "query" in r or "location" in r)]


def item_key(r: dict):
    """Stable per-item key so baseline and augmented records line up."""
    if r.get("idx") is not None:
        return ("idx", r["idx"])
    if r.get("query"):
        return ("query", r["query"])
    return ("loc", r.get("location"))


def index_by_system(recs: list[dict]) -> dict[str, dict]:
    """system -> {item_key: f1}."""
    out: dict[str, dict] = defaultdict(dict)
    for r in recs:
        try:
            out[r["system"]][item_key(r)] = float(r["f1"])
        except (TypeError, ValueError):
            continue
    return out


# ----------------------------- analysis ------------------------------------ #

def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def analyse_pair(base: dict, aug: dict, base_name: str, aug_name: str):
    keys = sorted(set(base) & set(aug))
    if not keys:
        return None
    b = [base[k] for k in keys]
    a = [aug[k] for k in keys]
    deltas = [ai - bi for ai, bi in zip(a, b)]

    base_mean = mean(b)
    aug_mean = mean(a)
    oracle_mean = mean(max(bi, ai) for bi, ai in zip(b, a))

    # win/loss/tie
    wins = sum(1 for d in deltas if d > 1e-9)
    losses = sum(1 for d in deltas if d < -1e-9)
    ties = len(deltas) - wins - losses

    # threshold sweep: apply augmentation iff baseline_f1 < t
    thresholds = sorted(set(round(x, 4) for x in b) | {0.0, 1.0001})
    best = (-1.0, None, 0)  # (f1, t, n_augmented)
    for t in thresholds:
        gated = [ai if bi < t else bi for bi, ai in zip(b, a)]
        gm = mean(gated)
        n_aug = sum(1 for bi in b if bi < t)
        if gm > best[0]:
            best = (gm, t, n_aug)
    thr_f1, thr_t, thr_naug = best

    corr = pearson(b, deltas)  # expect NEGATIVE: aug helps weak-baseline items

    return {
        "baseline": base_name, "augmented": aug_name, "n_items": len(keys),
        "base_mean_f1": base_mean, "aug_mean_f1": aug_mean,
        "delta_mean": aug_mean - base_mean,
        "oracle_f1": oracle_mean, "oracle_gain": oracle_mean - base_mean,
        "threshold_f1": thr_f1, "threshold_gain": thr_f1 - base_mean,
        "threshold_t": thr_t, "threshold_n_augmented": thr_naug,
        "wins": wins, "losses": losses, "ties": ties,
        "corr_baseline_vs_delta": corr,
    }


def verdict(res: dict) -> str:
    """Translate the oracle gain into a paper-readiness read."""
    g = res["oracle_gain"]
    tg = res["threshold_gain"]
    if g >= 0.08 and tg >= 0.04:
        return ("STRONG — perfect gating lifts F1 by {:.3f}; a baseline-only "
                "threshold gate already recovers {:.3f}. This is a positive "
                "result to build toward.".format(g, tg))
    if g < 0.03:
        return ("WEAK — even perfect gating only adds {:.3f} F1. The conditional "
                "effect is real but small in magnitude; gating alone is not a "
                "compelling contribution.".format(g))
    return ("MODERATE — oracle gain {:.3f}, threshold-gate gain {:.3f}. Viable if "
            "a *learned* gate captures most of the oracle gap; marginal if it "
            "can't.".format(g, tg))


# ------------------------------- main -------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="per-item JSONL or run/summary JSON")
    ap.add_argument("--baseline", default=None,
                    help="baseline system name (auto-detected if omitted)")
    ap.add_argument("--augmented", default=None,
                    help="augmented system name; default = every non-baseline system")
    ap.add_argument("--csv", default=None, help="also write per-item deltas here")
    args = ap.parse_args()

    recs = load_per_item(args.path)
    if not recs:
        print("[error] no usable per-item records (need system + f1 + idx/query/location).")
        return 1
    by_sys = index_by_system(recs)
    systems = list(by_sys)
    if len(systems) < 2:
        print(f"[error] need >=2 systems in the file; found: {systems}")
        return 1

    # pick baseline
    base_name = args.baseline
    if base_name is None:
        cands = [s for s in systems if "baseline" in s.lower()]
        if cands:
            base_name = cands[0]
        else:  # fall back to the non-agentic system with the highest mean f1
            non_ag = [s for s in systems if "agentic" not in s.lower()] or systems
            base_name = max(non_ag, key=lambda s: mean(by_sys[s].values()))
    if base_name not in by_sys:
        print(f"[error] baseline '{base_name}' not in {systems}")
        return 1

    aug_names = ([args.augmented] if args.augmented
                 else [s for s in systems if s != base_name])

    print(f"\nBaseline system: {base_name}   ({len(by_sys[base_name])} items)")
    print("=" * 72)

    results = []
    for aug_name in aug_names:
        if aug_name not in by_sys:
            print(f"[skip] augmented '{aug_name}' not found")
            continue
        res = analyse_pair(by_sys[base_name], by_sys[aug_name], base_name, aug_name)
        if res is None:
            print(f"[skip] no shared items between {base_name} and {aug_name}")
            continue
        results.append(res)

        print(f"\n▶ {aug_name}   (n={res['n_items']} shared items)")
        print(f"    baseline mean F1        : {res['base_mean_f1']:.3f}")
        print(f"    always-augment mean F1  : {res['aug_mean_f1']:.3f}   "
              f"(Δ {res['delta_mean']:+.3f})")
        print(f"    ORACLE-gated  mean F1   : {res['oracle_f1']:.3f}   "
              f"(gain {res['oracle_gain']:+.3f})   ← upper bound")
        print(f"    threshold-gate mean F1  : {res['threshold_f1']:.3f}   "
              f"(gain {res['threshold_gain']:+.3f})   "
              f"[apply aug if baseline_f1 < {res['threshold_t']:.3f}; "
              f"{res['threshold_n_augmented']}/{res['n_items']} items augmented]")
        print(f"    win / loss / tie        : "
              f"{res['wins']} / {res['losses']} / {res['ties']}")
        print(f"    corr(baseline_f1, Δf1)  : {res['corr_baseline_vs_delta']:+.3f}"
              "   (negative ⇒ helps weak items, hurts strong — the gate's signal)")
        print(f"    VERDICT: {verdict(res)}")

    if args.csv and results:
        # dump per-item deltas for the best/first pair for plotting later
        aug_name = results[0]["augmented"]
        base, aug = by_sys[base_name], by_sys[aug_name]
        keys = sorted(set(base) & set(aug))
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write("item_key,baseline_f1,augmented_f1,delta,max_f1\n")
            for k in keys:
                bi, ai = base[k], aug[k]
                f.write(f"{k[1]},{bi:.4f},{ai:.4f},{ai-bi:+.4f},{max(bi,ai):.4f}\n")
        print(f"\n[wrote] per-item deltas → {args.csv}")

    print("\n" + "=" * 72)
    print("Interpretation: the ORACLE gain is the ceiling of any competence-gating")
    print("policy. If it's large and the threshold-gate captures much of it, the")
    print("pivot has a positive result in it — build the pipeline + a learned gate.")
    print("If the oracle gain is small, the conditional effect isn't worth gating.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
