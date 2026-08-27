"""
analyze_taskeval.py — paired significance analysis for the on-task eval.

WHY
---
run_judge.py reports macro means (e.g. baseline F1 0.340 vs agentic 0.310,
delta -0.030). A macro mean alone cannot tell you whether a delta that small is
distinguishable from noise. Because every system answers the SAME items and is
graded by the SAME judge, the comparison is naturally PAIRED: for each item we
have (baseline_f1, agentic_f1). Pairing removes between-item difficulty variance
(some cultural questions are just harder), which is the dominant noise source
here, so a paired test is both correct and much more sensitive than an unpaired
one.

WHAT IT REPORTS (per system, vs culfit_baseline, on F1/precision/recall)
  - n pairs, mean delta, SD of deltas
  - 95% CI on the mean delta (t-based)
  - paired t-test p-value            (parametric; assumes ~normal deltas)
  - Wilcoxon signed-rank p-value     (non-parametric; robust to skew/outliers)
  - Cohen's d_z (paired effect size) (matches the d reported on the team poster)
  - win/loss/tie counts across items
  - the most negative and most positive items (for error analysis)

Both tests are reported because per-item F1 deltas are often skewed and can have
outliers (the n=1 smoke item was a 14x outlier), so a t-test alone could mislead.
If t and Wilcoxon disagree, trust Wilcoxon and say so.

MULTIPLE COMPARISONS
  Two systems are compared to one baseline, so p-values are also reported with a
  Holm-Bonferroni correction across the comparisons within each metric.

USAGE
    python analyze_taskeval.py --scores runs/taskeval_<ts>.jsonl
    python analyze_taskeval.py --scores runs/taskeval_<ts>.jsonl \
        --baseline culfit_baseline --out runs/significance.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict


# --------------------------------------------------------------------------- #
# Stats helpers. Implemented directly so this runs with no SciPy dependency on
# the cluster; SciPy is used when available for exact p-values.
# --------------------------------------------------------------------------- #
try:
    from scipy import stats as _scipy_stats
    _HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    _HAVE_SCIPY = False


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _sd(xs, ddof=1):
    n = len(xs)
    if n <= ddof:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - ddof))


def _t_cdf(t, df):
    """Student-t CDF via the regularized incomplete beta function."""
    x = df / (df + t * t)
    ib = _betainc(df / 2.0, 0.5, x)
    p = 0.5 * ib
    return p if t <= 0 else 1.0 - p


def _betainc(a, b, x):
    """Regularized incomplete beta I_x(a,b) via continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1 - x) / b


def _betacf(a, b, x, itmax=200, eps=3e-12):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _t_ppf_975(df):
    """Two-sided 95% t critical value; bisection on the CDF."""
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _t_cdf(mid, df) < 0.975:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def paired_t_test(deltas):
    """Two-sided paired t-test on the deltas (H0: mean delta = 0)."""
    n = len(deltas)
    if n < 2:
        return None, None
    sd = _sd(deltas)
    if sd == 0:
        return (0.0, 1.0) if _mean(deltas) == 0 else (float("inf"), 0.0)
    t = _mean(deltas) / (sd / math.sqrt(n))
    if _HAVE_SCIPY:
        t_s, p_s = _scipy_stats.ttest_rel(
            [d for d in deltas], [0.0] * n)
        return float(t_s), float(p_s)
    p = 2 * (1 - _t_cdf(abs(t), n - 1))
    return t, p


def wilcoxon(deltas):
    """Two-sided Wilcoxon signed-rank test (zeros dropped, ties averaged)."""
    nz = [d for d in deltas if d != 0]
    n = len(nz)
    if n < 1:
        return None, None
    if _HAVE_SCIPY:
        try:
            w, p = _scipy_stats.wilcoxon(nz, alternative="two-sided")
            return float(w), float(p)
        except Exception:  # noqa: BLE001
            pass
    # Normal approximation with tie correction.
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j + 2) / 2.0  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for d, r in zip(nz, ranks) if d > 0)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return w_plus, 1.0
    z = (w_plus - mu) / sigma
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return w_plus, p


def cohens_dz(deltas):
    """Paired effect size: mean(delta) / sd(delta)."""
    sd = _sd(deltas)
    return (_mean(deltas) / sd) if sd > 0 else 0.0


def holm_bonferroni(pvals: dict) -> dict:
    """Holm-Bonferroni step-down adjustment. Input/return: {name: p}."""
    items = sorted((p, k) for k, p in pvals.items() if p is not None)
    m = len(items)
    adjusted, prev = {}, 0.0
    for rank, (p, k) in enumerate(items):
        adj = min(1.0, max(prev, (m - rank) * p))
        adjusted[k] = adj
        prev = adj
    return adjusted


# --------------------------------------------------------------------------- #
# Loading / pairing
# --------------------------------------------------------------------------- #
def load_scores(path):
    """Return {system: {idx: record}} from a taskeval_*.jsonl."""
    by_sys = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            sys_name, idx = r.get("system"), r.get("idx")
            if sys_name is None or idx is None:
                continue
            by_sys[sys_name][idx] = r
    return by_sys


def compare(by_sys, baseline, system, metric):
    """Paired deltas (system - baseline) over items present in BOTH."""
    b, s = by_sys.get(baseline, {}), by_sys.get(system, {})
    idxs = sorted(set(b) & set(s))
    deltas, rows = [], []
    for i in idxs:
        bv, sv = b[i].get(metric), s[i].get(metric)
        if bv is None or sv is None:
            continue
        d = sv - bv
        deltas.append(d)
        rows.append((i, bv, sv, d, b[i].get("location")))
    return deltas, rows


def analyze(scores_path, baseline, out_path=None):
    by_sys = load_scores(scores_path)
    systems = [s for s in by_sys if s != baseline]
    if baseline not in by_sys:
        raise SystemExit(f"baseline '{baseline}' not found. "
                         f"Systems present: {sorted(by_sys)}")

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"# Paired significance analysis")
    emit()
    emit(f"- scores file: `{scores_path}`")
    emit(f"- baseline: `{baseline}`")
    emit(f"- systems compared: {', '.join(f'`{s}`' for s in systems)}")
    emit(f"- SciPy available: {_HAVE_SCIPY} "
         f"({'exact p-values' if _HAVE_SCIPY else 'internal approximations'})")
    emit()

    for metric in ("f1", "precision", "recall"):
        emit(f"## {metric.upper()}")
        emit()
        emit("| system | n | mean Δ | 95% CI | Cohen's d_z | t p | Wilcoxon p | "
             "win/loss/tie |")
        emit("|---|---|---|---|---|---|---|---|")
        raw_p = {}
        detail = {}
        for s in systems:
            deltas, rows = compare(by_sys, baseline, s, metric)
            if not deltas:
                emit(f"| {s} | 0 | — | — | — | — | — | — |")
                continue
            n = len(deltas)
            md, sd = _mean(deltas), _sd(deltas)
            se = sd / math.sqrt(n) if n > 1 else 0.0
            tcrit = _t_ppf_975(n - 1) if n > 1 else 0.0
            ci = (md - tcrit * se, md + tcrit * se)
            _, pt = paired_t_test(deltas)
            _, pw = wilcoxon(deltas)
            dz = cohens_dz(deltas)
            wins = sum(1 for d in deltas if d > 0)
            losses = sum(1 for d in deltas if d < 0)
            ties = sum(1 for d in deltas if d == 0)
            raw_p[s] = pw if pw is not None else pt
            detail[s] = (deltas, rows)
            emit(f"| {s} | {n} | {md:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                 f"{dz:+.3f} | {pt:.4g} | {pw:.4g} | {wins}/{losses}/{ties} |")
        # Holm correction across systems within this metric.
        if len(raw_p) > 1:
            adj = holm_bonferroni(raw_p)
            emit()
            emit("Holm-Bonferroni adjusted p (across the "
                 f"{len(raw_p)} comparisons in this metric): "
                 + ", ".join(f"`{k}` = {v:.4g}" for k, v in adj.items()))
        emit()

        # Error-analysis pointers on F1 only (keeps the report readable).
        if metric == "f1":
            for s, (deltas, rows) in detail.items():
                rows_sorted = sorted(rows, key=lambda r: r[3])
                emit(f"<details><summary>{s}: most-hurt / most-helped items"
                     f"</summary>")
                emit()
                emit("| idx | location | baseline | system | Δ |")
                emit("|---|---|---|---|---|")
                for i, bv, sv, d, loc in rows_sorted[:5]:
                    emit(f"| {i} | {loc} | {bv:.3f} | {sv:.3f} | {d:+.3f} |")
                emit("| … | | | | |")
                for i, bv, sv, d, loc in rows_sorted[-5:]:
                    emit(f"| {i} | {loc} | {bv:.3f} | {sv:.3f} | {d:+.3f} |")
                emit()
                emit("</details>")
                emit()

    emit("## How to read this")
    emit()
    emit("- **Paired** because every system answers the same items under the "
         "same judge; pairing removes per-item difficulty variance.")
    emit("- **Wilcoxon** is the safer headline when per-item deltas are skewed "
         "or have outliers; report it if it disagrees with the t-test.")
    emit("- **Cohen's d_z** is the paired effect size (mean Δ / SD of Δ). "
         "~0.2 small, ~0.5 medium, ~0.8 large.")
    emit("- **CI crossing zero** means the delta is not distinguishable from "
         "no effect at 95%.")
    emit("- All deltas are within one judge protocol, so they are comparable to "
         "each other but not to CulFiT's published absolute numbers.")

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n[wrote {out_path}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True,
                   help="runs/taskeval_<ts>.jsonl from run_judge.py")
    p.add_argument("--baseline", default="culfit_baseline")
    p.add_argument("--out", default=None, help="optional markdown output path")
    args = p.parse_args()
    analyze(args.scores, args.baseline, args.out)
