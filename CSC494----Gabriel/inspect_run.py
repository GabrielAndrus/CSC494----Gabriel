#!/usr/bin/env python3
"""
inspect_run.py - human-readable view of an ablation run.

Reads the JSONL + summary files written by `run_ablation.py` into `runs/` and
prints a per-run breakdown plus the aggregated summary table. Defensive about
field presence: it shows whatever keys exist rather than assuming a fixed
schema, so it keeps working if the trace/summary shape changes.

Usage:
    python inspect_run.py                       # most recent run in runs/
    python inspect_run.py 20260616_135829       # by timestamp
    python inspect_run.py runs/ablation_20260616_135829.jsonl   # by path
    python inspect_run.py --summary-only        # skip per-run detail
    python inspect_run.py --runs-dir some/dir   # look elsewhere than ./runs
    python inspect_run.py --out report.txt      # write to a file instead of terminal

Exit codes: 0 ok, 1 nothing found / bad input.
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import sys

# Windows consoles default to a legacy codepage (e.g. cp1252) that can't encode
# the box-drawing / check characters this script prints (- # [x] ->), causing a
# UnicodeEncodeError. Force the streams to UTF-8 so terminal output works
# everywhere. (Python 3.7+; no-op if already UTF-8.)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass  # not a reconfigurable stream (e.g. redirected/piped) - leave as-is


# ---------- locating files -------------------------------------------------

def resolve_jsonl(arg: str | None, runs_dir: str) -> str | None:
    """Turn a CLI arg (None | timestamp | path) into a JSONL path."""
    if arg:
        if os.path.isfile(arg):
            return arg
        # treat as timestamp
        cand = os.path.join(runs_dir, f"ablation_{arg}.jsonl")
        if os.path.isfile(cand):
            return cand
        # last resort: glob for the timestamp anywhere in the name
        hits = glob.glob(os.path.join(runs_dir, f"*{arg}*.jsonl"))
        return hits[0] if hits else None
    # no arg: newest *.jsonl in runs_dir
    hits = sorted(glob.glob(os.path.join(runs_dir, "ablation_*.jsonl")))
    return hits[-1] if hits else None


def summary_path_for(jsonl_path: str) -> str | None:
    cand = jsonl_path.replace(".jsonl", "_summary.json")
    return cand if os.path.isfile(cand) else None


# ---------- loading --------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ! skipping malformed line {ln}: {e}", file=sys.stderr)
    return records


# ---------- formatting helpers --------------------------------------------

def fmt(v, nd=2):
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def bar(frac: float, width: int = 12) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = round(frac * width)
    return "#" * filled + "." * (width - filled)


def yn(b) -> str:
    return "[x]" if b else "[ ]"


# ---------- printing -------------------------------------------------------

def print_record(rec: dict, out=sys.stdout) -> None:
    q = rec.get("query", "?")
    loc = rec.get("location", "?")
    topo = rec.get("topology", "?")
    trace = rec.get("trace", {}) or {}

    approved = trace.get("final_approved")
    prec = trace.get("final_mean_precision", 0.0) or 0.0
    loops = trace.get("loops", 0)
    repairs = trace.get("repairs", 0)
    npaths = rec.get("n_final_paths", "?")
    calls = rec.get("llm_calls", {}) or {}

    print(f"\n  -- {q!r} @ {loc}  [{topo}] " + "-" * max(0, 40 - len(q) - len(loc) - len(topo)), file=out)
    print(f"     approved      : {yn(approved)}  ({approved})", file=out)
    print(f"     mean_precision: {bar(prec)} {fmt(prec)}", file=out)
    print(f"     loops/repairs : {loops} / {repairs}", file=out)
    print(f"     final paths   : {npaths}", file=out)
    print(f"     llm_calls     : A={calls.get('agent_a')}  B={calls.get('agent_b')}", file=out)

    # Per-path verdicts from the final critique, if present.
    critique = trace.get("critique", {}) or {}
    per_path = critique.get("per_path", []) or []
    if per_path:
        print(f"     per-path verdicts ({len(per_path)}):", file=out)
        for i, pp in enumerate(per_path, 1):
            verdicts = pp.get("verdicts", {}) or {}
            compact = " ".join(
                f"{k.split()[0][:4]}:{v}" for k, v in verdicts.items()
            )
            ps = pp.get("precision_score", 0.0) or 0.0
            text = (pp.get("path") or "")[:54]
            print(f"        {i}. {yn(pp.get('approved'))} p={fmt(ps)}  {compact}", file=out)
            if text:
                print(f"           \"{text}{'...' if len(pp.get('path') or '') > 54 else ''}\"", file=out)

    # Sequential per-iteration precision trajectory, if present.
    iters = trace.get("iterations", []) or []
    if iters:
        traj = []
        for it in iters:
            c = (it.get("critique") or {}) if isinstance(it, dict) else {}
            traj.append(c.get("mean_precision", it.get("mean_precision")))
        traj = [t for t in traj if t is not None]
        if traj:
            arrow = " -> ".join(fmt(t) for t in traj)
            print(f"     precision/loop: {arrow}", file=out)


def print_summary(summary: dict, out=sys.stdout) -> None:
    print("\n" + "=" * 64, file=out)
    print("  SUMMARY (aggregated per topology)", file=out)
    print("=" * 64, file=out)
    if not summary:
        print("  (no summary file found)", file=out)
        return

    # Collect the union of metric keys across topologies, in a sensible order.
    preferred = ["n_runs", "approval_rate", "mean_precision",
                 "avg_loops", "avg_repairs",
                 "avg_calls_agent_a", "avg_calls_agent_b"]
    seen = set()
    keys = []
    for topo, metrics in summary.items():
        if isinstance(metrics, dict):
            for k in metrics:
                seen.add(k)
    keys = [k for k in preferred if k in seen] + sorted(seen - set(preferred))

    topos = list(summary.keys())
    label_w = max((len(k) for k in keys), default=10)
    col_w = max(12, max((len(t) for t in topos), default=12))

    header = " " * (label_w + 2) + "".join(f"{t:>{col_w}}" for t in topos)
    print(header, file=out)
    print(" " * (label_w + 2) + "".join(f"{'-'*(col_w-1):>{col_w}}" for _ in topos), file=out)
    for k in keys:
        row = f"  {k:<{label_w}}"
        for t in topos:
            val = (summary.get(t) or {}).get(k, "-")
            row += f"{fmt(val):>{col_w}}"
        print(row, file=out)


# ---------- main -----------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect an ablation run.")
    ap.add_argument("run", nargs="?", default=None,
                    help="timestamp (e.g. 20260616_135829) or path to a .jsonl; "
                         "default = most recent in runs/")
    ap.add_argument("--runs-dir", default="runs", help="directory holding run files")
    ap.add_argument("--summary-only", action="store_true",
                    help="print only the aggregated summary table")
    ap.add_argument("--out", "-o", default=None, metavar="FILE",
                    help="write the report to FILE (UTF-8) instead of the terminal")
    args = ap.parse_args()

    jsonl = resolve_jsonl(args.run, args.runs_dir)
    if not jsonl:
        where = args.run or f"{args.runs_dir}/ablation_*.jsonl"
        print(f"No run found for: {where}", file=sys.stderr)
        return 1

    # Route report output to a file if --out was given, else the terminal.
    # Explicit UTF-8 so the box-drawing chars (# - [x] ->) survive on Windows.
    out = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    try:
        print("=" * 64, file=out)
        print(f"  RUN: {os.path.basename(jsonl)}", file=out)
        print("=" * 64, file=out)

        records = load_jsonl(jsonl)
        print(f"  {len(records)} record(s) "
              f"({len({r.get('query') for r in records})} queries x "
              f"{len({r.get('topology') for r in records})} topologies)", file=out)

        if not args.summary_only:
            # Group by topology for readability.
            by_topo: dict[str, list[dict]] = {}
            for r in records:
                by_topo.setdefault(r.get("topology", "?"), []).append(r)
            for topo in sorted(by_topo):
                print(f"\n{'-'*64}\n  TOPOLOGY: {topo}\n{'-'*64}", file=out)
                for rec in by_topo[topo]:
                    print_record(rec, out=out)

        sp = summary_path_for(jsonl)
        summary = {}
        if sp:
            try:
                with open(sp, encoding="utf-8") as f:
                    summary = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"  ! could not read summary: {e}", file=sys.stderr)
        print_summary(summary, out=out)

        print(file=out)
    finally:
        if out is not sys.stdout:
            out.close()
            print(f"Wrote report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
