#!/usr/bin/env python
"""
score_debate.py -- Score the multi-agent debate against paper Table 2 (row LLAMA-3 + GEMMA-2).

Metrics (accuracy %, paper scorer AND strict first-word parser, like score.py):
  D(Llama-3)  : Llama-3-8B's final decision after debate
  D(Gemma-2)  : Gemma-2-9B's final decision after debate
  D           : adjudicated decision = the shared final decision if the agents agree,
                otherwise the Gemma-2-27B judge's decision (needs the judge file)
Also: agreement rate, number of judged items, judge accuracy on judged items.

    python score_debate.py ../results_narval/debate_fixed.jsonl [--judge ...judge.jsonl]
"""

from __future__ import annotations
import argparse
import json
import sys

from common import paper_predict, strict_predict

# Paper Table 2, row LLAMA-3 (M1) + GEMMA-2 (M2), rule-of-thumb in all prompts
PAPER = {"Si(Llama-3)": 63.7, "Si(Gemma-2)": 68.9, "Ora": 82.6,
         "D(Llama-3)": 66.5, "D(Gemma-2)": 76.7, "D": 79.7}
LABELS = {"yes", "no", "neutral"}


def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("debate")
    ap.add_argument("--judge", default=None, help="default: <debate>.judge.jsonl if it exists")
    args = ap.parse_args()

    deb = load(args.debate)
    judge = {}
    jp = args.judge or args.debate + ".judge.jsonl"
    try:
        judge = {j["ID"]: j for j in load(jp)}
    except FileNotFoundError:
        print(f"(no judge file at {jp}: adjudicated D will be skipped)")
    n = len(deb)
    variant = deb[0].get("variant", "?") if deb else "?"

    res = {}
    for name, fn in (("paper", paper_predict), ("strict", strict_predict)):
        ok_l = ok_g = ok_d = 0
        agree = judged = judged_ok = unjudged_disagree = 0
        for d in deb:
            gold = d["Gold Label"].lower()
            pl, pg = fn(d["llama3_final"]), fn(d["gemma_final"])
            ok_l += pl == gold
            ok_g += pg == gold
            if pl == pg and pl in LABELS:
                agree += 1
                ok_d += pl == gold
            elif d["ID"] in judge:
                judged += 1
                pj = fn(judge[d["ID"]]["judge_generation"])
                ok_d += pj == gold
                judged_ok += pj == gold
            else:
                unjudged_disagree += 1
        res[name] = dict(l=ok_l / n * 100, g=ok_g / n * 100, d=ok_d / n * 100,
                         agree=agree / n * 100, judged=judged,
                         judge_acc=(judged_ok / judged * 100 if judged else None),
                         unjudged=unjudged_disagree)

    print("=" * 96)
    print(f"MULTI-AGENT DEBATE (Llama-3-8B + Gemma-2-9B, variant={variant}), n={n}")
    print("=" * 96)
    print(f"{'metric':22} {'paper scorer':>13} {'strict':>9} {'paper Table 2':>15}")
    for label, key, ref in (("D(Llama-3) final", "l", "D(Llama-3)"), ("D(Gemma-2) final", "g", "D(Gemma-2)")):
        print(f"{label:22} {res['paper'][key]:13.1f} {res['strict'][key]:9.1f} {PAPER[ref]:15.1f}")
    if judge or res["paper"]["unjudged"] == 0:
        print(f"{'D adjudicated (judge)':22} {res['paper']['d']:13.1f} {res['strict']['d']:9.1f} {PAPER['D']:15.1f}")
    p = res["paper"]
    print("-" * 96)
    print(f"agents agree on final decision: {p['agree']:.1f}% of items | judged: {p['judged']} | "
          f"disagreements NOT judged (counted wrong): {p['unjudged']}")
    if p["judge_acc"] is not None:
        print(f"judge accuracy on the items it decided: {p['judge_acc']:.1f}% (paper scorer)")
    print(f"(reference singles with rule, paper Table 1: Llama-3 {PAPER['Si(Llama-3)']}, "
          f"Gemma-2 {PAPER['Si(Gemma-2)']}; oracle {PAPER['Ora']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
