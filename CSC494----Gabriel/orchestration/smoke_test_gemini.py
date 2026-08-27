"""
smoke_test_gemini.py — verify the Gemini backend end to end before wiring it
into run_generate / run_judge.

Run from the CSC494 project root (where .env with GEMINI_API_KEY lives):

    python -m orchestration.smoke_test_gemini
    # or, if not packaged:
    python orchestration/smoke_test_gemini.py

Checks, in order:
  1. make_backend("gemini") constructs (key present, openai importable).
  2. A trivial chat() round-trips and returns non-empty text.
  3. A judge-style Yes/No prompt returns a parseable verdict — this is the
     shape run_judge depends on, so if this works the judge swap will too.

Exit 0 = safe to proceed to Step 1 (re-score existing answers with Gemini).
"""

from __future__ import annotations
import sys


def main() -> int:
    try:
        from orchestration.llm_backend import make_backend
    except Exception:
        # allow running as a loose script too
        from llm_backend import make_backend  # type: ignore

    # 1) construct
    try:
        be = make_backend("gemini")  # default model gemini-2.0-flash
    except Exception as e:
        print(f"[FAIL] could not construct gemini backend: {e}")
        return 1
    print(f"[ok] constructed GeminiBackend model={be.model_name}")

    # 2) trivial round-trip
    try:
        out = be.chat([{"role": "user", "content": "Reply with exactly: PONG"}])
    except Exception as e:
        print(f"[FAIL] chat() raised: {e}")
        return 1
    if not out or not out.strip():
        print("[FAIL] chat() returned empty string")
        return 1
    print(f"[ok] chat() round-trip -> {out.strip()[:60]!r}")

    # 3) judge-shaped Yes/No prompt (mirrors EVAL_CULTURAL_POINTS usage)
    judge_prompt = (
        "You are evaluating a knowledge point against a reference answer.\n"
        "Knowledge point: 'Japanese tea ceremony emphasizes the principle of "
        "ichigo ichie (one time, one meeting).'\n"
        "Reference: 'The Japanese tea ceremony is rooted in Zen and stresses "
        "that each gathering is unique and unrepeatable.'\n"
        "Is the knowledge point supported by the reference? Answer Yes or No."
    )
    try:
        verdict = be.chat([{"role": "user", "content": judge_prompt}])
    except Exception as e:
        print(f"[FAIL] judge-style chat() raised: {e}")
        return 1
    yes = "yes" in verdict.lower()
    no = "no" in verdict.lower()
    if not (yes or no):
        print(f"[WARN] verdict not parseable as Yes/No: {verdict.strip()[:80]!r}")
        print("       (backend works, but check the judge prompt/extractor.)")
        return 0
    print(f"[ok] judge-style verdict parseable -> {verdict.strip()[:40]!r} "
          f"({'Yes' if yes and not no else 'No' if no and not yes else 'ambiguous'})")

    print("\nAll checks passed. Safe to proceed to Step 1 (re-score with Gemini judge).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
