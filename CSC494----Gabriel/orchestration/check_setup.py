"""
Sanity check — run BEFORE the full ablation to confirm the orchestration layer
dropped into the right place and can see both repos.

    cd CSC494
    python -m orchestration.check_setup

Prints the resolved repo paths and confirms the CulFiT EVAL prompts and the
Tonga generation prompts load from the expected files. Exits non-zero on any
problem, so it's safe to chain before a run:  python -m orchestration.check_setup && sbatch ...
"""

from __future__ import annotations
import os
import sys


def main() -> int:
    print("=== PluralTree setup check ===")
    from orchestration import bootstrap

    # 1. Resolved locations
    print(f"CSC494 root : {bootstrap.CSC494}")
    print(f"CulFiT      : {bootstrap.CULFIT}")
    print(f"Agent A     : {bootstrap.AGENT_A}")

    ok = True
    for label, path in [("CulFiT", bootstrap.CULFIT),
                        ("Agent A", bootstrap.AGENT_A),
                        ("Agent A src", bootstrap.AGENT_A_SRC)]:
        exists = os.path.isdir(path)
        print(f"  [{'OK' if exists else 'MISSING'}] {label}: {path}")
        ok = ok and exists
    if not ok:
        print("\nFAIL: a repo folder is missing. Is orchestration/ a direct child "
              "of CSC494/, and are the repo folder names exactly correct?")
        return 1

    # 2. Load prompts via bootstrap
    try:
        mods = bootstrap.install()
    except Exception as e:
        print(f"\nFAIL: bootstrap.install() raised: {e}")
        return 1

    # 3. Confirm CulFiT EVAL prompts
    pu = mods["culfit_prompt_utils"]
    eval_names = ["EVAL_CULTURAL_GROUP_PROMPT", "EVAL_TOPIC_PROMPT",
                  "EVAL_CULTURAL_POINTS_PROMPT"]
    missing_eval = [n for n in eval_names if not hasattr(pu, n)]
    print(f"\nCulFiT prompt_utils -> {pu.__file__}")
    if missing_eval:
        print(f"  [FAIL] missing EVAL prompts: {missing_eval}")
        ok = False
    else:
        print(f"  [OK] EVAL prompts present: {eval_names}")

    # 4. Confirm Tonga generation prompts
    pt = mods["agentA_prompt_templates"]
    print(f"Agent A prompt_templates -> {pt.__file__}")
    if not hasattr(pt, "generation_prompts"):
        print("  [FAIL] generation_prompts not found")
        ok = False
    else:
        keys = list(pt.generation_prompts.keys())
        print(f"  [OK] generation_prompts keys: {keys}")

    # 5. Confirm parser import
    rp = mods["agentA_response_parser"]
    print(f"Agent A response_parser -> {rp.__file__}")
    if not hasattr(rp, "parse_llm_response"):
        print("  [FAIL] parse_llm_response not found")
        ok = False
    else:
        print("  [OK] parse_llm_response present")

    print("\n" + ("PASS — setup looks good." if ok else "FAIL — see issues above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
