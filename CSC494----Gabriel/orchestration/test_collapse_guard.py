#!/usr/bin/env python
"""
test_collapse_guard.py - Exercise the sequential repair path-collapse guard
WITHOUT a GPU or any model. Runs on the Narval login node in seconds.

It replicates the guard's decision rule exactly as written in orchestrator.py:

    n_before       = len(current)
    n_after        = len(repaired) if repaired else 0
    collapse_floor = max(2, int(0.5 * n_before))
    reject (collapsed=True) iff  n_after < collapse_floor

We assert the guard:
  - REJECTS a repair that returns 0 paths (the `-> 0` collapse you saw live)
  - REJECTS a repair that returns 1 path from many (the `27 -> 1` collapse)
  - REJECTS a shrink below 50% retention
  - ACCEPTS a healthy repair that keeps >=50% of paths
  - ACCEPTS a repair that grows the path set

If all pass, the guard logic is verified before the 12h run.
"""

def guard_decision(current, repaired):
    """Mirror of the orchestrator guard; returns (collapsed, n_after, floor)."""
    n_before = len(current)
    n_after = len(repaired) if repaired else 0
    collapse_floor = max(2, int(0.5 * n_before))
    collapsed = n_after < collapse_floor
    return collapsed, n_after, collapse_floor


def case(name, n_before, n_after, expect_collapsed):
    current = list(range(n_before))
    repaired = list(range(n_after))
    collapsed, na, floor = guard_decision(current, repaired)
    ok = (collapsed == expect_collapsed)
    print("%-32s before=%2d after=%2d floor=%2d -> collapsed=%-5s  %s"
          % (name, n_before, na, floor, str(collapsed),
             "PASS" if ok else "*** FAIL (expected collapsed=%s) ***" % expect_collapsed))
    return ok


def main():
    results = [
        # The two collapses you actually observed in the n=5 run:
        case("repair -> 0 paths (empty)",        12, 0,  True),
        case("repair 27 -> 1 path",              27, 1,  True),
        # Boundary / shrink cases:
        case("repair 12 -> 5 (below 50%)",       12, 5,  True),   # floor=6, 5<6 reject
        case("repair 12 -> 6 (exactly 50%)",     12, 6,  False),  # floor=6, 6>=6 accept
        case("repair 4 -> 2 (small set, =floor)", 4, 2,  False),  # floor=2, 2>=2 accept
        case("repair 4 -> 1 (small set, <floor)", 4, 1,  True),   # floor=2, 1<2 reject
        # Healthy repairs:
        case("repair 20 -> 18 (healthy)",        20, 18, False),
        case("repair 12 -> 15 (grew)",           12, 15, False),
        # Edge: repaired is None-like (empty)
        case("repair -> None/[] from 10",        10, 0,  True),
    ]
    print("-" * 78)
    if all(results):
        print("ALL %d CASES PASS - guard rejects collapses, accepts healthy repairs." % len(results))
        return 0
    print("SOME CASES FAILED - guard logic does not match expectations; review before run.")
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())