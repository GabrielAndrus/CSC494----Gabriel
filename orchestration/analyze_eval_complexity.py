"""
analyze_eval_complexity.py -- Analyze the complexity of evaluation sets.

No model calls. Measures question length, verified-points count/length, topic
diversity, vocab richness -- PLUS (added 2026-09-14, see "WHY THIS WAS
AUGMENTED" below) a second set of axes aimed at judgment-style items like
NormAd, where the original open-QA metrics systematically read as "simple" no
matter how hard the underlying reasoning actually is.

Usage:
    python orchestration/analyze_eval_complexity.py runs/eval_n100.jsonl
    python orchestration/analyze_eval_complexity.py runs/eval_n100.jsonl --out runs/complexity.json
    python orchestration/analyze_eval_complexity.py candidate_datasets/normad_eval_set.jsonl --task-type judgment

    Or, if using python3...

    python3 orchestration/analyze_eval_complexity.py runs/eval_n100.jsonl

Input:
    JSONL file with records: {query, location, sub_topic, ground_truth: {verified_points}}

Output:
    JSON dict with complexity metrics (stdout, optionally to file).
    Metrics: question_length, verified_points_{count,length}, topic_diversity,
    vocab_size, PLUS narrative_beats, action_density, lexical_specificity,
    ambiguity_rate. Two classification views (open_qa-style, judgment-style),
    both always computed -- see "WHY THIS WAS AUGMENTED".

Dependencies: Python stdlib only (json, argparse, re). Nothing to pip-install.

Gabriel Andrus, 8/28/26. Augmented 2026-09-14 (NormAd-aware complexity axes).

================================================================================
WHY THIS WAS AUGMENTED (2026-09-14)
================================================================================

The original version of this script defined "sufficiently complex" as ONE rule:
verified_points >= 8 AND question_length >= 30 words. That threshold was tuned
against GlobalCultureQA, where a rich answer really does mean "many decomposed
facts" and a hard question really is a long one.

Running it as-is against an 18-row NormAd sample (see
candidate_datasets/NORMAD_REPORT.md) produced 0/18 items classified as
"sufficiently complex" -- not because those items are trivial (they're
social-acceptability judgments that can require real cultural reasoning to get
right), but because NormAd's ground truth is STRUCTURALLY shallow by design: one
Rule-of-Thumb, one verdict, a sentence or two of justification. No matter how
elaborate the scenario is, `verified_points` will rarely exceed 3-4. The metric
was measuring "how many atomic facts did the dataset authors write down," which
happens to correlate with question difficulty on GlobalCultureQA and does NOT
on NormAd. Reusing a GlobalCultureQA-tuned number on a differently-shaped task
was silently wrong, not just imprecise.

The temptation at this point is to design ONE better number -- a "cultural
reasoning score" that supposedly captures difficulty for any dataset. That
was deliberately NOT done here. Cultural complexity is not one thing: whether
an item is hard depends on how many actions are bundled into one scenario, how
niche vs. universal the invoked norm is, whether the "right" answer is
genuinely contested, how much of the judgment rests on cultural knowledge vs.
common sense, and probably other axes nobody has named yet -- most of which are
subjective and none of which reduce cleanly to a scalar. Collapsing all of that
into a single score just relocates the same problem (which weights? whose
judgment of "complex"?) one level down, while looking more authoritative than
it is. So instead of one smarter metric, this script now reports a wider VECTOR
of cheap, legible signals and lets two different classification RULES draw the
"is this complex" line differently depending on task shape -- both always
printed side by side, so the disagreement between them is visible instead of
hidden behind a single number.

New axes (all still O(n), stdlib-only, no model calls):

  - narrative_beats: sentence count in the query. A scenario is harder to
    reason about the more distinct events it strings together before asking
    for a verdict, independent of how long any single sentence is.
  - action_density: count of sequencing/coordinating words (and, then, after,
    before, while, when, but, so) in the query. A crude proxy for how many
    distinct actions or a story bundles -- more bundled actions means more
    candidate distractors for "which action is the culturally-loaded one."
  - lexical_specificity: fraction of non-stopword tokens in the ground-truth
    verified_points. A rule stated in generic words ("be polite", "show
    respect") scores low; a rule naming a specific custom, object, or gesture
    scores high. This is a WEAK proxy (word-frequency, not meaning) and is
    computed as a percentile WITHIN the current batch, not against a fixed
    number, because "specific" is relative to what a dataset's rules usually
    look like, not an absolute constant one dataset can set for all future
    ones.
  - ambiguity_rate / ambiguity_flag: whether the verified_points text itself
    contains hedging/contested-ness language ("ambiguous", "context-dependent",
    "depends", "no clear", "varies", "debatable", "subjective", ...). This is
    a generic, schema-agnostic way to surface NormAd-style "neutral" /
    contested items (or any future dataset's equivalent) WITHOUT hardcoding a
    NormAd-specific field name, so it generalizes to whatever the next
    candidate dataset turns out to be.

None of these are individually trustworthy either -- action_density is fooled
by a story that just happens to use "and" a lot; lexical_specificity is fooled
by proper nouns that aren't culturally loaded. They're reported as a profile
so a human reviewing a candidate dataset can see the shape of the disagreement,
not to be trusted as a ground truth complexity oracle.

Two classification views, always both computed:

  1. `open_qa` (the ORIGINAL rule, unchanged): verified_points >= min_vp AND
     question_length >= min_q_len. Still the right call for GlobalCultureQA-
     shaped data (rich, hand-decomposed multi-fact answers).
  2. `judgment` (NEW): narrative_beats >= min_narrative_beats AND
     (lexical_specificity >= this batch's median OR ambiguity_flag is set).
     Built for NormAd-shaped data (one rule, one verdict, a short story) --
     "complex" here means "multi-step scenario AND (the invoked norm is
     unusually specific OR the case is genuinely contested)", not "many
     verified points," because verified-point count is structurally
     uninformative for this task shape (see above).

`detect_task_type()` guesses which view is the intended primary one per
dataset (by checking how many queries look like a yes/no acceptability
question), but BOTH views are always printed / written to --out, labeled
primary vs. alternate, specifically so a mismatch like the NormAd 0%-complex
result above is visible on every future dataset instead of silently
recurring.

================================================================================
HOW TO READ RESULTS
================================================================================

PART 1: PER-ITEM ANALYSIS
-------------------------

Format:
    [idx] location        | topic           | Q_len=NN VP_count=NN beats=N conn=N spec=0.NN ambig=Y/N
          Q: [first 90 chars of question]
          Verified points:
            1. [first verified point, up to 80 chars]
            2. [second verified point]
            3. [third verified point]
            ... and N more

Example:
    [  0] China           | breakfast       | Q_len=18 VP_count= 6 beats=1 conn=0 spec=0.62 ambig=N
          Q: What is a traditional Chinese breakfast and what does it typically inc
          Verified points:
            1. Traditional Chinese breakfasts often include congee (rice porridge).
            2. Youtiao (fried dough sticks) are a common breakfast item.
            3. Soy milk is frequently served alongside breakfast dishes.
            ... and 3 more

How to read each field:

  • [idx]
    Item number (0-indexed). Use this to track a specific item across runs.

  • location / topic
    Cultural group / sub-category, left-aligned, padded to 15 characters.

  • Q_len
    Question length in WORDS (count of space-separated tokens).
    - Q_len=5-10: SHORT questions, usually simple factual ("What is X?")
    - Q_len=15-25: MEDIUM questions, may require some context ("How is X practiced?")
    - Q_len=30+: LONG questions, likely multi-step reasoning, OR (see NormAd)
      simply a longer narrative setup that doesn't by itself imply a richer
      answer -- read this alongside beats/conn, not alone.

  • VP_count
    Number of verified points (atomic knowledge units) in the ground truth.
    - VP_count=1-3: FEW -- simple factual answer, OR (see NormAd) a task whose
      ground truth is structurally a single rule + verdict regardless of how
      hard the judgment is. Do not read low VP_count as "easy" without also
      checking beats/spec/ambig for judgment-shaped data.
    - VP_count=4-7: MODERATE. VP_count=8+: MANY, rich multi-faceted answer.

  • beats (narrative_beats)
    Sentence count in the query. More beats = more events to track before the
    question can be answered.

  • conn (action_density)
    Count of sequencing/coordinating words (and/then/after/before/while/
    when/but/so) in the query. Rough proxy for bundled actions.

  • spec (lexical_specificity)
    Fraction (0.00-1.00) of non-stopword tokens in the verified_points text.
    Higher = the rule/answer is stated in more specific, less generic language.

  • ambig (ambiguity_flag)
    Y if the verified_points text contains hedging/contested-ness language
    (see the marker list above). Flags likely-contested items generically.

What this tells you:
  - High Q_len + high VP_count = a COMPLEX open-QA item.
  - High beats + (high spec OR ambig=Y) = a COMPLEX judgment item, even with
    VP_count as low as 2-3.
  - Low across the board = plausibly a SIMPLE item either way.
  - A judgment item with high VP_count would be unusual and worth a second
    look (possibly mis-normalized).


PART 2: AGGREGATE COMPLEXITY PROFILE
-------------------------------------

  1. "question_length" {min, max, mean} -- words per question.
       mean < 15: mostly simple/direct. 15-30: moderate. > 30: verbose/multi-step
       (or, for judgment data, just a longer scenario -- see caveat above).

  2. "verified_points_per_item" {min, max, mean} -- atomic facts per answer.
       mean < 4: shallow ground truth (expected & fine for judgment-shaped
       tasks). mean 4-7: moderate. mean > 7: rich multi-faceted answers
       (open-QA-shaped tasks).

  3. "verified_points_total_words" {min, max, mean} -- ground-truth "volume."

  4. "topic_diversity" / "location_diversity" -- coverage breadth.

  5. "vocabulary_size" -- unique words across all verified points.

  6. "narrative_beats" {min, max, mean} -- sentence count per query.
       mean ~1: single-beat scenarios. mean 2+: multi-step scenarios, more
       opportunity for a distractor action to be mistaken for the norm-
       relevant one.

  7. "action_density" {min, max, mean} -- sequencing-word count per query.
       Higher means more bundled actions per query on average.

  8. "lexical_specificity" {min, max, mean, median} -- non-stopword fraction
       of verified_points text. The median here IS the adaptive threshold the
       judgment classifier uses (see below) -- it moves with the batch on
       purpose, since "specific" only means something relative to a
       dataset's own baseline phrasing.

  9. "ambiguity_rate" -- fraction of items whose verified_points contain a
       hedging/contested-ness marker. High values suggest a dataset (or
       sample) that leans on genuinely contested cases rather than settled
       facts -- informative on its own, independent of any threshold.


PART 3: TWO COMPLEXITY CLASSIFICATIONS (always both; primary is marked)
------------------------------------------------------------------------

`open_qa` classification (UNCHANGED from the original baseline, 8/28/26):
  complex iff verified_points >= min_vp (default 8) AND question_length >=
  min_q_len (default 30 words).

  Rationale (original): verified_points >= 8 means rich ground truth with
  scaffolding material for augmentation to work with; question_length >= 30
  implies multi-step reasoning rather than simple recall. Tuned for
  GlobalCultureQA; iterate per the ORIGINAL "ITERATING ON THE THRESHOLD"
  section retained below.

`judgment` classification (NEW, 2026-09-14, exploratory -- expect to retune):
  complex iff narrative_beats >= min_narrative_beats (default 2) AND
  (lexical_specificity >= this batch's median specificity OR ambiguity_flag).

  Rationale: verified_points count is not a meaningful richness signal for a
  single-rule judgment task (see "WHY THIS WAS AUGMENTED"), so richness is
  read instead from (a) how many events the scenario bundles, crossed with
  either (b) how specific/non-generic the invoked norm's language is, or (c)
  whether the case is flagged as genuinely contested. A multi-beat scenario
  resting on a generic, uncontested norm ("be polite") is NOT counted complex
  even though it's a longer story -- length alone was exactly the wrong signal
  discovered on NormAd.

  This threshold is a first pass, calibrated on an 18-item NormAd sample --
  treat it the same way the original threshold below asks you to treat itself:
  as a starting point to retune once real model performance is observed on
  judgment-shaped items (e.g., if items with ambig=Y turn out to be where
  augmentation helps most, that alone might become sufficient without needing
  the beats/spec conjunction; adjust here and log the change).


ITERATING ON EITHER THRESHOLD
-------------------------------

As we experiment and pivot to new datasets, we will:
  1. Measure performance on classified-complex vs. non-complex items separately
     (using whichever view -- open_qa or judgment -- matches the task shape).
  2. Observe where augmentation helps (and hurts).
  3. Adjust thresholds based on empirical findings, and document the change
     here (both the old and new value, and why).

For example:
  - If augmentation helps on VP >= 10 but hurts on VP = 8-9, raise the open_qa
    threshold to 10.
  - If ambig=Y alone turns out to predict where augmentation helps on judgment
    data, drop the narrative_beats/specificity conjunction and gate on that
    alone.
  - If a new dataset shape shows up that neither view fits (e.g. multi-turn
    dialogue), add a THIRD view rather than stretching these two to cover it.
"""

from __future__ import annotations
import argparse
import json
import logging
import re
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("analyze_eval_complexity")


# --- New-axis vocabulary (deliberately small, stdlib, English-only; see
# "WHY THIS WAS AUGMENTED" for why these stay weak proxies on purpose) --------

_STOPWORDS = frozenset("""
a an the and or but so if then than that this these those is are was were be
been being to of in on at for with by from as it its it's he she they them
his her their you your yours we our ours i me my mine do does did done have
has had not no nor can could should would may might will shall about into
over under again further once here there when where why how all any both
each few more most other some such only own same too very just also which
who whom what when while because during before after above below between
out up down off again into through during
""".split())

_CONNECTIVES = ("and", "then", "after", "before", "while", "when", "but", "so")

_HEDGE_MARKERS = (
    "ambiguous", "context-dependent", "context dependent", "depends",
    "no clear", "not necessarily", "may vary", "varies", "mixed", "debatable",
    "subjective", "contested", "uncertain", "could be seen", "not applicable",
    "does not apply", "do not apply", "not relevant",
)

_JUDGMENT_QUERY_RE = re.compile(
    r"\b(is|was|would|were|did)\b.{0,200}\b"
    r"(acceptable|appropriate|polite|rude|impolite|inappropriate|offensive|okay|ok)\b"
    r"\s*\??\s*$",
    re.IGNORECASE,
)


def load_items(jsonl_path: str) -> list[dict]:
    """Load evaluation items from JSONL file."""
    items = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    log.info("Loaded %d items from %s", len(items), jsonl_path)
    return items


def _split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", (text or "").lower())


def _narrative_beats(query: str) -> int:
    return max(1, len(_split_sentences(query)))


def _action_density(query: str) -> int:
    tokens = _tokenize(query)
    return sum(1 for t in tokens if t in _CONNECTIVES)


def _lexical_specificity(verified_points: list[str]) -> float:
    tokens = _tokenize(" ".join(verified_points))
    if not tokens:
        return 0.0
    content = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    return len(content) / len(tokens)


def _ambiguity_flag(verified_points: list[str]) -> bool:
    text = " ".join(verified_points).lower()
    return any(marker in text for marker in _HEDGE_MARKERS)


def _looks_like_judgment_query(query: str) -> bool:
    return bool(_JUDGMENT_QUERY_RE.search((query or "").strip()))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def compute_per_item_signals(items: list[dict]) -> list[dict]:
    """All per-item signals used by both the profile and both classifiers."""
    out = []
    for item in items:
        query = item["query"]
        vps = item["ground_truth"]["verified_points"]
        out.append({
            "query": query,
            "location": item["location"],
            "sub_topic": item["sub_topic"],
            "q_len": len(query.split()),
            "vp_count": len(vps),
            "vp_len": sum(len(vp.split()) for vp in vps),
            "narrative_beats": _narrative_beats(query),
            "action_density": _action_density(query),
            "lexical_specificity": _lexical_specificity(vps),
            "ambiguity_flag": _ambiguity_flag(vps),
            "looks_like_judgment": _looks_like_judgment_query(query),
            "verified_points": vps,
        })
    return out


def detect_task_type(signals: list[dict]) -> dict:
    """
    Guess whether this eval set is shaped like GlobalCultureQA (open_qa) or
    like NormAd (judgment), by checking how many queries read as yes/no
    acceptability questions. Heuristic, not authoritative -- override with
    --task-type when known.
    """
    n = len(signals) or 1
    n_judgment_like = sum(1 for s in signals if s["looks_like_judgment"])
    fraction = n_judgment_like / n
    detected = "judgment" if fraction >= 0.5 else "open_qa"
    return {
        "detected": detected,
        "n_judgment_like_queries": n_judgment_like,
        "n_total": n,
        "fraction_judgment_like": fraction,
    }


def compute_complexity_profile(items: list[dict]) -> dict:
    """Compute aggregate complexity metrics (original axes + new axes)."""
    n = len(items)
    signals = compute_per_item_signals(items)

    q_lengths = [s["q_len"] for s in signals]
    vp_counts = [s["vp_count"] for s in signals]
    vp_lengths = [s["vp_len"] for s in signals]
    beats = [s["narrative_beats"] for s in signals]
    conn = [s["action_density"] for s in signals]
    spec = [s["lexical_specificity"] for s in signals]
    ambig_flags = [s["ambiguity_flag"] for s in signals]

    topics = [i["sub_topic"] for i in items]
    locations = [i["location"] for i in items]

    all_words = set()
    for item in items:
        for vp in item["ground_truth"]["verified_points"]:
            all_words.update(vp.lower().split())

    def _stats(vals):
        return {"min": min(vals), "max": max(vals), "mean": sum(vals) / n}

    profile = {
        "n_items": n,
        "question_length": _stats(q_lengths),
        "verified_points_per_item": _stats(vp_counts),
        "verified_points_total_words": _stats(vp_lengths),
        "topic_diversity": {
            "unique_topics": len(set(topics)),
            "topics": sorted(set(topics)),
        },
        "location_diversity": {
            "unique_locations": len(set(locations)),
            "locations": sorted(set(locations)),
        },
        "vocabulary_size": len(all_words),
        # --- new axes (2026-09-14) ---
        "narrative_beats": _stats(beats),
        "action_density": _stats(conn),
        "lexical_specificity": {
            **_stats(spec),
            "median": _median(spec),
        },
        "ambiguity_rate": sum(ambig_flags) / n,
    }
    return profile


def classify_open_qa(items: list[dict], min_vp: int = 8, min_q_len: int = 30) -> dict:
    """
    ORIGINAL classification rule (unchanged). Tuned for GlobalCultureQA-shaped
    data: rich, hand-decomposed multi-fact answers to open questions.
    """
    complex_items, simple_items = [], []
    for item in items:
        vp_count = len(item["ground_truth"]["verified_points"])
        q_len = len(item["query"].split())
        (complex_items if (vp_count >= min_vp and q_len >= min_q_len) else simple_items).append(item)

    n_total = len(items) or 1
    return {
        "view": "open_qa",
        "classification_threshold": {"min_verified_points": min_vp, "min_question_words": min_q_len},
        "complex_items": complex_items,
        "simple_items": simple_items,
        "n_complex": len(complex_items),
        "n_simple": len(simple_items),
        "pct_complex": len(complex_items) / n_total * 100,
        "pct_simple": len(simple_items) / n_total * 100,
    }


def classify_judgment(items: list[dict], min_narrative_beats: int = 2) -> dict:
    """
    NEW classification rule for judgment-shaped data (NormAd-like): a single
    rule + verdict, where verified_points count is structurally uninformative.
    complex iff narrative_beats >= min_narrative_beats AND (lexical_specificity
    >= this batch's median OR ambiguity_flag). See "WHY THIS WAS AUGMENTED".
    """
    signals = compute_per_item_signals(items)
    spec_values = [s["lexical_specificity"] for s in signals]
    median_spec = _median(spec_values)

    complex_items, simple_items = [], []
    for item, s in zip(items, signals):
        is_complex = (
            s["narrative_beats"] >= min_narrative_beats
            and (s["lexical_specificity"] >= median_spec or s["ambiguity_flag"])
        )
        (complex_items if is_complex else simple_items).append(item)

    n_total = len(items) or 1
    return {
        "view": "judgment",
        "classification_threshold": {
            "min_narrative_beats": min_narrative_beats,
            "specificity_threshold_this_batch_median": median_spec,
        },
        "complex_items": complex_items,
        "simple_items": simple_items,
        "n_complex": len(complex_items),
        "n_simple": len(simple_items),
        "pct_complex": len(complex_items) / n_total * 100,
        "pct_simple": len(simple_items) / n_total * 100,
    }


# Backward-compat alias: old callers/scripts importing `classify_by_complexity`
# get the original open_qa behavior unchanged.
def classify_by_complexity(items: list[dict], min_vp: int = 8, min_q_len: int = 30) -> dict:
    return classify_open_qa(items, min_vp=min_vp, min_q_len=min_q_len)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl_path", help="Path to eval set JSONL file")
    ap.add_argument("--out", help="Optional output JSON file for aggregate profile")
    ap.add_argument("--min-vp", type=int, default=8,
                    help="[open_qa view] Minimum verified points for 'complex' (default: 8)")
    ap.add_argument("--min-q-len", type=int, default=30,
                    help="[open_qa view] Minimum question word count for 'complex' (default: 30)")
    ap.add_argument("--min-narrative-beats", type=int, default=2,
                    help="[judgment view] Minimum query sentence count for 'complex' (default: 2)")
    ap.add_argument("--task-type", choices=["auto", "open_qa", "judgment"], default="auto",
                    help="Which view to mark as PRIMARY. 'auto' guesses from query phrasing "
                         "(default: auto). Both views are always computed regardless.")
    args = ap.parse_args()

    items = load_items(args.jsonl_path)
    signals = compute_per_item_signals(items)
    profile = compute_complexity_profile(items)
    type_info = detect_task_type(signals)
    primary_type = type_info["detected"] if args.task_type == "auto" else args.task_type

    open_qa_result = classify_open_qa(items, min_vp=args.min_vp, min_q_len=args.min_q_len)
    judgment_result = classify_judgment(items, min_narrative_beats=args.min_narrative_beats)
    views = {"open_qa": open_qa_result, "judgment": judgment_result}

    # Per-item details
    print("\n" + "=" * 100)
    print("PER-ITEM ANALYSIS")
    print("=" * 100)
    for i, (item, s) in enumerate(zip(items, signals)):
        print(f"\n[{i:3d}] {item['location']:15} | {item['sub_topic']:15} | "
              f"Q_len={s['q_len']:2d} VP_count={s['vp_count']:2d} "
              f"beats={s['narrative_beats']} conn={s['action_density']} "
              f"spec={s['lexical_specificity']:.2f} ambig={'Y' if s['ambiguity_flag'] else 'N'}")
        print(f"        Q: {item['query'][:90]}")
        print(f"        Verified points:")
        for j, vp in enumerate(item["ground_truth"]["verified_points"][:3]):
            print(f"          {j+1}. {vp[:80]}")
        if len(item["ground_truth"]["verified_points"]) > 3:
            print(f"          ... and {len(item['ground_truth']['verified_points']) - 3} more")

    # Aggregate profile
    print("\n" + "=" * 100)
    print("AGGREGATE COMPLEXITY PROFILE")
    print("=" * 100)
    print(json.dumps(profile, indent=2))

    # Task-type detection
    print("\n" + "=" * 100)
    print("TASK-TYPE DETECTION")
    print("=" * 100)
    print(json.dumps(type_info, indent=2))
    print(f"\nPrimary view: {primary_type}"
          f"{' (forced via --task-type)' if args.task_type != 'auto' else ' (auto-detected)'}")

    # Both classifications, primary marked
    for view_name, result in views.items():
        tag = "PRIMARY" if view_name == primary_type else "alternate (for comparison)"
        print("\n" + "=" * 100)
        print(f"COMPLEXITY CLASSIFICATION -- {view_name} view [{tag}]")
        print("=" * 100)
        print(f"Threshold: {result['classification_threshold']}")
        print(f"\n  Complex: {result['n_complex']:3d} items ({result['pct_complex']:5.1f}%)")
        print(f"  Simple:  {result['n_simple']:3d} items ({result['pct_simple']:5.1f}%)")

    if args.out:
        def _brief(result):
            return {
                "threshold": result["classification_threshold"],
                "n_complex": result["n_complex"],
                "n_simple": result["n_simple"],
                "pct_complex": result["pct_complex"],
                "pct_simple": result["pct_simple"],
                "complex_items": [
                    {"idx": i, "query": it["query"], "location": it["location"],
                     "sub_topic": it["sub_topic"],
                     "verified_points_count": len(it["ground_truth"]["verified_points"])}
                    for i, it in enumerate(items) if it in result["complex_items"]
                ],
            }
        output_data = {
            "aggregate_profile": profile,
            "task_type_detection": type_info,
            "primary_view": primary_type,
            "classification": {name: _brief(result) for name, result in views.items()},
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        log.info("Full analysis written to %s", args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
