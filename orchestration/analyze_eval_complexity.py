"""
analyze_eval_complexity.py -- Analyze the complexity of evaluation sets.

Analyzes the n=100 PluralTree validation set from CulFiT's GlobalCultureQA.csv. 

No model calls. Measures: question length, verified-points count/length, topic 
diversity, vocab richness. Useful for comparing current eval sets vs candidate 
new datasets (for more complex cultural tasks) without regenerating answers.

Usage:
    python orchestration/analyze_eval_complexity.py runs/eval_n100.jsonl
    python orchestration/analyze_eval_complexity.py runs/eval_n100.jsonl --out runs/complexity.json
	
	Or, if using python3...
	
    python3 orchestration/analyze_eval_complexity.py runs/eval_n100.jsonl
    python3 orchestration/analyze_eval_complexity.py runs/eval_n100.jsonl

Input: 
    JSONL file with records: {query, location, sub_topic, ground_truth: {verified_points}}

Output: 
    JSON dict with complexity metrics (stdout, optionally to file).
    Metrics: question_length, verified_points_{count,length}, topic_diversity, vocab_size
    Plus: per-item breakdown for pattern analysis.

Dependencies: Python stdlib only (json, argparse). Nothing to pip-install.

Gabriel Andrus, 8/28/26.

================================================================================
HOW TO READ RESULTS
================================================================================

PART 1: PER-ITEM ANALYSIS
-------------------------

Format:
    [idx] location        | topic           | Q_len=NN VP_count=NN
          Q: [first 90 chars of question]
          Verified points:
            1. [first verified point, up to 80 chars]
            2. [second verified point]
            3. [third verified point]
            ... and N more

Example:
    [  0] China           | breakfast       | Q_len=18 VP_count= 6
          Q: What is a traditional Chinese breakfast and what does it typically inc
          Verified points:
            1. Traditional Chinese breakfasts often include congee (rice porridge).
            2. Youtiao (fried dough sticks) are a common breakfast item.
            3. Soy milk is frequently served alongside breakfast dishes.
            ... and 3 more

How to read each field:

  • [idx]
    Item number (0-indexed). Use this to track a specific item across runs.

  • location
    The cultural group being asked about (e.g., China, Indonesia, Egypt).
    Left-aligned, padded to 15 characters for readability.

  • topic
    The sub-category of cultural knowledge (e.g., breakfast, ceremony, customs).
    Left-aligned, padded to 15 characters.

  • Q_len
    Question length in WORDS (count of space-separated tokens).
    Right-aligned, 2 digits.
    - Q_len=5-10: SHORT questions, usually simple factual ("What is X?")
    - Q_len=15-25: MEDIUM questions, may require some context ("How is X practiced?")
    - Q_len=30+: LONG questions, likely multi-step reasoning ("Explain why X matters
      in Y context and what it reveals about Z values")

  • VP_count
    Number of VERIFIED POINTS (atomic knowledge units) in the ground truth answer.
    Right-aligned, 2 digits.
    - VP_count=1-3: FEW verified points, simple factual answers
    - VP_count=4-7: MODERATE verified points, mid-level complexity
    - VP_count=8+: MANY verified points, rich multi-faceted answer required

What this tells you:
  - High Q_len + high VP_count = a COMPLEX item (requires reasoning + rich answer)
  - Low Q_len + low VP_count = a SIMPLE item (factual recall only)
  - Mixed (e.g., high Q_len + low VP_count) = possibly poorly-written or ambiguous


PART 2: AGGREGATE COMPLEXITY PROFILE
-------------------------------------

The JSON summary at the end aggregates all 100 items across five dimensions:

  1. "question_length": {min, max, mean}
     The distribution of how many words are in each question.
     
     min:  Shortest question in the set (usually 1-5 words for an outlier)
     max:  Longest question in the set (upper bound on verbosity)
     mean: Average question length across all 100 items

     Interpretation:
       - mean < 15 words: Questions are mostly simple/direct
       - mean 15-30 words: Questions are moderately complex
       - mean > 30 words: Questions are verbose/multi-step

  2. "verified_points_per_item": {min, max, mean}
     The distribution of how many atomic facts compose a correct answer.
     
     min:  Minimum number of verified points in any item (how shallow can an answer be?)
     max:  Maximum number of verified points in any item (how deep can it get?)
     mean: Average verified points per item

     Interpretation:
       - mean < 4: Ground truth is relatively shallow (factual, not reasoning-heavy)
       - mean 4-7: Ground truth is moderate (typical multi-fact answers)
       - mean > 7: Ground truth is rich (complex multi-faceted answers required)

  3. "verified_points_total_words": {min, max, mean}
     The total word count across ALL verified points per item.
     Measures the "volume" of ground truth knowledge per question.
     
     min:  Smallest ground truth word count (how minimal?)
     max:  Largest ground truth word count (how verbose?)
     mean: Average total words in the answer across all 100 items

     Interpretation:
       - mean < 50: Answers are terse (1-2 sentence answers)
       - mean 50-100: Answers are moderate (3-5 sentence answers)
       - mean > 100: Answers are rich (5+ sentence, deeply explained)

  4. "topic_diversity": {unique_topics, topics: [...]}
     How many different cultural sub-domains are represented.
     
     unique_topics: Count of distinct topic values
     topics: List of all unique topics (e.g., ["breakfast", "ceremony", "customs"])

     Interpretation:
       - Low diversity: Set is narrow (all items about food, or all about rituals)
       - High diversity: Set covers many cultural aspects

  5. "location_diversity": {unique_locations, locations: [...]}
     How many different cultural groups are represented.
     
     unique_locations: Count of distinct cultural groups
     locations: List of all unique locations

     Interpretation:
       - Low diversity: Set is focused on few cultures (good for deep study, bad for generalization)
       - High diversity: Set is broad (harder to analyze, but better for cross-cultural claims)

  6. "vocabulary_size": N
     Total number of unique words across ALL verified points.
     Proxy for semantic richness of ground truth.

     Interpretation:
       - vocabulary_size < 300: Limited lexical diversity
       - vocabulary_size 300-600: Moderate diversity
       - vocabulary_size > 600: Rich vocabulary (complex concepts)


DEFINING "SUFFICIENTLY COMPLEX" DATA
-------------------------------------

For this project, we are studying when cultural augmentation helps. Augmentation is
only useful on items where the model struggles, AND where the answer requires
reasoning chains (not just factual recall).

BASELINE THRESHOLD (as of 8/28/26):

An item is "SUFFICIENTLY COMPLEX" if it has BOTH:
  1. verified_points >= 8
  2. question_length >= 30 words

Rationale:
  • verified_points >= 8: Rich ground truth with multiple atomic facts. Augmentation
    has more scaffolding material to work with. Simple factual items (1-3 points)
    don't benefit from path reconstruction.
  • question_length >= 30 words: Implies multi-step reasoning ("explain why...",
    "describe the relationship between...") rather than simple recall ("what is...?").
    Longer questions correlate with deeper cultural reasoning.

Example of SUFFICIENTLY COMPLEX item:
  Q_len=42, VP_count=9: "Explain the role of tea ceremonies in Japanese culture and what values
    they reflect about social hierarchy, spirituality, and hospitality"
  Ground truth: 9 verified points covering preparation, historical origins, social
    significance, spiritual meaning, etc.

Example of NOT SUFFICIENTLY COMPLEX item:
  Q_len=8, VP_count=3: "What is a traditional breakfast?"
  Ground truth: 3 verified points about common breakfast foods.
  Reason: Too simple for augmentation to provide meaningful scaffolding.

ITERATING ON THE THRESHOLD
---------------------------

These thresholds (8 VP, 30 Q_len) are BASELINE, not final. As we experiment and
pivot to new datasets, we will:
  1. Measure performance on sufficiently-complex vs non-complex items separately
  2. Observe where augmentation helps (and hurts)
  3. Adjust thresholds based on empirical findings

For example:
  - If augmentation helps on VP >= 10 but hurts on VP = 8-9, raise threshold to 10
  - If 30-word questions don't show improvement but 40-word do, raise Q_len threshold
  - If we find a secondary signal (e.g., topic diversity or vocab size) matters, add it

This is documented as we go. Update this section when new thresholds are adopted.
"""

from __future__ import annotations
import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("analyze_eval_complexity")


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


def compute_complexity_profile(items: list[dict]) -> dict:
    """Compute aggregate complexity metrics."""
    n = len(items)
    
    q_lengths = [len(i['query'].split()) for i in items]
    vp_counts = [len(i['ground_truth']['verified_points']) for i in items]
    vp_lengths = [
        sum(len(vp.split()) for vp in i['ground_truth']['verified_points'])
        for i in items
    ]
    
    topics = [i['sub_topic'] for i in items]
    locations = [i['location'] for i in items]
    
    all_words = set()
    for item in items:
        for vp in item['ground_truth']['verified_points']:
            all_words.update(vp.lower().split())
    
    return {
        "n_items": n,
        "question_length": {
            "min": min(q_lengths),
            "max": max(q_lengths),
            "mean": sum(q_lengths) / n,
        },
        "verified_points_per_item": {
            "min": min(vp_counts),
            "max": max(vp_counts),
            "mean": sum(vp_counts) / n,
        },
        "verified_points_total_words": {
            "min": min(vp_lengths),
            "max": max(vp_lengths),
            "mean": sum(vp_lengths) / n,
        },
        "topic_diversity": {
            "unique_topics": len(set(topics)),
            "topics": sorted(set(topics)),
        },
        "location_diversity": {
            "unique_locations": len(set(locations)),
            "locations": sorted(set(locations)),
        },
        "vocabulary_size": len(all_words),
    }


def classify_by_complexity(items: list[dict], 
                          min_vp: int = 8, 
                          min_q_len: int = 30) -> dict:
    """
    Separate items into sufficiently complex and non-complex.
    
    Baseline thresholds (as of 8/28/26):
      - min_vp: >= 8 verified points (rich ground truth)
      - min_q_len: >= 30 question words (multi-step reasoning)
    
    Returns dict with complex_items, simple_items, counts, and percentages.
    """
    complex_items = []
    simple_items = []
    
    for item in items:
        vp_count = len(item['ground_truth']['verified_points'])
        q_len = len(item['query'].split())
        
        if vp_count >= min_vp and q_len >= min_q_len:
            complex_items.append(item)
        else:
            simple_items.append(item)
    
    n_total = len(items)
    return {
        "classification_threshold": {
            "min_verified_points": min_vp,
            "min_question_words": min_q_len,
        },
        "complex_items": complex_items,
        "simple_items": simple_items,
        "n_complex": len(complex_items),
        "n_simple": len(simple_items),
        "pct_complex": (len(complex_items) / n_total * 100) if n_total > 0 else 0.0,
        "pct_simple": (len(simple_items) / n_total * 100) if n_total > 0 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl_path", help="Path to eval set JSONL file")
    ap.add_argument("--out", help="Optional output JSON file for aggregate profile")
    ap.add_argument("--min-vp", type=int, default=8,
                    help="Minimum verified points for 'sufficiently complex' (default: 8)")
    ap.add_argument("--min-q-len", type=int, default=30,
                    help="Minimum question word count for 'sufficiently complex' (default: 30)")
    args = ap.parse_args()
    
    items = load_items(args.jsonl_path)
    profile = compute_complexity_profile(items)
    classification = classify_by_complexity(items, min_vp=args.min_vp, min_q_len=args.min_q_len)
    
    # Per-item details
    print("\n" + "="*100)
    print("PER-ITEM ANALYSIS")
    print("="*100)
    for i, item in enumerate(items):
        q_len = len(item['query'].split())
        vp_count = len(item['ground_truth']['verified_points'])
        location = item['location']
        topic = item['sub_topic']
        
        # Flag if sufficiently complex
        is_complex = vp_count >= args.min_vp and q_len >= args.min_q_len
        flag = "[COMPLEX]" if is_complex else "[SIMPLE] "
        
        print(f"\n{flag} [{i:3d}] {location:15} | {topic:15} | Q_len={q_len:2d} VP_count={vp_count:2d}")
        print(f"        Q: {item['query'][:90]}")
        print(f"        Verified points:")
        for j, vp in enumerate(item['ground_truth']['verified_points'][:3]):
            print(f"          {j+1}. {vp[:80]}")
        if len(item['ground_truth']['verified_points']) > 3:
            print(f"          ... and {len(item['ground_truth']['verified_points']) - 3} more")
    
    # Aggregate profile
    print("\n" + "="*100)
    print("AGGREGATE COMPLEXITY PROFILE")
    print("="*100)
    print(json.dumps(profile, indent=2))
    
    # Classification summary
    print("\n" + "="*100)
    print("COMPLEXITY CLASSIFICATION SUMMARY")
    print("="*100)
    print(f"Threshold: verified_points >= {args.min_vp} AND question_words >= {args.min_q_len}")
    print(f"\n  Sufficiently complex: {classification['n_complex']:3d} items ({classification['pct_complex']:5.1f}%)")
    print(f"  Not complex:         {classification['n_simple']:3d} items ({classification['pct_simple']:5.1f}%)")
    
    # Write to file if requested
    if args.out:
        output_data = {
            "aggregate_profile": profile,
            "classification": {
                "threshold": classification["classification_threshold"],
                "n_complex": classification["n_complex"],
                "n_simple": classification["n_simple"],
                "pct_complex": classification["pct_complex"],
                "pct_simple": classification["pct_simple"],
            },
            "complex_items": [
                {
                    "idx": i,
                    "query": item["query"],
                    "location": item["location"],
                    "sub_topic": item["sub_topic"],
                    "verified_points_count": len(item["ground_truth"]["verified_points"]),
                }
                for i, item in enumerate(items)
                if len(item["ground_truth"]["verified_points"]) >= args.min_vp
                and len(item["query"].split()) >= args.min_q_len
            ],
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        log.info("Full analysis written to %s", args.out)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
