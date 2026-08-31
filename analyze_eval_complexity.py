"""
analyze_eval_complexity.py -- Analyze the complexity of evaluation sets.

Analyzes the n=100 PluralTree validation set from CulFiT's GlobalCultureQA.csv. 

No model calls. Measures: question length, verified-points count/length, topic 
diversity, vocab richness, AND reasoning structure (causal chains, answer depth,
information density, semantic richness).

Usage:
    python orchestration/analyze_eval_complexity.py runs/eval_n100.jsonl
    python orchestration/analyze_eval_complexity.py runs/eval_n100.jsonl --out runs/complexity.json


Input: 
    JSONL file with records: {query, location, sub_topic, ground_truth: {verified_points}}

Output: 
    JSON dict with complexity metrics (stdout, optionally to file).
    Metrics: question_length, verified_points_{count,length}, topic_diversity, vocab_size
    PLUS: causal_density, answer_depth, information_density, semantic_richness
    Plus: per-item breakdown for pattern analysis.

Dependencies: Python stdlib only (json, argparse, re). Nothing to pip-install.

Gabriel Andrus, 8/28/26. Extended 8/31/26 with reasoning structure metrics.

================================================================================
HOW TO READ RESULTS
================================================================================

PART 1: PER-ITEM ANALYSIS
-------------------------

Format:
    [idx] location        | topic           | Q_len=NN VP_count=NN | Reasoning Score
          Q: [first 90 chars of question]
          Verified points:
            1. [first verified point, up to 80 chars]
            2. [second verified point]
            3. [third verified point]
            ... and N more

Example:
    [COMPLEX] [  0] China           | breakfast       | Q_len=18 VP_count= 6 | R_score=0.45
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

  • R_score (Reasoning Score)
    Combined metric (0.0-1.0) measuring answer reasoning depth:
      - causal_density (0.0-1.0): Fraction of VPs containing causal words ("because",
        "due to", "causes", "results in", "therefore", "thus", "hence", "so that", etc.)
        High = answers explain WHY, not just WHAT.
      - answer_depth (0.0-1.0): Ratio of avg VP length to max expected length (50 words).
        High = each VP is detailed, not terse.
      - information_density (0.0-1.0): Unique words per VP (normalized to 0-1).
        High = VPs add new information, not repetitive.
    R_score = (causal_density + answer_depth + information_density) / 3
    
    Interpretation:
      - R_score > 0.6: Strong reasoning structure (explains WHY, detailed, diverse)
      - R_score 0.4-0.6: Moderate reasoning (some causal links, moderate depth)
      - R_score < 0.4: Weak reasoning (mostly facts, shallow, repetitive)

What this tells you:
  - High Q_len + high VP_count + high R_score = DEEPLY COMPLEX (reasoning + rich answer)
  - Low Q_len + low VP_count + low R_score = SIMPLE (fact recall only)
  - High VP_count + low R_score = FACT-HEAVY but shallow (many independent facts, no reasoning)


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

  4. "causal_density": {min, max, mean, percentiles}
     Fraction of verified points containing causal/reasoning markers.
     Causal words: "because", "due to", "causes", "results in", "therefore", 
                   "thus", "hence", "so that", "which", "enable", "allow", 
                   "reflect", "reveal", "demonstrate", "explain", "reason"
     
     min:  Lowest causal density in any item (pure fact-list?)
     max:  Highest causal density in any item (heavily reasoned?)
     mean: Average causal density across all items
     percentiles: 25th, 50th (median), 75th

     Interpretation:
       - mean < 0.3: Answers are mostly fact-lists, little reasoning
       - mean 0.3-0.6: Moderate causal structure (some WHY explanations)
       - mean > 0.6: Strong reasoning (most VPs explain WHY or HOW)

  5. "answer_depth": {min, max, mean, percentiles}
     Average words per verified point (normalized to 0-1 scale, max 50).
     Depth = (avg_words_per_vp / 50), capped at 1.0
     
     min:  Shallowest VP (minimum words per fact)
     max:  Deepest VP (maximum words per fact)
     mean: Average depth across all items

     Interpretation:
       - mean < 0.4: VPs are terse (< 20 words each)
       - mean 0.4-0.7: VPs are moderate (20-35 words each)
       - mean > 0.7: VPs are detailed (> 35 words each)

  6. "information_density": {min, max, mean, percentiles}
     Unique words per verified point (as fraction of total words in VP).
     density = unique_words / total_words
     Higher = each VP adds new information (not repetitive).
     
     min:  Lowest density (most repetitive/short VPs)
     max:  Highest density (most novel/long VPs)
     mean: Average density across all items

     Interpretation:
       - mean < 0.6: VPs repeat concepts, lower novelty
       - mean 0.6-0.75: Moderate information density
       - mean > 0.75: High novelty (each VP brings new vocabulary)

  7. "topic_diversity": {unique_topics, topics: [...]}
     How many different cultural sub-domains are represented.
     
     unique_topics: Count of distinct topic values
     topics: List of all unique topics (e.g., ["breakfast", "ceremony", "customs"])

     Interpretation:
       - Low diversity: Set is narrow (all items about food, or all about rituals)
       - High diversity: Set covers many cultural aspects

  8. "location_diversity": {unique_locations, locations: [...]}
     How many different cultural groups are represented.
     
     unique_locations: Count of distinct cultural groups
     locations: List of all unique locations

     Interpretation:
       - Low diversity: Set is focused on few cultures (good for deep study, bad for generalization)
       - High diversity: Set is broad (harder to analyze, but better for cross-cultural claims)

  9. "vocabulary_size": N
     Total number of unique words across ALL verified points.
     Proxy for semantic richness of ground truth.

     Interpretation:
       - vocabulary_size < 300: Limited lexical diversity
       - vocabulary_size 300-600: Moderate diversity
       - vocabulary_size > 600: Rich vocabulary (complex concepts)

  10. "reasoning_score": {min, max, mean, percentiles}
      Combined metric (0.0-1.0) = (causal_density + answer_depth + information_density) / 3
      Captures overall reasoning complexity in a single number.
      
      min:  Item with weakest reasoning structure
      max:  Item with strongest reasoning structure
      mean: Average across all items
      percentiles: 25th, 50th (median), 75th

      Interpretation:
        - mean < 0.4: Eval set is mostly fact-lists, limited reasoning scaffolding
        - mean 0.4-0.6: Eval set has moderate reasoning (suitable for baseline)
        - mean > 0.6: Eval set is reasoning-heavy (complex cultural knowledge required)

      Use this to COMPARE DATASETS: 
        - GlobalCultureQA baseline: reasoning_score mean = X
        - Candidate dataset A: reasoning_score mean = Y
        - If Y > X by >0.15, dataset A is significantly harder on reasoning.


DEFINING "SUFFICIENTLY COMPLEX" DATA
-------------------------------------

For this project, we are studying when cultural augmentation helps. Augmentation is
only useful on items where the model struggles, AND where the answer requires
reasoning chains (not just factual recall).

BASELINE THRESHOLD (as of 8/31/26):

An item is "SUFFICIENTLY COMPLEX" if it has ALL THREE:
  1. verified_points >= 8
  2. question_length >= 30 words
  3. reasoning_score >= 0.45  (NEW: ensure it's not just fact-heavy)

Rationale:
  • verified_points >= 8: Rich ground truth with multiple atomic facts. Augmentation
    has more scaffolding material to work with. Simple factual items (1-3 points)
    don't benefit from path reconstruction.
  • question_length >= 30 words: Implies multi-step reasoning ("explain why...",
    "describe the relationship between...") rather than simple recall ("what is...?").
    Longer questions correlate with deeper cultural reasoning.
  • reasoning_score >= 0.45 (NEW): Filters out "fact-list" items that have many VPs
    but no causal structure (e.g., "what are 8 ingredients in X dish?" has VP_count=8
    but causal_density=0). Items with reasoning structure benefit MORE from augmentation
    because the model can scaffold chains of thought, not just recall facts.

Example of SUFFICIENTLY COMPLEX item:
  Q_len=42, VP_count=9, R_score=0.68: "Explain the role of tea ceremonies in Japanese culture and what values
    they reflect about social hierarchy, spirituality, and hospitality"
  Ground truth:
    1. Tea ceremonies reflect Japanese values of harmony, respect, and mindfulness.
    2. The practice emphasizes aesthetic beauty and simplicity (wabi-sabi).
    3. Historically, ceremonies served as a way to forge relationships among warriors.
    ... (6 more VPs with causal language like "because", "thus", "this enables")
  Reasoning: High causal density, detailed VPs, high information density. Augmentation
    can scaffold: "The ceremony reflects X value → historically used for Y purpose →
    modern significance is Z because of cultural continuity."

Example of NOT SUFFICIENTLY COMPLEX item (trap 1: fact-list with many VPs):
  Q_len=15, VP_count=8, R_score=0.22: "What are the ingredients in a traditional breakfast?"
  Ground truth:
    1. Congee (rice porridge) is common.
    2. Youtiao (fried dough) is common.
    3. Soy milk is common.
    ... (5 more items, each a single ingredient with no explanation)
  Reason: VP_count=8 looks rich, but causal_density=0 (no "because", "thus", etc.).
    This is a fact-list, not reasoning. Model just needs to memorize ingredients.
    Augmentation won't help because there's no reasoning chain to scaffold.

Example of NOT SUFFICIENTLY COMPLEX item (trap 2: too simple):
  Q_len=8, VP_count=3, R_score=0.35: "What is a traditional breakfast?"
  Ground truth: 3 verified points about common breakfast foods.
  Reason: Too shallow on all three dimensions.

ITERATING ON THE THRESHOLD
---------------------------

The thresholds (8 VP, 30 Q_len, 0.45 R_score) are BASELINE, not final. As we
experiment and pivot to new datasets, we will:
  1. Measure performance on sufficiently-complex vs non-complex items separately
  2. Observe where augmentation helps (and hurts)
  3. Adjust thresholds based on empirical findings

For example:
  - If augmentation helps on R_score >= 0.55 but hurts on 0.45-0.54, raise to 0.55
  - If we find causal_density matters more than depth, weight it higher
  - If we discover secondary signals (e.g., information_density > 0.8), add them

This is documented as we go. Update this section when new thresholds are adopted.
"""

from __future__ import annotations
import argparse
import json
import logging
import re
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("analyze_eval_complexity")


# Causal and reasoning keywords (expanded set)
CAUSAL_WORDS = {
    "because", "due to", "caused by", "causes", "caused", "causing",
    "reason", "reasons", "result", "results", "resulted", "resulting",
    "therefore", "thus", "hence", "so that", "as a result", "consequently",
    "enable", "enables", "enabled", "allowing", "allow",
    "lead to", "leads", "led",
    "reflect", "reflects", "reveal", "reveals", "demonstrate", "demonstrates",
    "explain", "explains", "explanation", "reason for",
    "which", "that",  # relative pronouns connecting ideas
    "thereby", "whereupon",
}


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


def compute_causal_density(vp_list: list[str]) -> float:
    """
    Compute fraction of verified points containing causal/reasoning language.
    
    Returns: float in [0.0, 1.0]
      0.0 = no VPs have causal words (pure fact-list)
      1.0 = all VPs have causal words (strongly reasoned)
    """
    if not vp_list:
        return 0.0
    
    causal_count = 0
    for vp in vp_list:
        words = vp.lower().split()
        if any(word.rstrip('.,;:') in CAUSAL_WORDS for word in words):
            causal_count += 1
    
    return causal_count / len(vp_list)


def compute_answer_depth(vp_list: list[str]) -> float:
    """
    Compute average words per verified point, normalized to 0-1.
    
    depth = (avg_words_per_vp / 50), capped at 1.0
    
    Returns: float in [0.0, 1.0]
      0.0 = VPs are very short (< 5 words avg)
      1.0 = VPs are detailed (50+ words avg)
    """
    if not vp_list:
        return 0.0
    
    total_words = sum(len(vp.split()) for vp in vp_list)
    avg_words = total_words / len(vp_list)
    
    # Normalize to 50 words as the "deep" threshold
    return min(avg_words / 50.0, 1.0)


def compute_information_density(vp_list: list[str]) -> float:
    """
    Compute average information density across VPs.
    
    For each VP: density = unique_words / total_words
    High density = each VP introduces new concepts (not repetitive).
    
    Returns: float in [0.0, 1.0]
      0.0 = all VPs are single words or highly repetitive
      1.0 = all VPs have completely unique vocabulary
    """
    if not vp_list:
        return 0.0
    
    densities = []
    for vp in vp_list:
        words = vp.lower().split()
        if not words:
            densities.append(0.0)
        else:
            unique_words = len(set(words))
            density = unique_words / len(words)
            densities.append(density)
    
    return sum(densities) / len(densities) if densities else 0.0


def compute_reasoning_score(vp_list: list[str]) -> float:
    """
    Combined reasoning complexity score (0.0-1.0).
    
    score = (causal_density + answer_depth + information_density) / 3
    
    Returns: float in [0.0, 1.0]
      < 0.4 = weak reasoning (mostly facts, shallow)
      0.4-0.6 = moderate reasoning (some causality, moderate depth)
      > 0.6 = strong reasoning (explains WHY, detailed, novel)
    """
    causal = compute_causal_density(vp_list)
    depth = compute_answer_depth(vp_list)
    density = compute_information_density(vp_list)
    
    return (causal + depth + density) / 3.0


def percentiles(values: list[float]) -> dict:
    """Compute 25th, 50th (median), 75th percentiles."""
    if not values:
        return {"p25": 0.0, "p50": 0.0, "p75": 0.0}
    
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    
    p25_idx = int(n * 0.25)
    p50_idx = int(n * 0.50)
    p75_idx = int(n * 0.75)
    
    return {
        "p25": sorted_vals[p25_idx],
        "p50": sorted_vals[p50_idx],
        "p75": sorted_vals[p75_idx],
    }


def compute_complexity_profile(items: list[dict]) -> dict:
    """Compute aggregate complexity metrics including reasoning structure."""
    n = len(items)
    
    q_lengths = [len(i['query'].split()) for i in items]
    vp_counts = [len(i['ground_truth']['verified_points']) for i in items]
    vp_lengths = [
        sum(len(vp.split()) for vp in i['ground_truth']['verified_points'])
        for i in items
    ]
    
    # NEW: Reasoning metrics
    causal_densities = [
        compute_causal_density(i['ground_truth']['verified_points'])
        for i in items
    ]
    answer_depths = [
        compute_answer_depth(i['ground_truth']['verified_points'])
        for i in items
    ]
    info_densities = [
        compute_information_density(i['ground_truth']['verified_points'])
        for i in items
    ]
    reasoning_scores = [
        compute_reasoning_score(i['ground_truth']['verified_points'])
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
        "causal_density": {
            "min": min(causal_densities),
            "max": max(causal_densities),
            "mean": sum(causal_densities) / n,
            **percentiles(causal_densities),
        },
        "answer_depth": {
            "min": min(answer_depths),
            "max": max(answer_depths),
            "mean": sum(answer_depths) / n,
            **percentiles(answer_depths),
        },
        "information_density": {
            "min": min(info_densities),
            "max": max(info_densities),
            "mean": sum(info_densities) / n,
            **percentiles(info_densities),
        },
        "reasoning_score": {
            "min": min(reasoning_scores),
            "max": max(reasoning_scores),
            "mean": sum(reasoning_scores) / n,
            **percentiles(reasoning_scores),
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
                          min_q_len: int = 30,
                          min_reasoning_score: float = 0.45) -> dict:
    """
    Separate items into sufficiently complex and non-complex.
    
    Baseline thresholds (as of 8/31/26):
      - min_vp: >= 8 verified points (rich ground truth)
      - min_q_len: >= 30 question words (multi-step reasoning)
      - min_reasoning_score: >= 0.45 (ensures reasoning structure, not just facts)
    
    Returns dict with complex_items, simple_items, counts, and percentages.
    """
    complex_items = []
    simple_items = []
    
    for item in items:
        vp_count = len(item['ground_truth']['verified_points'])
        q_len = len(item['query'].split())
        r_score = compute_reasoning_score(item['ground_truth']['verified_points'])
        
        if vp_count >= min_vp and q_len >= min_q_len and r_score >= min_reasoning_score:
            complex_items.append(item)
        else:
            simple_items.append(item)
    
    n_total = len(items)
    return {
        "classification_threshold": {
            "min_verified_points": min_vp,
            "min_question_words": min_q_len,
            "min_reasoning_score": min_reasoning_score,
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
    ap.add_argument("--min-reasoning-score", type=float, default=0.45,
                    help="Minimum reasoning score for 'sufficiently complex' (default: 0.45)")
    args = ap.parse_args()
    
    items = load_items(args.jsonl_path)
    profile = compute_complexity_profile(items)
    classification = classify_by_complexity(items, 
                                           min_vp=args.min_vp, 
                                           min_q_len=args.min_q_len,
                                           min_reasoning_score=args.min_reasoning_score)
    
    # Per-item details with reasoning scores
    print("\n" + "="*120)
    print("PER-ITEM ANALYSIS")
    print("="*120)
    for i, item in enumerate(items):
        q_len = len(item['query'].split())
        vp_count = len(item['ground_truth']['verified_points'])
        r_score = compute_reasoning_score(item['ground_truth']['verified_points'])
        location = item['location']
        topic = item['sub_topic']
        
        # Flag if sufficiently complex
        is_complex = (vp_count >= args.min_vp and 
                     q_len >= args.min_q_len and 
                     r_score >= args.min_reasoning_score)
        flag = "[COMPLEX]" if is_complex else "[SIMPLE] "
        
        print(f"\n{flag} [{i:3d}] {location:15} | {topic:15} | Q_len={q_len:2d} VP_count={vp_count:2d} | R_score={r_score:.2f}")
        print(f"        Q: {item['query'][:90]}")
        print(f"        Verified points:")
        for j, vp in enumerate(item['ground_truth']['verified_points'][:3]):
            print(f"          {j+1}. {vp[:80]}")
        if len(item['ground_truth']['verified_points']) > 3:
            print(f"          ... and {len(item['ground_truth']['verified_points']) - 3} more")
    
    # Aggregate profile
    print("\n" + "="*120)
    print("AGGREGATE COMPLEXITY PROFILE")
    print("="*120)
    print(json.dumps(profile, indent=2))
    
    # Classification summary
    print("\n" + "="*120)
    print("COMPLEXITY CLASSIFICATION SUMMARY")
    print("="*120)
    print(f"Threshold: verified_points >= {args.min_vp} AND question_words >= {args.min_q_len} AND reasoning_score >= {args.min_reasoning_score}")
    print(f"\n  Sufficiently complex: {classification['n_complex']:3d} items ({classification['pct_complex']:5.1f}%)")
    print(f"  Not complex:         {classification['n_simple']:3d} items ({classification['pct_simple']:5.1f}%)")
    
    # Reasoning breakdown
    print("\n" + "-"*120)
    print("REASONING STRUCTURE BREAKDOWN")
    print("-"*120)
    print(f"  Causal density:       mean={profile['causal_density']['mean']:.3f} (fraction of VPs with 'because', 'thus', etc.)")
    print(f"  Answer depth:         mean={profile['answer_depth']['mean']:.3f} (avg words per VP, normalized to 50)")
    print(f"  Information density:  mean={profile['information_density']['mean']:.3f} (unique words per VP)")
    print(f"  Reasoning score:      mean={profile['reasoning_score']['mean']:.3f} (combined metric, 0-1)")
    
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
                    "reasoning_score": compute_reasoning_score(item["ground_truth"]["verified_points"]),
                }
                for i, item in enumerate(items)
                if (len(item["ground_truth"]["verified_points"]) >= args.min_vp and
                    len(item["query"].split()) >= args.min_q_len and
                    compute_reasoning_score(item["ground_truth"]["verified_points"]) >= args.min_reasoning_score)
            ],
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        log.info("Full analysis written to %s", args.out)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
