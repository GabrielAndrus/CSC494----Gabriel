#!/usr/bin/env python
"""
normalize_normad.py -- Map a NormAd sample onto the PluralTree eval-set schema
(the same {query, location, sub_topic, ground_truth:{location, sub_topic,
verified_points}} shape produced by orchestration/build_eval_set.py for
GlobalCultureQA).

Source: https://huggingface.co/datasets/akhilayerukola/NormAd
NormAd row schema (confirmed via datasets-server /info):
    ID, Country, Background, Axis, Subaxis, Value, Rule-of-Thumb, Story,
    Explanation, Gold Label

NormAd is a social-acceptability-judgment task (a short "Story" ending in a
yes/no question, judged against a single "Rule-of-Thumb"), not an open cultural
QA task with a hand-decomposed multi-point answer like GlobalCultureQA. See
NORMAD_REPORT.md in this folder for the full rationale behind every mapping
decision below and what is lost in translation.

Mapping
-------
    query                        <- Story (verbatim; already a self-contained
                                     scenario + yes/no acceptability question)
    location                     <- Country (title-cased, underscores -> spaces)
    sub_topic                    <- Subaxis (title-cased, underscores -> spaces)
    ground_truth.location        <- same as location
    ground_truth.sub_topic       <- same as sub_topic
    ground_truth.verified_points <- [Rule-of-Thumb (yes/no rows only -- see below),
                                      <derived acceptability statement from Gold Label>,
                                      ...sentences split out of Explanation]

  Rule-of-Thumb is OMITTED from verified_points when Gold Label == "neutral":
  NormAd's neutral rows deliberately pair a story with a Rule-of-Thumb/Background
  from a country that does NOT govern it (that mismatch is the test). Treating a
  non-applicable rule as ground truth would be actively wrong, not just noisy.

Extra, non-schema key `_normad_source` is carried through on every record for
traceability (original Country/Axis/Subaxis/Value/Gold Label/Background/ID).
Per AGENT_CONTRACT.md, extra keys on a record are allowed and ignored by the
orchestrator/agents, so this does not interfere with the pipeline.

Usage
-----
    python normalize_normad.py --in normad_raw_sample.jsonl \
        --out normad_eval_set.jsonl
"""

from __future__ import annotations
import argparse
import json
import re


_LABEL_STATEMENT = {
    "yes": "In {country}, the behavior described is considered socially acceptable "
           "given this norm.",
    "no": "In {country}, the behavior described is considered socially unacceptable "
          "(it violates this norm).",
    "neutral": "In {country}, whether this behavior is acceptable is ambiguous / "
               "context-dependent -- NormAd's annotators did not reach a clear "
               "yes-or-no consensus for this case.",
}


def _humanize(slug: str) -> str:
    return slug.replace("_", " ").strip().title()


def _split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter for the short (1-3 sentence) Explanation field."""
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def to_eval_record(row: dict) -> dict | None:
    country_raw = str(row.get("Country", "")).strip()
    story = str(row.get("Story", "")).strip()
    if not (country_raw and story):
        return None

    location = _humanize(country_raw)
    sub_topic = _humanize(str(row.get("Subaxis", "")).strip()) or "General Etiquette"

    rule_of_thumb = str(row.get("Rule-of-Thumb", "")).strip()
    label = str(row.get("Gold Label", "")).strip().lower()
    explanation_sentences = _split_sentences(row.get("Explanation", ""))

    verified_points: list[str] = []
    # NormAd's "neutral" rows are deliberately mismatched: the Rule-of-Thumb /
    # Background attached to the row is a DISTRACTOR drawn from a country whose
    # norm does not actually govern the story (that mismatch is the whole point
    # of the neutral class -- see NORMAD_REPORT.md, "Nuances lost"). Including
    # that rule as a "verified point" would hand Agent B a wrong fact to grade
    # candidate answers against, so we only do it for yes/no rows, where the
    # rule genuinely does apply to the story.
    if rule_of_thumb and label != "neutral":
        verified_points.append(rule_of_thumb)

    label_stmt_template = _LABEL_STATEMENT.get(label)
    if label_stmt_template:
        verified_points.append(label_stmt_template.format(country=location))

    # Only add explanation sentences that aren't near-duplicates of the
    # rule-of-thumb we already added (Explanation often restates it verbatim
    # before adding the story-specific justification).
    rot_lower = rule_of_thumb.lower()
    for sent in explanation_sentences:
        if sent.lower() != rot_lower and sent not in verified_points:
            verified_points.append(sent)

    if not verified_points:
        return None

    return {
        "query": story,
        "location": location,
        "sub_topic": sub_topic,
        "ground_truth": {
            "location": location,
            "sub_topic": sub_topic,
            "verified_points": verified_points,
        },
        "_normad_source": {
            "id": row.get("ID"),
            "country_raw": country_raw,
            "axis": row.get("Axis"),
            "subaxis": row.get("Subaxis"),
            "value": row.get("Value"),
            "gold_label": row.get("Gold Label"),
            "background": row.get("Background"),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Raw NormAd JSONL")
    ap.add_argument("--out", "-o", default="normad_eval_set.jsonl")
    args = ap.parse_args()

    records = []
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rec = to_eval_record(row)
            if rec:
                records.append(rec)

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} normalized records to {args.out}")
    labels = [r["_normad_source"]["gold_label"] for r in records]
    print("Gold label distribution:",
          {l: labels.count(l) for l in sorted(set(labels))})
    print("Locations:", sorted({r["location"] for r in records}))


if __name__ == "__main__":
    main()
