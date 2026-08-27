"""
task_eval.py — CulFiT GlobalCultureQA *task-level* scorer.

WHY THIS EXISTS
---------------
Everything currently in the pipeline (agent_b_engine.precision_score, kp_recall)
scores how well Agent A's *reasoning paths* agree with the ground truth, i.e. it
measures inter-agent behaviour. The professor's repeated point across both
meetings is that this is not the task. GlobalCultureQA is an OPEN-ENDED QA task:
read a cultural question, WRITE a free-text answer, and score that answer against
the golden answer's atomic knowledge units. The only way to know whether the
agentic augmentation helps is to run THIS task with THIS metric and compare.

This module reproduces CulFiT's own metric exactly (paper §3.4.1, eval prompt
§7.9):

  1. Decompose a model answer into atomic knowledge units (one LLM call).
     The golden answer's units are already provided as `verified_points`
     (these ARE grounded_answer_knowledge_points.knowledge_points).
  2. Cultural Precision  S_p = (# answer units that match some golden unit) / m
  3. Cultural Recall     S_r = (# golden units matched by some answer unit) / n
  4. Cultural F1         = 2 * S_p * S_r / (S_p + S_r)
  where "match" is a Yes/No LLM judgement using CulFiT's verbatim eval prompt.

Reference numbers to beat (paper Table 1, Llama-3.1-8B):
  bare Llama-3.1 : P 62.52 / R 68.96 / F1 64.53
  CulFiT(Llama)  : P 74.73 / R 71.21 / F1 72.94

The scorer is answer-source-agnostic: give it (question, answer, verified_points)
and it returns P/R/F1 with a full per-unit trace. So the SAME scorer grades the
CulFiT baseline answer, the agentic-system answer, and (later) a CCKG answer.

PROMPT PROVENANCE
-----------------
This module does NOT invent its own prompts. It loads CulFiT's own strings from
CulFiT/utils/prompt_utils.py via bootstrap (the same file agent_b_engine.py
pulls from), so the answer generation, knowledge-unit extraction, and Yes/No
judging are byte-identical to CulFiT's protocol and the scores are directly
comparable to their reported numbers:
  - ANSWER_GENERATION_SYS_PROMPT / ANSER_GENERATION_USER_PROMT  (baseline answer)
  - ANSWER_EXTRACT_PROMPT_SYS / ANSWER_EXTRACT_PROMPT_USER      (answer -> units)
  - EVAL_CULTURAL_POINTS_PROMPT                                 (Yes/No judge)
Their src/ scripts (answer_extract.py, answer_generation.py, critique_generation.py)
are batch file-in/file-out data-synthesis jobs, not importable evaluators, and
they do `from utils.X import ...` which reawakens the utils/ import collision
bootstrap exists to prevent. So we reuse their PROMPTS (the stable contract) and
supply the thin in-memory evaluator harness their repo doesn't ship. If bootstrap
is unavailable (offline scratch dir), local paraphrase fallbacks keep dev working
but log a warning — a real run must use the bootstrapped originals.

No upstream CulFiT files are modified. The judge reuses the same backend.chat
contract as the rest of the system and the same robust yes/no parser.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Load CulFiT's OWN prompt strings so scoring matches their reported protocol   #
# exactly. These are the same constants agent_b_engine.py pulls; bootstrap      #
# registers CulFiT/utils/prompt_utils.py by absolute path to dodge the utils/   #
# import collision. If bootstrap isn't importable (e.g. running task_eval.py    #
# standalone in a scratch dir), we fall back to local paraphrases so offline    #
# development still works — but a real Narval run MUST use the bootstrapped      #
# originals for comparability, so we log loudly when the fallback is used.      #
# --------------------------------------------------------------------------- #
_USING_CULFIT_PROMPTS = False
try:
    from orchestration import bootstrap  # type: ignore
    _pu = bootstrap.install()["culfit_prompt_utils"]
    ANSWER_EXTRACT_PROMPT_SYS = _pu.ANSWER_EXTRACT_PROMPT_SYS
    ANSWER_EXTRACT_PROMPT_USER = _pu.ANSWER_EXTRACT_PROMPT_USER
    ANSWER_GENERATION_SYS_PROMPT = _pu.ANSWER_GENERATION_SYS_PROMPT
    ANSER_GENERATION_USER_PROMT = _pu.ANSER_GENERATION_USER_PROMT  # sic: CulFiT's spelling
    EVAL_CULTURAL_POINTS_PROMPT = _pu.EVAL_CULTURAL_POINTS_PROMPT
    _USING_CULFIT_PROMPTS = True
    logger.info("task_eval: using CulFiT's own prompt strings (bootstrapped).")
except Exception as e:  # noqa: BLE001 — degrade gracefully for offline dev
    logger.warning(
        "task_eval: could NOT load CulFiT prompts via bootstrap (%s); using "
        "local paraphrase fallbacks. Scores will NOT be directly comparable to "
        "CulFiT's reported numbers. Fix before the real run.", e)
    ANSWER_EXTRACT_PROMPT_SYS = (
        "You are a helpful assistant for a cultural knowledge question answering "
        "scenario. Extract the meaningful cultural knowledge points from the "
        "answer. A knowledge point is an atomic, self-contained single sentence. "
        'Return JSON: {"knowledge_points": ["point1", "point2", ...]}. '
        "Extract in the language of the answer.")
    ANSWER_EXTRACT_PROMPT_USER = "input:\n{}\n\nYour extracted knowledge points:\n"
    ANSWER_GENERATION_SYS_PROMPT = (
        "You are a helpful consultant for a cultural knowledge question answering "
        "scenario. Generate a culturally-aware answer. Return JSON: "
        '{"answer": "", "cultural_group": "", "language": "", "topic": ""}')
    ANSER_GENERATION_USER_PROMT = (
        "You are a helpful consultant for a cultural knowledge question answering "
        "scenario. The question is as follows:\n\n{}\n\n"
        "You should only return the json object above.\n\nYour Answer:")
    EVAL_CULTURAL_POINTS_PROMPT = (
        "You are an expert evaluator for a cultural knowledge question answering "
        "system. You are given a piece of cultural knowledge point and a list of "
        "reference cultural knowledge. Evaluate whether the given point satisfies "
        "one of the reference points. First output 'Yes' or 'No', then a concise "
        "explanation.\n\ncultural knowledge points:\n{}\n\n"
        "reference cultural knowledge points:\n{}\n\nYour output:\n")


# --------------------------------------------------------------------------- #
# Robust Yes/No parsing (mirrors agent_b_engine._verdict_is_yes so the two     #
# judges agree on what a "Yes" is).                                            #
# --------------------------------------------------------------------------- #
def verdict_is_yes(res: str) -> bool:
    if not res:
        return False
    s = res.strip().lstrip("*_#>-• ").strip()
    head = re.split(r"[\s,.:;!]+", s, maxsplit=1)[0].lower() if s else ""
    return head in ("yes", "y", "true", "correct", "aligned")


def _parse_knowledge_points(raw: str) -> Optional[list[str]]:
    """
    Parse CulFiT's decomposition output. Their ANSWER_EXTRACT prompt asks for
    {"knowledge_points": ["...", ...]}. Be tolerant: accept that object, a bare
    JSON array, or an embedded JSON object/array inside surrounding prose.
    """
    if not raw:
        return None

    def _extract(val):
        if isinstance(val, dict) and "knowledge_points" in val:
            kps = val["knowledge_points"]
            if isinstance(kps, list):
                return [str(x).strip() for x in kps if str(x).strip()]
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
        return None

    # Direct parse first.
    try:
        got = _extract(json.loads(raw))
        if got is not None:
            return got
    except json.JSONDecodeError:
        pass
    # Embedded object {"knowledge_points": [...]}.
    m = re.search(r'\{.*"knowledge_points".*\}', raw, re.DOTALL)
    if m:
        try:
            got = _extract(json.loads(m.group(0)))
            if got is not None:
                return got
        except json.JSONDecodeError:
            pass
    # Embedded bare array.
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            got = _extract(json.loads(m.group(0)))
            if got is not None:
                return got
        except json.JSONDecodeError:
            pass
    return None


def _parse_answer_field(raw: str) -> str:
    """
    CulFiT's ANSWER_GENERATION prompt asks for
    {"answer","cultural_group","language","topic"}. Pull the 'answer' field;
    if the model returned plain prose instead, use it verbatim.
    """
    if not raw:
        return ""
    try:
        val = json.loads(raw)
        if isinstance(val, dict) and "answer" in val:
            return str(val["answer"]).strip()
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{.*"answer".*\}', raw, re.DOTALL)
    if m:
        try:
            val = json.loads(m.group(0))
            if isinstance(val, dict) and "answer" in val:
                return str(val["answer"]).strip()
        except json.JSONDecodeError:
            pass
    return raw.strip()


def _parse_json_list(raw: str) -> Optional[list[str]]:
    """Best-effort extraction of a JSON string array from a model response."""
    if not raw:
        return None
    try:
        val = json.loads(raw)
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            val = json.loads(m.group(0))
            if isinstance(val, list):
                return [str(x).strip() for x in val if str(x).strip()]
        except json.JSONDecodeError:
            pass
    return None


def _fallback_sentence_split(answer: str) -> list[str]:
    """Deterministic fallback if the decomposition call doesn't return JSON."""
    parts = re.split(r"(?<=[.!?])\s+", (answer or "").strip())
    return [p.strip() for p in parts if len(p.strip()) > 15]


class TaskEvaluator:
    """
    Scores a free-text answer against golden knowledge units using CulFiT's
    cultural P/R/F1. Backend is anything with .chat(messages, temperature)->str
    (the project's LLMBackend, mock or real).
    """

    def __init__(self, backend, judge_temperature: float = 0.0):
        self.backend = backend
        self.judge_temperature = judge_temperature

    # --- answer generation for the baseline ------------------------------- #
    def generate_answer(self, question: str, group: str = "", topic: str = "",
                        temperature: float = 0.7) -> str:
        """
        CulFiT-baseline answer using their own ANSWER_GENERATION prompts. Their
        SYS prompt asks for a JSON object {"answer","cultural_group","language",
        "topic"} and the USER prompt takes the question in one positional slot.
        We return only the 'answer' field, which is what the task scores.
        """
        messages = [
            {"role": "system", "content": ANSWER_GENERATION_SYS_PROMPT},
            {"role": "user", "content": ANSER_GENERATION_USER_PROMT.format(question)},
        ]
        raw = self.backend.chat(messages, temperature=temperature) or ""
        return _parse_answer_field(raw)

    # --- answer -> atomic units ------------------------------------------- #
    def decompose(self, answer: str, group: str = "", question: str = "") -> list[str]:
        """
        Decompose an answer into atomic knowledge points using CulFiT's own
        ANSWER_EXTRACT prompts. Their USER prompt takes a JSON blob
        {"question","answer"} in one positional slot and expects
        {"knowledge_points": [...]} back. The question improves extraction, so we
        thread it through when available.
        """
        payload = json.dumps({"question": question, "answer": answer}, indent=4)
        messages = [
            {"role": "system", "content": ANSWER_EXTRACT_PROMPT_SYS},
            {"role": "user", "content": ANSWER_EXTRACT_PROMPT_USER.format(payload)},
        ]
        raw = self.backend.chat(messages, temperature=0.0) or ""
        units = _parse_knowledge_points(raw)
        if not units:
            logger.warning("decompose(): unparseable response; using sentence split.")
            units = _fallback_sentence_split(answer)
        return [u for u in units if str(u).strip()]

    # --- single unit vs reference list (one judge call) ------------------- #
    def _unit_matches_reference(self, unit: str, reference: list[str]) -> bool:
        """
        One judge call using CulFiT's verbatim EVAL_CULTURAL_POINTS_PROMPT. Their
        prompt has two positional slots: the candidate point, then the reference
        list. The reference is passed as a JSON array (matching how their eval
        renders the list), preserving non-ASCII for multilingual points.
        """
        prompt = EVAL_CULTURAL_POINTS_PROMPT.format(
            unit, json.dumps(reference, ensure_ascii=False))
        res = self.backend.chat(
            [{"role": "user", "content": prompt}],
            temperature=self.judge_temperature) or ""
        return verdict_is_yes(res)

    # --- full P/R/F1 over one (answer, golden) pair ----------------------- #
    def score_answer(self, answer: str, verified_points: list[str],
                     group: str = "Unknown", question: str = "",
                     answer_units: Optional[list[str]] = None) -> dict:
        """
        Cultural Precision / Recall / F1 for a single answer.

        Precision loop: each ANSWER unit is judged against the golden list.
        Recall loop:    each GOLDEN unit is judged against the answer-unit list.
        Both directions use the same Yes/No judge, matching CulFiT Eq. 8-12.

        Returns a dict with the three scores plus a full trace for inspection.
        """
        golden = [p for p in (verified_points or []) if str(p).strip()]
        if answer_units is None:
            answer_units = self.decompose(answer, group=group, question=question)
        answer_units = [u for u in answer_units if str(u).strip()]

        m, n = len(answer_units), len(golden)
        if m == 0 or n == 0:
            # Vacuous — report explicitly rather than silently scoring 1.0.
            return {
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_answer_units": m, "n_golden_units": n,
                "answer_units": answer_units, "golden_units": golden,
                "precision_hits": [], "recall_hits": [],
                "note": "empty answer units or empty golden units",
            }

        # Precision: which answer units are supported by the golden set?
        precision_hits = [
            self._unit_matches_reference(u, golden) for u in answer_units]
        s_p = sum(precision_hits) / m

        # Recall: which golden units are covered by the answer units?
        # (judge each golden unit against the ANSWER-unit list, symmetric to CulFiT)
        recall_hits = [
            self._unit_matches_reference(g, answer_units) for g in golden]
        s_r = sum(recall_hits) / n

        f1 = (2 * s_p * s_r / (s_p + s_r)) if (s_p + s_r) > 0 else 0.0

        return {
            "precision": s_p, "recall": s_r, "f1": f1,
            "n_answer_units": m, "n_golden_units": n,
            "answer_units": answer_units, "golden_units": golden,
            "precision_hits": [bool(x) for x in precision_hits],
            "recall_hits": [bool(x) for x in recall_hits],
            "covered_golden": [g for g, h in zip(golden, recall_hits) if h],
            "missed_golden": [g for g, h in zip(golden, recall_hits) if not h],
        }


# --------------------------------------------------------------------------- #
# Answer providers: each turns one eval item into an answer string. The scorer #
# is identical across systems; only the provider changes. This is exactly the  #
# "same 100 items, swap the system, compare F1" design the professor asked for.#
# --------------------------------------------------------------------------- #
def culfit_baseline_provider(evaluator: "TaskEvaluator"):
    """CulFiT-framework baseline: bare model answers the QA task one-shot."""
    def provide(item: dict) -> str:
        gt = item.get("ground_truth", item)
        return evaluator.generate_answer(
            question=item["query"],
            group=gt.get("location", item.get("location", "Unknown")),
            topic=gt.get("sub_topic", item.get("sub_topic", "Unknown")))
    return provide


def agentic_answer_provider(build_orchestrator: Callable[[dict], object],
                            evaluator: "TaskEvaluator",
                            topology: str = "sequential",
                            style: str = "verbose"):
    """
    Agentic-system answer: run the orchestrator to produce final paths + the
    reconstructed context, then have the model WRITE a task answer conditioned on
    that augmentation. This is the professor's framing — the paths/critique are
    *augmentation*, and an orchestrator LLM solves the actual task on top of them.

    style:
      "verbose" — original prompt. Tells the model to "use the augmentation",
                  which in practice makes it NARRATE the scaffolding into the
                  answer ("according to the reasoning paths...", "as seen in the
                  event where..."). Those meta-sentences become knowledge units
                  that match nothing in the golden set, so precision collapses.
      "terse"   — same augmentation, but the model is told to treat paths as
                  BACKGROUND ONLY, answer the question directly, state only
                  concrete cultural facts, and NEVER mention paths/events/agents.
                  Isolates "is the augmentation unhelpful?" from "is it being
                  consumed badly?" — the delta between the two styles quantifies
                  how much of the precision loss is scaffolding-narration.
    """
    def provide(item: dict) -> str:
        orch = build_orchestrator(item)
        gt = item["ground_truth"]
        if topology == "static":
            result = orch.static_integration(item["query"], gt)
        elif topology == "parallel":
            result = orch.parallel_debate(item["query"], gt)
        else:
            result = orch.sequential_debate(item["query"], gt)

        paths = result.get("final_paths", [])
        trace = result.get("trace", {})
        central = trace.get("central_nodes", [])
        critique_fb = ""
        crit = trace.get("critique") or {}
        if isinstance(crit, dict):
            critique_fb = crit.get("feedback", "") or ""

        aug = []
        if paths:
            aug.append("Reconstructed cultural reasoning paths:\n"
                       + json.dumps(paths, ensure_ascii=False)[:2000])
        if central:
            aug.append("Central cultural concepts: " + ", ".join(map(str, central)))
        if critique_fb:
            aug.append("Critique of the above evidence:\n" + critique_fb[:800])
        augmentation = "\n\n".join(aug) if aug else "(no augmentation available)"

        if style == "terse":
            prompt = (
                "Answer the cultural question below directly and concisely. You "
                "are given background evidence from a multi-agent system. Treat it "
                "strictly as BACKGROUND: draw on it only where it states a "
                "concrete, verifiable cultural fact, and ignore anything generic "
                "or speculative. State only specific cultural facts (names, "
                "practices, dates, customs). Do NOT describe your reasoning, do "
                "NOT mention 'paths', 'events', 'evidence', or 'agents', and do "
                "NOT hedge with 'may' or 'might'. Write it as a direct factual "
                "answer a knowledgeable local would give.\n\n"
                f"Cultural group: {gt.get('location','Unknown')}\n"
                f"Topic: {gt.get('sub_topic','Unknown')}\n"
                f"Question: {item['query']}\n\n"
                f"--- Background (do not quote or reference) ---\n{augmentation}\n"
                "--- End background ---\n\n"
                "Direct factual answer:")
        else:  # verbose (original)
            prompt = (
                "You are solving an open-ended cultural question. You are given "
                "augmentation from a multi-agent system: reconstructed reasoning "
                "paths, central concepts, and a critique of that evidence. Use the "
                "augmentation where it is helpful and IGNORE any part the critique "
                "flags as unhelpful or generic. Then write a specific, culturally "
                "grounded answer.\n\n"
                f"Cultural group: {gt.get('location','Unknown')}\n"
                f"Topic: {gt.get('sub_topic','Unknown')}\n"
                f"Question: {item['query']}\n\n"
                f"--- Agentic augmentation ---\n{augmentation}\n--- End augmentation ---\n\n"
                "Answer:")
        return evaluator.backend.chat(
            [{"role": "user", "content": prompt}], temperature=0.7) or ""
    return provide


def evaluate_system(items: list[dict], provider: Callable[[dict], str],
                    evaluator: "TaskEvaluator", system_name: str) -> dict:
    """
    Run one system over all items, score each answer, and aggregate. Returns a
    dict with the macro-averaged P/R/F1 (CulFiT reports macro over items) and a
    per-item record list for inspection / error analysis.
    """
    per_item = []
    for i, item in enumerate(items):
        gt = item.get("ground_truth", item)
        group = gt.get("location", item.get("location", "Unknown"))
        verified = gt.get("verified_points", [])
        try:
            answer = provider(item)
        except Exception as e:  # noqa: BLE001 — keep the batch alive, log the item
            logger.exception("provider failed on item %d (%s): %s", i, system_name, e)
            answer = ""
        scored = evaluator.score_answer(
            answer, verified, group=group, question=item.get("query", ""))
        rec = {
            "idx": i,
            "query": item.get("query"),
            "location": group,
            "system": system_name,
            "answer": answer,
            **{k: scored[k] for k in ("precision", "recall", "f1",
                                      "n_answer_units", "n_golden_units",
                                      "covered_golden", "missed_golden")
               if k in scored},
        }
        per_item.append(rec)
        logger.info("[%s] item %d/%d  P=%.3f R=%.3f F1=%.3f",
                    system_name, i + 1, len(items),
                    scored["precision"], scored["recall"], scored["f1"])

    n = len(per_item) or 1
    macro = {
        "system": system_name,
        "n_items": len(per_item),
        "precision": sum(r["precision"] for r in per_item) / n,
        "recall": sum(r["recall"] for r in per_item) / n,
        "f1": sum(r["f1"] for r in per_item) / n,
    }
    return {"summary": macro, "per_item": per_item}


# --------------------------------------------------------------------------- #
# Judge-reliability subset.                                                    #
#                                                                             #
# The judge's atomic operation is: (candidate point, reference list) -> Yes/No.#
# We cannot validate a Llama judge against gpt-4o-mini (no API), so the honest #
# substitute is human agreement on a small stratified subset. These helpers    #
# (1) dump real (candidate, reference, judge_verdict) triples to a CSV with a   #
# blank human_label column, and (2) read the filled CSV back and report        #
# judge<->human agreement plus the disagreements. Labeling is a one-time ~15min #
# calibration done OUTSIDE the main run.                                        #
# --------------------------------------------------------------------------- #
def dump_judge_pairs(evaluator: "TaskEvaluator", items: list[dict],
                     out_csv: str, n_pairs: int = 20,
                     stratify: bool = True) -> int:
    """
    Generate real judge pairs from the eval items and write them for labeling.

    For a spread of items we decompose the GOLDEN answer's own verified_points as
    'candidate' points (guaranteed real cultural text) and judge each against the
    full golden list. We deliberately also inject some cross-item candidates
    (a point from a DIFFERENT cultural group) so the subset contains obvious
    non-matches and borderline cases, not just trivial self-matches.

    Writes CSV columns: pair_id, candidate, reference_list_json, judge_verdict,
    human_label(blank), note. Returns number of pairs written.
    """
    import csv
    import random

    rng = random.Random(13)
    rows = []
    pid = 0
    n_items = len(items)
    # Iterate items in a shuffled order so a small n_pairs still samples across
    # different cultural groups rather than just the first few items.
    order = list(range(n_items))
    rng.shuffle(order)
    for i in order:
        item = items[i]
        gt = item.get("ground_truth", item)
        golden = [p for p in (gt.get("verified_points", []) or []) if str(p).strip()]
        if len(golden) < 2:
            continue
        # (a) a real in-item candidate: one golden point vs the full golden list
        #     -> should be a MATCH (self-consistency probe).
        cand_in = golden[0]
        v_in = evaluator._unit_matches_reference(cand_in, golden)
        rows.append((pid, cand_in, golden, v_in, "in-item (expect match)"))
        pid += 1
        if len(rows) >= n_pairs:
            break
        # (b) a cross-item candidate: a golden point from a DIFFERENT item judged
        #     against THIS item's golden list -> usually a NON-match; borderline
        #     when topics overlap. This is the discriminating part of the subset.
        if stratify and n_items > 1:
            for _try in range(5):
                j = rng.randrange(n_items)
                if j == i:
                    continue
                other = items[j].get("ground_truth", items[j])
                other_golden = [p for p in (other.get("verified_points", []) or [])
                                if str(p).strip()]
                if other_golden:
                    cand_x = other_golden[0]
                    v_x = evaluator._unit_matches_reference(cand_x, golden)
                    rows.append((pid, cand_x, golden, v_x,
                                 "cross-item (expect non-match)"))
                    pid += 1
                    break
        if len(rows) >= n_pairs:
            break

    rows = rows[:n_pairs]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "candidate", "reference_list_json",
                    "judge_verdict", "human_label", "note"])
        for pid_, cand, ref, verdict, note in rows:
            w.writerow([pid_, cand, json.dumps(ref, ensure_ascii=False),
                        "Yes" if verdict else "No", "", note])
    return len(rows)


def score_judge_reliability(csv_path: str) -> dict:
    """
    Read a human-labeled judge-pair CSV and report judge<->human agreement.
    human_label accepts Yes/No/Y/N/1/0/match/no (case-insensitive). Rows with a
    blank human_label are skipped (and counted as unlabeled).
    """
    import csv

    def norm(x):
        s = (x or "").strip().lower()
        if s in ("yes", "y", "1", "true", "match", "m"):
            return True
        if s in ("no", "n", "0", "false", "nomatch", "no-match"):
            return False
        return None

    total, labeled, agree = 0, 0, 0
    disagreements = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            human = norm(row.get("human_label"))
            judge = norm(row.get("judge_verdict"))
            if human is None:
                continue
            labeled += 1
            if human == judge:
                agree += 1
            else:
                disagreements.append({
                    "pair_id": row.get("pair_id"),
                    "candidate": row.get("candidate"),
                    "judge": row.get("judge_verdict"),
                    "human": row.get("human_label"),
                    "note": row.get("note"),
                })
    return {
        "n_rows": total,
        "n_labeled": labeled,
        "n_unlabeled": total - labeled,
        "agreement": (agree / labeled) if labeled else None,
        "n_agree": agree,
        "n_disagree": labeled - agree,
        "disagreements": disagreements,
    }
