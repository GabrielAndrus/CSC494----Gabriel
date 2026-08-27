"""
Agent B critique engine (in-place): CulFiT Group/Topic/KnowledgePoint checks
using the REAL EVAL_* prompts from CulFiT/utils/prompt_utils.py (loaded in place
via bootstrap), routed through the model-agnostic backend. Adds get_static_schema().

Optional graph contextualization (Option A — shared evidence):
  The methods accept an optional `context` string (the condensed natural-language
  graph reconstruction from Agent A). When supplied, it is PREPENDED as a separate
  evidence block to the Knowledge-Path check ONLY — the Cultural-Group and Topic
  checks, the scoring, and the return shape are all unchanged. This isolates the
  graph structure as the single changed variable, so an ablation that toggles
  `context` on/off measures exactly one thing: does showing the connective
  structure help Agent B judge knowledge-path alignment?

  The upstream CulFiT prompt is NOT modified or reformatted. The context is added
  as a wrapper around the verbatim formatted prompt, preserving in-place
  integration (we never edit CulFiT/utils/prompt_utils.py).
"""

from __future__ import annotations
import logging
from typing import Optional
import re

from orchestration.llm_backend import LLMBackend
from orchestration import bootstrap

_mods = bootstrap.install()
_pu = _mods["culfit_prompt_utils"]
EVAL_CULTURAL_POINTS_PROMPT = _pu.EVAL_CULTURAL_POINTS_PROMPT
EVAL_CULTURAL_GROUP_PROMPT = _pu.EVAL_CULTURAL_GROUP_PROMPT
EVAL_TOPIC_PROMPT = _pu.EVAL_TOPIC_PROMPT

logger = logging.getLogger(__name__)

# helper: robust yes/no on a single judge response
def _verdict_is_yes(res: str) -> bool:
    """
    Robust replacement for the fragile `"Yes" in res` test: strip markdown /
    whitespace, look at the leading token. Avoids 'Yes' matching inside words
    like 'Yesterday' or a verbose 'No, but yes in part...' answer.
    """
    if not res:
        return False
    s = res.strip().lstrip("*_#>-• ").strip()
    head = re.split(r"[\s,.:;!]+", s, maxsplit=1)[0].lower() if s else ""
    return head in ("yes", "y", "true", "correct", "aligned")


# helper: cheap lexical overlap (no model call) 
_STOP = {
    "the", "a", "an", "of", "in", "on", "to", "and", "or", "is", "are", "was",
    "were", "be", "been", "by", "for", "with", "as", "that", "this", "their",
    "they", "it", "its", "at", "from", "may", "can", "such", "which", "who",
}


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if w not in _STOP and len(w) > 2}


def _lexical_covers(path_text: str, point_text: str, threshold: float = 0.34) -> bool:
    """
    Deterministic, model-free coverage test: does the path share enough content
    tokens with the verified point? Recall-of-point-tokens-in-path >= threshold.
    Conservative; meant as a reproducible floor, not a semantic judge.
    """
    pt = _tokens(point_text)
    if not pt:
        return False
    overlap = len(pt & _tokens(path_text)) / len(pt)
    return overlap >= threshold


class AgentBCritiqueEngine:
    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def _generate(self, messages: list[dict]) -> str:
        return self.backend.chat(messages, temperature=0.0) or ""

    @staticmethod
    def _wrap_with_context(kp_prompt: str, context: Optional[str]) -> str:
        """
        Prepend the graph contextualization as a separate evidence block before
        the verbatim CulFiT Knowledge-Path prompt. The upstream prompt string is
        untouched; we only add a labelled preamble so the judge can use the
        connective structure as shared evidence. Returns kp_prompt unchanged when
        context is empty, so the no-context arm is byte-identical to the baseline.
        """
        if not context:
            return kp_prompt
        return (
            "You are also given a reconstructed cultural knowledge graph that shows "
            "how the candidate reasoning paths connect, including which actions are "
            "central (recur across many paths). Use it as supporting context when "
            "judging the knowledge point below.\n"
            "--- Reconstructed cultural context ---\n"
            f"{context}\n"
            "--- End context ---\n\n"
            f"{kp_prompt}"
        )

    def get_static_schema(self, query: str) -> dict:
        return {
            "query": query,
            "evaluation_dimensions": [
                "Cultural Group Alignment", "Topic Alignment", "Knowledge Path Alignment"],
            "note": "Static CulFiT schema; no deliberation performed.",
        }

    def evaluate_single_path(self, agent_a_node: dict, ground_truth: dict,
                             context: Optional[str] = None) -> dict:
        answer_cultural_group = agent_a_node.get("location", "Unknown")
        answer_topic = agent_a_node.get("sub_topic", "Unknown")
        answer_knowledge_point = agent_a_node.get("llm_result", "")
        grounded_cultural_group = ground_truth.get("location", "Unknown")
        grounded_topic = ground_truth.get("sub_topic", "Unknown")
        grounded_knowledge_points = ground_truth.get("verified_points", [])

        msg_group = [{"role": "user", "content": EVAL_CULTURAL_GROUP_PROMPT.format(
            answer_cultural_group, grounded_cultural_group)}]
        msg_topic = [{"role": "user", "content": EVAL_TOPIC_PROMPT.format(
            answer_topic, grounded_topic)}]
        # Graph context is shared EVIDENCE for the knowledge-path judgement only.
        # Group/Topic checks are deliberately left untouched so the only changed
        # variable in a context-on/off ablation is the KP check's evidence.
        kp_prompt = EVAL_CULTURAL_POINTS_PROMPT.format(
            answer_knowledge_point, grounded_knowledge_points)
        msg_kp = [{"role": "user", "content": self._wrap_with_context(kp_prompt, context)}]

        res_group = self._generate(msg_group)
        res_topic = self._generate(msg_topic)
        res_kp = self._generate(msg_kp)

        labels = ["Cultural Group Alignment", "Topic Alignment", "Knowledge Path Alignment"]
        verdicts, critique_details = [], []
        for label, res in zip(labels, [res_group, res_topic, res_kp]):
            if res and "Yes" in res:
                verdicts.append("Yes")
            else:
                verdicts.append("No")
                critique_details.append(f"[{label} Failure]: {res}")

        precision = sum(1 for v in verdicts if v == "Yes") / len(verdicts)
        approved = (precision == 1.0)
        return {
            "approved": approved, "precision_score": precision,
            "verdicts": dict(zip(labels, verdicts)),
            "feedback": " ".join(critique_details) if not approved
                        else "Culturally verified and validated.",
        }

    def evaluate_payload_batch(self, agent_a_payload_list, 
                               ground_truth_schema,
                               context=None, max_paths_binary=5):
        scored_payload = agent_a_payload_list[:max_paths_binary] if max_paths_binary else agent_a_payload_list
        all_approved = True
        accumulated_feedback, scores, per_path = [], [], []
        for item in scored_payload:
            ev = self.evaluate_single_path(item, ground_truth_schema, context=context)
            scores.append(ev["precision_score"])
            per_path.append({"path": item.get("llm_result", ""), **ev})
            if not ev["approved"]:
                all_approved = False
                accumulated_feedback.append(
                    f"Offending Path: '{item.get('llm_result','')}' -> Reason: {ev['feedback']}")
        mean_precision = sum(scores) / len(scores) if scores else 0.0
        path_texts = [item.get("llm_result", "") for item in agent_a_payload_list]
        kp = self.score_knowledge_points(path_texts, ground_truth_schema,
                                         mode="judge", max_paths_scored=5)
        return {
            "approved": all_approved, "mean_precision": mean_precision,
            "n_paths": len(agent_a_payload_list), "per_path": per_path,
            "context_used": bool(context),  # log the ablation arm for this batch
            "feedback": "\n".join(accumulated_feedback) if accumulated_feedback
                        else "All structural reasoning paths passed CulFiT verification.",
            "kp": kp,
        }
    
    # The existing `precision_score` in evaluate_single_path is CulFiT's metric:
    #     (#"Yes" among 3 rubric labels) / 3
    # i.e. a binary judge over Cultural-Group / Topic / Knowledge-Path alignment.
    # It saturates to 1.0 on plausible paths and does NOT count how many of the
    # ground-truth `verified_points` a path actually covers. That binary judge must
    # stay intact (it drives `approved` and the repair loop, and is the number
    # comparable to CulFiT's own reported metric).
    #
    # This addition computes a SEPARATE, point-level score over verified_points:
    #   - kp_recall    : fraction of ground-truth verified_points covered by the
    #                    path set (did the system surface the right facts?)
    #   - kp_precision : fraction of generated paths that map to >=1 verified_point
    #                    (are the generated paths on-target rather than generic?)
    # These are additive: existing fields are unchanged, so no control flow or
    # CulFiT-comparability is affected. Everything new lives under new dict keys.

    # COST NOTE
    # Judge mode issues (n_paths * n_points) extra chat() calls per critique. That is
    # why it is OPT-IN (score_kpts=False by default). For a cheap first pass, mode
    # "lexical" uses token-overlap and makes ZERO model calls — good for a fast
    # variance check before paying for the judge-based version on the full run.
    #
    # kp_recall via the LLM judge is still an LLM-graded metric; it is more
    # discriminating than the binary rubric but not an oracle. It is a
    # LLM-judged knowledge-point recall, not as exact-match F1, unless we also
    # add a human-checked subset. The lexical mode is fully deterministic and
    # reproducible, which makes it a good sanity floor.
    
    # single point x single path: does the path express this fact? 
    def _point_covered_by_path_llm(self, path_text: str, point_text: str,
                                   cultural_group: str) -> bool:
        """One judge call: does `path_text` express or entail `point_text`?"""
        prompt = (
            "You are checking whether a candidate reasoning path expresses a "
            "specific ground-truth cultural fact.\n"
            f"Cultural group: {cultural_group}\n"
            f"Candidate reasoning path:\n{path_text}\n\n"
            f"Ground-truth fact:\n{point_text}\n\n"
            "Does the candidate path express, entail, or directly support the "
            "ground-truth fact? Answer with a single word: Yes or No."
        )
        res = self.backend.chat([{"role": "user", "content": prompt}],
                                temperature=0.0) or ""
        return _verdict_is_yes(res)

    # knowledge-point scoring over a set of paths 
    def score_knowledge_points(self, path_texts: list[str], ground_truth: dict,
                               mode: str = "lexical",
                               max_paths_scored: int | None = None) -> dict:
        """
        Compute point-level recall/precision of the generated paths against
        ground_truth['verified_points'].

        mode = "lexical" : deterministic token overlap, ZERO model calls (fast).
        mode = "judge"   : one chat() per (path, point) pair (slow, semantic).

        Returns a dict (all NEW keys; nothing here mutates existing scores):
          {
            "kp_recall": float,        # covered_points / total_points
            "kp_precision": float,     # on_target_paths / total_paths
            "n_points": int,
            "n_points_covered": int,
            "covered_points": [str, ...],
            "missed_points": [str, ...],
            "kp_mode": "lexical" | "judge",
          }
        """
        points = ground_truth.get("verified_points", []) or []
        group = ground_truth.get("location", "Unknown")
        n_points = len(points)

        # Cost cap: in judge mode, scoring every generated path against every
        # point is n_paths * n_points model calls. Agent A can emit 30+ paths, so
        # we cap how many paths are scored. We keep the FIRST max_paths_scored
        # paths (Agent A emits them in generation order; this is a deterministic,
        # reproducible subset — NOT a quality ranking, so report it as such).
        # n_paths_total is preserved for honesty about the denominator.
        n_paths_total = len(path_texts)
        scored_paths = path_texts
        if max_paths_scored is not None and n_paths_total > max_paths_scored:
            scored_paths = path_texts[:max_paths_scored]
        n_paths = len(scored_paths)

        if n_points == 0 or n_paths == 0:
            # Vacuous case made explicit rather than silently scoring 1.0.
            return {
                "kp_recall": 0.0, "kp_precision": 0.0,
                "n_points": n_points, "n_points_covered": 0,
                "n_paths_scored": n_paths, "n_paths_total": n_paths_total,
                "covered_points": [], "missed_points": list(points),
                "kp_mode": mode, "note": "no points or no paths",
            }

        def covers(path_text, point_text):
            if mode == "judge":
                return self._point_covered_by_path_llm(path_text, point_text, group)
            return _lexical_covers(path_text, point_text)

        # Compute the path x point coverage grid ONCE (the old code ran covers()
        # twice — recall loop + precision loop — doubling judge-mode calls).
        grid = [[covers(p, pt) for pt in points] for p in scored_paths]

        covered_points, missed_points = [], []
        for j, pt in enumerate(points):
            if any(grid[i][j] for i in range(n_paths)):
                covered_points.append(pt)
            else:
                missed_points.append(pt)

        on_target_paths = sum(1 for i in range(n_paths) if any(grid[i]))

        return {
            "kp_recall": len(covered_points) / n_points,
            "kp_precision": on_target_paths / n_paths,
            "n_points": n_points,
            "n_points_covered": len(covered_points),
            "n_paths_scored": n_paths,
            "n_paths_total": n_paths_total,
            "covered_points": covered_points,
            "missed_points": missed_points,
            "kp_mode": mode,
        }

