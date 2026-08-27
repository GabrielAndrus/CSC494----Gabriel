"""
CulturalAgentOrchestrator: mediates Agent A (CCKG generator) and Agent B
(CulFiT critique) across three communication topologies for the ablation study.

Fixes over the original orchestrator.py:
  - static_integration / parallel_debate previously called a non-existent
    `agent_b.get_static_schema()` -> now provided by the engine.
  - Agent A now returns parsed lists (not raw strings), so `_safe_json_parse`
    is applied only where a backend might still hand back a raw string.
  - Every mode returns a structured `trace` (verdicts, precision, loop count,
    repair attempts) so metrics can be chosen after runs.

An `arbiter_backend` (LLMBackend) is used by parallel_debate for synthesis.

Graph-context ablation (Agent A reconstruction -> Agent B shared evidence):
  The orchestrator carries a single switch, `use_context`, that controls whether
  Agent A's reconstructed graph contextualization is passed to Agent B's
  knowledge-path check. This is the ONLY lever that changes between ablation arms:
    - use_context=False  -> Arm 1 "no-context": every evaluate_payload_batch call
      receives context=None, so behaviour is byte-identical to the pre-context
      orchestrator. This is the control / baseline.
    - use_context=True, context_mode="template" -> Arm 2 "template-context": the
      deterministic graph_reconstruction NL summary is prepended as shared
      evidence to the KP check.
    - context_mode="llm" -> Arm 3 "llm-rewrite" (EXTENSION POINT, not built yet):
      reserved for an LLM-rewritten contextualization. Selecting it today raises,
      so the arm can't be run silently/half-wired.
  All three topologies route generation+contextualization through one helper so
  the arms stay parallel, and the chosen arm is recorded in every trace.
"""

from __future__ import annotations
import json
import logging

logger = logging.getLogger(__name__)


class CulturalAgentOrchestrator:
    def __init__(self, agent_a, agent_b_engine, arbiter_backend=None,
                 use_context: bool = False, context_mode: str = "template"):
        self.agent_a = agent_a
        self.agent_b = agent_b_engine
        self.arbiter_backend = arbiter_backend
        # Ablation arm controls.
        self.use_context = use_context
        self.context_mode = context_mode

    # ------------------------------------------------------------------ #
    # Arm logic: one place decides whether/how context is produced.       #
    # ------------------------------------------------------------------ #
    def _arm_label(self) -> str:
        return f"{self.context_mode}-context" if self.use_context else "no-context"

    def _generate_and_contextualize(self, query):
        """
        Returns (paths, context_str, central_nodes).
        - use_context=False: plain generate(), context None -> baseline arm.
        - use_context=True, "template": deterministic reconstruction NL summary.
        - use_context=True, "llm": reserved extension point (raises for now).
        """
        if not self.use_context:
            return self.agent_a.generate(query), None, []

        if self.context_mode == "template":
            out = self.agent_a.generate_with_context(query)
            return out["paths"], out.get("contextualization"), out.get("central_nodes", [])

        if self.context_mode == "llm":
            raise NotImplementedError(
                "context_mode='llm' (LLM-rewrite arm) is not wired yet. Use "
                "'template' or set use_context=False. The extension point lives "
                "here: produce the paths via agent_a.generate_with_context(query) "
                "to get the graph, then rewrite out['contextualization'] with an "
                "LLM call before passing it to Agent B.")

        raise ValueError(f"Unknown context_mode: {self.context_mode!r} "
                         "(expected 'template' or 'llm').")

    def _recontextualize(self, paths):
        """
        Recompute context from a (possibly repaired) path list mid-loop, so the
        evidence Agent B sees always matches the paths being judged. Returns
        (context_str, central_nodes); ('', []) when context is disabled or paths
        are empty.
        """
        if not self.use_context or not paths:
            return None, []
        if self.context_mode == "template":
            artifact = self.agent_a._reconstruct(paths)
            if not artifact:
                return None, []
            return artifact.get("contextualization"), artifact.get("central_nodes", [])
        if self.context_mode == "llm":
            raise NotImplementedError("context_mode='llm' not wired yet (see _generate_and_contextualize).")
        raise ValueError(f"Unknown context_mode: {self.context_mode!r}")

    @staticmethod
    def _coerce_paths(maybe_paths, fallback):
        """Agent A returns a list already; tolerate a raw string just in case."""
        if isinstance(maybe_paths, list):
            return maybe_paths
        if isinstance(maybe_paths, str):
            import re
            try:
                return json.loads(maybe_paths)
            except json.JSONDecodeError:
                m = re.search(r'\[\s*\{.*?\}\s*\]', maybe_paths, re.DOTALL)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except json.JSONDecodeError:
                        pass
            logger.warning("Could not coerce Agent A output; using fallback.")
        return fallback

    def static_integration(self, query, ground_truth_schema):
        logger.info("[static] start (arm=%s)", self._arm_label())
        paths, context, central = self._generate_and_contextualize(query)
        schema = self.agent_b.get_static_schema(query)
        # Single critique pass for a comparable quality signal (no repair).
        critique = self.agent_b.evaluate_payload_batch(
            paths, ground_truth_schema, context=context) if paths else {
            "approved": False, "mean_precision": 0.0, "n_paths": 0, "per_path": [],
            "context_used": False, "feedback": "No paths generated."}
        return {
            "mode": "static",
            "final_paths": paths,
            "cultural_rules": schema,
            "trace": {"loops": 0, "repairs": 0,
                      "context_mode": self._arm_label(),
                      "central_nodes": central,
                      "final_approved": critique["approved"],
                      "final_mean_precision": critique["mean_precision"],
                      "critique": critique},
        }

    def parallel_debate(self, query, ground_truth_schema):
        logger.info("[parallel] start (arm=%s)", self._arm_label())
        draft, context, central = self._generate_and_contextualize(query)
        schema = self.agent_b.get_static_schema(query)
        critique = self.agent_b.evaluate_payload_batch(
            draft, ground_truth_schema, context=context) if draft else {
            "approved": False, "mean_precision": 0.0, "n_paths": 0, "per_path": [],
            "context_used": False, "feedback": "No paths generated."}

        synthesis = None
        if self.arbiter_backend is not None:
            prompt = (
                "You are the final arbiter. Reconcile Agent A's cultural graph "
                "paths and Agent B's critique into a single culturally sensitive "
                "JSON array.\n"
                f"Graph Paths: {json.dumps(draft)}\n"
                f"Critique: {json.dumps(critique.get('feedback'))}\n"
                f"Schema: {json.dumps(schema)}\nOutput strict JSON only."
            )
            raw = self.arbiter_backend.chat(
                [{"role": "user", "content": prompt}], temperature=0.3)
            synthesis = self._coerce_paths(raw, draft)

        final_paths = synthesis if synthesis else draft
        return {
            "mode": "parallel",
            "final_paths": final_paths,
            "cultural_rules": schema,
            "trace": {"loops": 1, "repairs": 0,
                      "context_mode": self._arm_label(),
                      "central_nodes": central,
                      "final_approved": critique["approved"],
                      "final_mean_precision": critique["mean_precision"],
                      "critique": critique,
                      "arbiter_used": self.arbiter_backend is not None},
        }

    def sequential_debate(self, query, ground_truth_schema, max_loops=3):
        logger.info("[sequential] start (max_loops=%d, arm=%s)", max_loops, self._arm_label())
        current, context, central = self._generate_and_contextualize(query)
        current = self._coerce_paths(current, [])
        if not current:
            logger.error("[sequential] empty initial draft; aborting.")
            return {"mode": "sequential", "final_paths": [],
                    "trace": {"loops": 0, "repairs": 0,
                              "context_mode": self._arm_label(), "central_nodes": [],
                              "final_approved": False,
                              "final_mean_precision": 0.0, "iterations": []}}

        iterations, repairs = [], 0
        final_critique = None
        last_central = central
        for i in range(max_loops):
            critique = self.agent_b.evaluate_payload_batch(
                current, ground_truth_schema, context=context)
            final_critique = critique
            iterations.append({
                "iteration": i + 1,
                "approved": critique["approved"],
                "mean_precision": critique["mean_precision"],
                "n_paths": critique["n_paths"],
                "context_used": critique.get("context_used", False),
            })
            if critique.get("approved"):
                logger.info("[sequential] approved at iteration %d", i + 1)
                break
            if i == max_loops - 1:
                break  # no point repairing on the last allowed loop
            repair_prompt = (
                "Your previous draft was critiqued.\n"
                f"Original Draft: {json.dumps(current)}\n"
                f"Critique: {critique.get('feedback')}\n"
                "Repair the empirical paths to satisfy the cultural critique. "
                "Output strict JSON only."
            )
            repaired = self._coerce_paths(self.agent_a.repair(repair_prompt), current)
            repairs += 1
            # Path-collapse guard. The sequential repair loop can probabilistically
            # collapse the path set (e.g. 27 -> 1, or -> 0), which can FAKE an
            # approval on a near-empty set and inflate approval rates without
            # better reasoning. We reject a repair that (a) is empty or (b) shrinks
            # the set below a floor relative to the pre-repair count, keeping the
            # prior paths instead. We log it so collapses are visible in the trace
            # rather than silently shaping the metric.
            n_before = len(current)
            n_after = len(repaired) if repaired else 0
            collapse_floor = max(2, int(0.5 * n_before))  # require >=50% retained, min 2
            if n_after < collapse_floor:
                logger.warning(
                    "[sequential] repair collapsed %d -> %d paths (floor=%d); "
                    "rejecting repair, keeping prior paths.",
                    n_before, n_after, collapse_floor)
                collapsed = True
                # keep `current` unchanged; do NOT recompute context from a bad set
            else:
                current = repaired
                collapsed = False
                # Paths changed -> recompute context so the evidence Agent B sees in
                # the next iteration matches the repaired paths (not stale context).
                context, last_central = self._recontextualize(current)
            iterations[-1]["repair_collapsed"] = collapsed
            iterations[-1]["n_paths_after_repair"] = n_after

        return {
            "mode": "sequential",
            "final_paths": current,
            "trace": {
                "loops": len(iterations),
                "repairs": repairs,
                "context_mode": self._arm_label(),
                "central_nodes": last_central,
                "final_approved": bool(final_critique and final_critique["approved"]),
                "final_mean_precision": final_critique["mean_precision"] if final_critique else 0.0,
                "iterations": iterations,
                "critique": final_critique,
            },
        }
