"""
Agent A adapter (in-place): wraps Tonga's CCKG generation into a callable agent
exposing .generate(query) and .repair(prompt), importing the REAL Tonga modules
in place via bootstrap (no copies).

Optional graph reconstruction:
  When `reconstruct_graph=True`, the adapter additionally merges the generated
  paths into a CulturalGraph and produces a condensed natural-language
  contextualization (see orchestration.graph_reconstruction). This is ADDITIVE:
  generate() still returns list[dict] exactly as the AGENT_CONTRACT specifies, so
  the orchestrator and topologies are unchanged. The reconstruction artifact is
  exposed two ways:
    - side-channel: self.last_context (populated on every generate() call)
    - explicit:     generate_with_context(query) -> {"paths", "contextualization", ...}
  The GraphReconstructor is built lazily so SBERT is never imported unless graph
  reconstruction is actually requested (mirrors the lazy-import discipline used
  elsewhere in this layer).
"""

from __future__ import annotations
import logging
from types import SimpleNamespace
from typing import Optional

from orchestration.llm_backend import LLMBackend
from orchestration import bootstrap

_mods = bootstrap.install()
generation_prompts = _mods["agentA_prompt_templates"].generation_prompts
parse_llm_response = _mods["agentA_response_parser"].parse_llm_response

logger = logging.getLogger(__name__)


class AgentA:
    def __init__(self, backend: LLMBackend, location: str = "England",
                sub_topic: str = "breakfast", language: str = "English",
                prompt_key: str | None = None, max_paths: int | None = None,
                reconstruct_graph: bool = False,
                merge_threshold: float = 0.8, use_sbert: bool = True):
        self.backend = backend
        self.location = location
        self.sub_topic = sub_topic
        self.language = language
        self.prompt_key = prompt_key or "England_gen"
        self.max_paths = max_paths  # None = no cap; useful to keep smoke runs fast

        # Graph reconstruction config (lazy: reconstructor built on first use).
        self.reconstruct_graph = reconstruct_graph
        self.merge_threshold = merge_threshold
        self.use_sbert = use_sbert
        self._reconstructor = None
        # Side-channel: last reconstruction artifact, or None. Lets the
        # orchestrator read context without changing generate()'s return type.
        self.last_context: Optional[dict] = None

    def _get_reconstructor(self):
        """Build the GraphReconstructor once, on first use (lazy import)."""
        if self._reconstructor is None:
            from orchestration.graph_reconstruction import GraphReconstructor
            self._reconstructor = GraphReconstructor(
                threshold=self.merge_threshold, use_sbert=self.use_sbert)
        return self._reconstructor

    def _build_generation_messages(self, query: str) -> list[dict]:
        template = generation_prompts[self.prompt_key]
        sub_topic = query or self.sub_topic
        msgs = []
        for m in template:
            content = m["content"].format(
                location=self.location, sub_topic=sub_topic,
                language=self.language, premise="",
            )
            msgs.append({"role": m["role"], "content": content})
        return msgs

    def _parse(self, raw_text: str) -> list[dict]:
        args = SimpleNamespace(action="initial_generation")
        try:
            parsed = parse_llm_response(args, raw_text)
        except Exception as e:
            logger.warning("Agent A parse failed (%s); returning empty.", e)
            return []
        if not isinstance(parsed, list):
            return []
        for entry in parsed:
            entry.setdefault("location", self.location)
            entry.setdefault("sub_topic", self.sub_topic)
        if self.max_paths is not None:
            parsed = parsed[:self.max_paths]
        return parsed

    def generate(self, query: str) -> list[dict]:
        messages = self._build_generation_messages(query)
        raw = self.backend.chat(messages, temperature=1.0)
        paths = self._parse(raw)
        logger.info("Agent A generated %d path(s).", len(paths))
        # Additive: populate the reconstruction side-channel when enabled.
        # generate() STILL returns list[dict] per the AGENT_CONTRACT.
        if self.reconstruct_graph:
            self.last_context = self._reconstruct(paths)
        else:
            self.last_context = None
        return paths

    def _reconstruct(self, paths: list[dict]) -> Optional[dict]:
        """Build the reconstruction artifact from paths, tolerant of failure."""
        if not paths:
            return None
        try:
            return self._get_reconstructor().reconstruct(paths)
        except Exception as e:
            logger.warning("Graph reconstruction failed (%s); continuing without it.", e)
            return None

    def generate_with_context(self, query: str) -> dict:
        """
        Explicit API for callers that want BOTH the paths and the graph
        contextualization in one call (e.g. the re-augmentation loop). Returns:
            {
              "paths": list[dict],            # same as generate()
              "contextualization": str|None,  # condensed NL augmentation
              "central_nodes": [(label, degree), ...] | [],
              "stats": {...} | {},
              "graph": CulturalGraph | None,  # for downstream centrality use
            }
        Forces reconstruction for this call regardless of the instance flag, so a
        caller can opt in per-call without reconfiguring the agent.
        """
        paths = self.generate(query)
        artifact = self.last_context
        if artifact is None and paths:
            # instance flag was off; reconstruct on demand for this call
            artifact = self._reconstruct(paths)
            self.last_context = artifact
        if artifact is None:
            return {"paths": paths, "contextualization": None,
                    "central_nodes": [], "stats": {}, "graph": None}
        return {
            "paths": paths,
            "contextualization": artifact.get("contextualization"),
            "central_nodes": artifact.get("central_nodes", []),
            "stats": artifact.get("stats", {}),
            "graph": artifact.get("graph"),
        }

    def repair(self, repair_prompt: str) -> list[dict]:
        messages = [
            {"role": "system",
             "content": "You are a cultural commonsense knowledge extraction "
                        "assistant. Output STRICT JSON only: a JSON array of "
                        "objects with keys action, knowledge, relation_type, result."},
            {"role": "user", "content": repair_prompt},
        ]
        raw = self.backend.chat(messages, temperature=0.7)
        paths = self._parse(raw)
        logger.info("Agent A repaired into %d path(s).", len(paths))
        return paths

    def static_context(self, query: str) -> list[dict]:
        return self.generate(query)
