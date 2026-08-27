"""
Graph reconstruction layer for PluralTree (Agent A side).

Motivation (from the supervisor's feedback):
  Agent A retrieves up to ~12 cultural reasoning *paths* for a query. As raw
  output these are largely disjoint linear chains, and downstream agents tend to
  read them individually -- losing the connective structure that is the whole
  point of using a knowledge graph. This module finds where the paths *intersect*
  (shared/near-duplicate action nodes), MERGES them into a single directed graph,
  scores node centrality, and emits a CONDENSED natural-language contextualization
  that states how the paths connect. That contextualization is the augmentation
  fed into the downstream task (CulFiT critique / CCKG-style in-context exemplar),
  replacing a flat list of repetitive paths.

Where this lives:
  This is an Agent-A-side post-processor. It consumes the path-dict list that
  AgentA.generate() already returns (see AGENT_CONTRACT.md section 1) and produces
  (a) a CulturalGraph object and (b) a natural-language string. Nothing in the
  orchestrator's interface changes: the orchestrator still receives path dicts;
  this just adds a richer `context` artifact it can choose to pass along.

Data contract it consumes (path dict, all six keys; extras tolerated):
    event       -> head action  (A_i)   -- CCKG triple head
    knowledge   -> tail action  (A_j)   -- CCKG triple tail
    relation    -> edge label   (R)     -- xEffect / xNext / xNeed / oNext / oEffect
    llm_result  -> NL rendering of the assertion (what Agent B critiques)
    location    -> cultural group
    sub_topic   -> topic

Node-merge methodology:
  Two action strings are merged when their similarity exceeds a threshold. We
  default to the SAME mechanism the CCKG paper uses for assertion dedup during
  expansion -- SBERT cosine similarity with a 0.8 cutoff -- so merging here is
  consistent with the upstream resource and defensible in the writeup. SBERT is
  imported lazily; if unavailable (e.g. mock mode, no GPU/deps) we fall back to a
  token Jaccard similarity so the layer still runs everywhere.

No external deps required for the fallback path; numpy/sentence-transformers are
used only when SBERT is requested and importable.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Similarity backends                                                         

_WORD_RE = re.compile(r"[A-Za-z0-9\u00C0-\uFFFF]+")


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall((text or "").lower()))


def _tokens(text: str) -> set:
    return set(_WORD_RE.findall((text or "").lower()))


def jaccard_similarity(a: str, b: str) -> float:
    """Token-overlap similarity in [0, 1]. Dependency-free fallback."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 1.0 if ta == tb else 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


class SimilarityBackend:
    """Decides whether two action strings refer to the 'same' node."""

    def __init__(self, threshold: float = 0.8,
                 use_sbert: bool = True,
                 model_name: str = "all-MiniLM-L6-v2"):
        self.threshold = threshold
        self.model_name = model_name
        self._model = None
        self._use_sbert = use_sbert
        if use_sbert:
            self._try_load_sbert(model_name)

    def _try_load_sbert(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # lazy
            self._model = SentenceTransformer(model_name)
            logger.info("Graph reconstruction using SBERT (%s) for node merge.", model_name)
        except Exception as e:  # ImportError, model-download failure, etc.
            logger.info("SBERT unavailable (%s); falling back to Jaccard token overlap.", e)
            self._model = None
            self._use_sbert = False

    # public API 

    def embed(self, texts: list[str]):
        """Return embeddings (np.ndarray) for a batch, or None in fallback mode."""
        if self._model is None:
            return None
        return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def pairwise_is_same(self, a: str, b: str,
                         emb_a=None, emb_b=None) -> tuple[bool, float]:
        """Return (is_same, similarity). Uses embeddings if provided, else Jaccard."""
        if _normalize(a) == _normalize(b):
            return True, 1.0
        if emb_a is not None and emb_b is not None:
            import numpy as np
            sim = float(np.dot(emb_a, emb_b))  # normalized -> cosine
        else:
            sim = jaccard_similarity(a, b)
        return sim >= self.threshold, sim


# Graph data structures                                                       

@dataclass
class GraphNode:
    node_id: int
    label: str                       # canonical action text
    aliases: list[str] = field(default_factory=list)  # merged-in variants
    member_count: int = 1            # how many raw actions collapsed here

    def display(self) -> str:
        return self.label


@dataclass
class GraphEdge:
    src: int                         # node_id
    dst: int                         # node_id
    relation: str
    llm_result: str = ""             # NL rendering of this assertion, if any


@dataclass
class CulturalGraph:
    nodes: dict[int, GraphNode]
    edges: list[GraphEdge]
    location: str = ""
    sub_topic: str = ""

    # topology helpers 

    def degree(self) -> dict[int, int]:
        deg = defaultdict(int)
        for e in self.edges:
            deg[e.src] += 1
            deg[e.dst] += 1
        for nid in self.nodes:
            deg.setdefault(nid, 0)
        return dict(deg)

    def central_nodes(self, top_k: int = 3) -> list[tuple[int, int]]:
        """Return [(node_id, degree)] sorted by degree desc, then member_count."""
        deg = self.degree()
        ranked = sorted(
            self.nodes.keys(),
            key=lambda nid: (deg[nid], self.nodes[nid].member_count),
            reverse=True,
        )
        return [(nid, deg[nid]) for nid in ranked[:top_k]]

    def merge_stats(self) -> dict:
        n_raw_actions = sum(n.member_count for n in self.nodes.values())
        n_merged_nodes = len(self.nodes)
        return {
            "n_nodes": n_merged_nodes,
            "n_edges": len(self.edges),
            "n_raw_actions": n_raw_actions,
            "merge_ratio": (1 - n_merged_nodes / n_raw_actions) if n_raw_actions else 0.0,
            "n_merged_nodes": sum(1 for n in self.nodes.values() if n.member_count > 1),
        }

    def adjacency(self) -> dict[int, list[tuple[str, int]]]:
        """node_id -> list of (relation, dst_node_id)."""
        adj = defaultdict(list)
        for e in self.edges:
            adj[e.src].append((e.relation, e.dst))
        return dict(adj)

# Reconstruction   

class GraphReconstructor:
    """
    Turn a list of path dicts into a merged CulturalGraph, then into a condensed
    natural-language contextualization.
    """

    def __init__(self, similarity: Optional[SimilarityBackend] = None,
                 threshold: float = 0.8, use_sbert: bool = True):
        self.sim = similarity or SimilarityBackend(threshold=threshold, use_sbert=use_sbert)

    # step 1+2: build & merge 

    def build_graph(self, paths: list[dict]) -> CulturalGraph:
        """
        Each path dict contributes one edge: event --relation--> knowledge.
        Node identity is resolved by similarity so equivalent actions across
        different paths collapse into one node (the 'intersection' the paths
        share). Returns a CulturalGraph.
        """
        # Collect every action string that needs a node.
        raw_actions: list[str] = []
        for p in paths:
            head = (p.get("event") or "").strip()
            tail = (p.get("knowledge") or "").strip()
            if head:
                raw_actions.append(head)
            if tail:
                raw_actions.append(tail)

        # Pre-embed once (batch) if SBERT is active.
        embeddings = self.sim.embed(raw_actions) if raw_actions else None
        emb_index: dict[str, int] = {}
        if embeddings is not None:
            for i, a in enumerate(raw_actions):
                emb_index.setdefault(a, i)  # first occurrence

        nodes: dict[int, GraphNode] = {}
        canonical_emb: dict[int, object] = {}  # node_id -> embedding (or None)
        next_id = 0

        def _emb_for(action: str):
            if embeddings is None:
                return None
            return embeddings[emb_index[action]]

        def resolve_node(action: str) -> int:
            """Find an existing node this action merges into, else create one."""
            nonlocal next_id
            a_emb = _emb_for(action)
            best_id, best_sim = None, 0.0
            for nid, node in nodes.items():
                is_same, sim = self.sim.pairwise_is_same(
                    action, node.label, emb_a=a_emb, emb_b=canonical_emb.get(nid))
                if is_same and sim >= best_sim:
                    best_id, best_sim = nid, sim
            if best_id is not None:
                node = nodes[best_id]
                node.member_count += 1
                if _normalize(action) != _normalize(node.label) and action not in node.aliases:
                    node.aliases.append(action)
                return best_id
            # create new
            nid = next_id
            next_id += 1
            nodes[nid] = GraphNode(node_id=nid, label=action)
            canonical_emb[nid] = a_emb
            return nid

        edges: list[GraphEdge] = []
        for p in paths:
            head = (p.get("event") or "").strip()
            tail = (p.get("knowledge") or "").strip()
            if not head or not tail:
                continue
            h_id = resolve_node(head)
            t_id = resolve_node(tail)
            if h_id == t_id:
                continue  # self-loop from a near-dupe head/tail; skip
            edges.append(GraphEdge(
                src=h_id, dst=t_id,
                relation=(p.get("relation") or "related").strip(),
                llm_result=(p.get("llm_result") or "").strip(),
            ))

        location = next((p.get("location", "") for p in paths if p.get("location")), "")
        sub_topic = next((p.get("sub_topic", "") for p in paths if p.get("sub_topic")), "")

        graph = CulturalGraph(nodes=nodes, edges=edges,
                              location=location, sub_topic=sub_topic)
        stats = graph.merge_stats()
        logger.info("Reconstructed graph: %d nodes, %d edges from %d raw actions "
                    "(merge_ratio=%.2f).", stats["n_nodes"], stats["n_edges"],
                    stats["n_raw_actions"], stats["merge_ratio"])
        return graph

    # step 4: graph -> natural language 

    # Human-readable gloss for ATOMIC-style relations (from CCKG Table 1).
    _REL_GLOSS = {
        "xEffect": "which affects the person by",
        "xNext": "after which the person tends to",
        "xNeed": "which first requires",
        "oNext": "after which others tend to",
        "oEffect": "which affects others by",
    }

    def to_natural_language(self, graph: CulturalGraph,
                            max_chains: int = 6) -> str:
        """
        Emit a condensed prose contextualization. Strategy:
          - Lead with the cultural framing (location / sub_topic).
          - Call out CENTRAL nodes (the shared intersection points) explicitly,
            because their reoccurrence is the signal the supervisor wants surfaced.
          - Then describe the merged connective structure as a few sentences,
            walking edges out of the most central nodes rather than repeating each
            of the original (often overlapping) paths verbatim.
        """
        if not graph.nodes:
            return "No cultural reasoning structure could be reconstructed for this query."

        loc = graph.location or "the target culture"
        topic = graph.sub_topic or "this topic"
        adj = graph.adjacency()
        central = graph.central_nodes(top_k=3)
        deg = graph.degree()

        parts: list[str] = []
        parts.append(
            f"Cultural reasoning structure for {loc} regarding {topic}, "
            f"reconstructed from {graph.merge_stats()['n_raw_actions']} reasoning steps "
            f"merged into {len(graph.nodes)} distinct actions."
        )

        # Surface central / recurring nodes (the intersection points).
        recurring = [(nid, d) for nid, d in central if d > 1]
        if recurring:
            names = "; ".join(f'"{graph.nodes[nid].label}" (connects to {d} steps)'
                              for nid, d in recurring)
            parts.append(
                f"Central practices that multiple reasoning paths converge on: {names}. "
                f"These recurring actions are the connective core of this cultural topic."
            )

        # Describe outgoing structure from the most central nodes first.
        described_edges = 0
        seen_pairs = set()
        chain_sentences: list[str] = []
        ordered_nodes = [nid for nid, _ in sorted(deg.items(), key=lambda kv: kv[1], reverse=True)]
        for nid in ordered_nodes:
            if described_edges >= max_chains:
                break
            outs = adj.get(nid, [])
            if not outs:
                continue
            head_label = graph.nodes[nid].label
            for relation, dst in outs:
                if described_edges >= max_chains:
                    break
                if (nid, dst, relation) in seen_pairs:
                    continue
                seen_pairs.add((nid, dst, relation))
                gloss = self._REL_GLOSS.get(relation, "which connects to")
                tail_label = graph.nodes[dst].label
                chain_sentences.append(f"{head_label} {gloss} {tail_label}")
                described_edges += 1

        if chain_sentences:
            parts.append("Key culturally grounded connections: " +
                         "; ".join(chain_sentences) + ".")

        return " ".join(parts)

    # convenience: full pipeline 

    def reconstruct(self, paths: list[dict], max_chains: int = 6) -> dict:
        """
        Run build_graph + to_natural_language and return a single artifact dict:
            {
              "graph": CulturalGraph,
              "contextualization": str,        # the NL augmentation
              "central_nodes": [(label, degree), ...],
              "stats": {...merge stats...},
            }
        """
        graph = self.build_graph(paths)
        nl = self.to_natural_language(graph, max_chains=max_chains)
        central = [(graph.nodes[nid].label, d) for nid, d in graph.central_nodes(top_k=3)]
        return {
            "graph": graph,
            "contextualization": nl,
            "central_nodes": central,
            "stats": graph.merge_stats(),
        }


# Self-test (mock data, no deps) -- run: python graph_reconstruction.py   

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Mimic Agent A output: several paths that SHARE actions, so they should merge.
    demo_paths = [
        {"event": "looking for breakfast", "knowledge": "find soto ayam",
         "relation": "xNext", "llm_result": "If looking for breakfast, then find soto ayam.",
         "location": "Indonesia", "sub_topic": "breakfast"},
        {"event": "find soto ayam", "knowledge": "order breakfast from warung",
         "relation": "xNext", "llm_result": "If find soto ayam, then order from a warung.",
         "location": "Indonesia", "sub_topic": "breakfast"},
        {"event": "looking for breakfast", "knowledge": "have a breakfast of lontong",
         "relation": "xNext", "llm_result": "If looking for breakfast, then have lontong.",
         "location": "Indonesia", "sub_topic": "breakfast"},
        {"event": "have a breakfast of lontong", "knowledge": "choose a traditional side dish",
         "relation": "xNext", "llm_result": "If having lontong, then choose a side dish.",
         "location": "Indonesia", "sub_topic": "breakfast"},
        {"event": "choose a traditional side dish", "knowledge": "find soto ayam",
         "relation": "xNext", "llm_result": "If choosing a side dish, then find soto ayam.",
         "location": "Indonesia", "sub_topic": "breakfast"},
        # near-duplicate phrasing of an existing node -> should merge under Jaccard
        {"event": "searching for breakfast", "knowledge": "feel warmth and comfort",
         "relation": "xEffect", "llm_result": "If searching for breakfast, feel warmth.",
         "location": "Indonesia", "sub_topic": "breakfast"},
    ]

    recon = GraphReconstructor(threshold=0.6, use_sbert=False)  # Jaccard for demo
    result = recon.reconstruct(demo_paths)

    print("\n=== MERGE STATS ===")
    for k, v in result["stats"].items():
        print(f"  {k}: {v}")
    print("\n=== CENTRAL NODES ===")
    for label, d in result["central_nodes"]:
        print(f"  [{d}] {label}")
    print("\n=== NL CONTEXTUALIZATION ===")
    print(result["contextualization"])
