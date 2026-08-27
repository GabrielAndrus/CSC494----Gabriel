# Agent Interface & Data Contract

The orchestrator never talks to CulFiT or Tonga directly. It talks to two
adapter objects and one backend, each defined by a small method contract. To
swap in a different Agent A or Agent B later, you implement these same methods —
the orchestrator, the three topologies, the ablation harness, and the logging
all keep working unchanged.

This is the single source of truth for "what a new agent must provide."

---

## 1. The path object (shared data unit)

Everything flows as a list of **path dicts**. One path = one cultural reasoning
unit. Required keys:

| key          | type   | meaning                                                        |
|--------------|--------|----------------------------------------------------------------|
| `event`      | str    | the action/situation (CCKG triple head)                        |
| `knowledge`  | str    | the cultural knowledge invoked                                 |
| `relation`   | str    | relation type (e.g. xEffect, xNeed)                            |
| `llm_result` | str    | natural-language "empirical reasoning path" — what Agent B judges |
| `location`   | str    | cultural group / region (e.g. "Indonesia")                     |
| `sub_topic`  | str    | topic (e.g. "breakfast")                                        |

- `llm_result` is the field Agent B critiques, so a new Agent A **must** populate
  it meaningfully.
- `location` and `sub_topic` are used by Agent B's Group/Topic alignment checks.
- Extra keys are allowed and ignored; missing required keys degrade the critique
  (Agent B falls back to "Unknown").

---

## 2. Agent A contract (generator)

A new Agent A is **any object** exposing:

```python
class AgentA:
    def generate(self, query: str) -> list[dict]:
        """Produce the initial reasoning paths for `query`.
        Returns a list of path dicts (see section 1). May be empty."""

    def repair(self, repair_prompt: str) -> list[dict]:
        """Given a critique-bearing prompt, return a revised list of path dicts.
        Used only by the sequential topology."""
```

Contract notes:
- **Return parsed lists, not raw strings.** The orchestrator tolerates a raw
  string as a fallback, but a compliant agent returns `list[dict]`.
- `generate` and `repair` may call any model; they receive their backend at
  construction. They must not assume a specific provider.
- If generation fails, return `[]` rather than raising — the orchestrator treats
  empty as "abort this run" gracefully.

---

## 3. Agent B contract (critique engine)

A new Agent B is **any object** exposing:

```python
class AgentBEngine:
    def evaluate_payload_batch(self, paths: list[dict], ground_truth: dict) -> dict:
        """Critique a whole list of paths. Returns:
           {
             "approved": bool,              # True iff every path passed
             "mean_precision": float,       # 0.0–1.0, mean over paths
             "n_paths": int,
             "per_path": list[dict],        # per-path verdicts (for logging)
             "feedback": str,               # human-readable, fed into repair
           }"""

    def get_static_schema(self, query: str) -> dict:
        """Lightweight schema used by static & parallel topologies (no loop).
           Any JSON-serializable dict describing the evaluation dimensions."""
```

The `ground_truth` dict it receives:

| key              | type      | meaning                              |
|------------------|-----------|--------------------------------------|
| `location`       | str       | reference cultural group             |
| `sub_topic`      | str       | reference topic                      |
| `verified_points`| list[str] | reference cultural knowledge points  |

Contract notes:
- `approved` and `mean_precision` are what the orchestrator branches on and logs;
  they must be present.
- `feedback` is passed verbatim into Agent A's `repair` prompt, so make it
  actionable.
- `per_path` is optional for correctness but recommended — it's what makes the
  ablation logs rich.

---

## 4. Backend contract (model access)

Both agents and the arbiter receive an `LLMBackend`:

```python
class LLMBackend:
    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        """messages are OpenAI-style [{'role','content'}, ...]; returns a string."""
```

Implementations: `MockBackend`, `APIBackend`, `LocalBackend` (in `llm_backend.py`).
A new agent should depend only on `.chat()`, never on a concrete backend.

---

## 5. How to actually swap an agent

1. Write an adapter class honoring section 2 (Agent A) or section 3 (Agent B).
2. If it wraps a new repo, add that repo's path + module loading to
   `bootstrap.py` (the one place that knows where repos live).
3. Construct it in `run_ablation.py` / `run_local.py` in place of the current
   `AgentA(...)` / `AgentBCritiqueEngine(...)`.
4. Run `python -m orchestration.check_setup` then the smoke test. Nothing in
   `orchestrator.py` changes.

The interface is intentionally minimal: two methods for A, two for B, one for the
backend. If a future agent needs richer interaction (e.g. multi-round critique
state), extend the contract here first, then update both the adapter and the
orchestrator together — and re-document in this file.
