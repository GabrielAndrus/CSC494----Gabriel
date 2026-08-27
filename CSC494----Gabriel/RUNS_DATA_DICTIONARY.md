# Ablation Runs — Data Dictionary

Reference for every variable produced by the ablation harnesses:

```bash
python -m orchestration.run_ablation --mode mock   # offline / API
python -m orchestration.run_local   --model ...     # cluster GPU
```

Both write the same record shape; differences are noted inline (the main one:
`run_local.py` records omit the `llm_calls` block — see §1.1).

Each invocation writes two files into `runs/`:

| File | One record per | Purpose |
|------|----------------|---------|
| `ablation_<timestamp>.jsonl` | `(query × topology)` | Granular per-run trace — the backing data |
| `ablation_<timestamp>_summary.json` | `topology` | Aggregated comparison table — the headline view |

`<timestamp>` is `YYYYMMDD_HHMMSS` at launch (e.g. `ablation_20260616_135829`).
With Q queries and 3 topologies (`static`, `parallel`, `sequential`), the JSONL
has `Q × 3` lines and the summary has 3 keys.

> **Metrics are deliberately raw.** The harness logs signal, not a single
> committed headline number. The summary is a convenience aggregate; choose what
> to actually report after inspecting runs.

---

## 1. JSONL record (`ablation_<timestamp>.jsonl`)

One JSON object per line. Top-level fields:

| Field | Type | Meaning |
|-------|------|---------|
| `query` | str | The evaluation item (e.g. `"breakfast"`). |
| `location` | str | Cultural group / region for this item (e.g. `"Indonesia"`). |
| `topology` | str | Experimental condition: `static`, `parallel`, or `sequential`. This is the variable being ablated. |
| `n_final_paths` | int | Number of knowledge paths in Agent A's output after the run finished. |
| `llm_calls` | object | Backend call counts per agent — the cost signal. **`run_ablation.py` (mock/api) only**; `run_local.py` cluster records omit this. See §1.1. |
| `trace` | object | The substance of the run: verdicts, precision, loops, repairs. See §1.2. |

### 1.1 `llm_calls`  (mock/api harness only)

Present in `run_ablation.py` output; absent from `run_local.py` (cluster). When
present:

| Field | Type | Meaning |
|-------|------|---------|
| `agent_a` | int \| null | Number of backend calls Agent A made this run. `null` if the backend doesn't expose `call_count`. |
| `agent_b` | int \| null | Number of backend calls Agent B made this run. |

> **Exact vs. averaged.** These are *exact per-run* counts. A sequential run can
> show e.g. 9 Agent B calls (3 critique passes × 3 dimensions), while the summary
> reports an *average* across queries. For cost analysis use the JSONL, not the
> summary. On the cluster (no `llm_calls`), derive cost from the trace:
> Agent B calls ≈ Σ over loops of (n_paths × 3 dimensions).

### 1.2 `trace`

| Field | Type | Present in | Meaning |
|-------|------|-----------|---------|
| `loops` | int | all | Critique→repair iterations. Always `0` for `static` (single pass, no repair). |
| `repairs` | int | all | Times Agent A regenerated in response to critique. `0` for `static`. |
| `final_approved` | bool | all | Whether Agent B approved the **final** output. |
| `final_mean_precision` | float | all | Agent B's mean precision on the final paths, `0.0`–`1.0`. |
| `critique` | object | all | The **final** critique object. See §1.3. |
| `iterations` | list[object] | `sequential` only | Per-loop history; each entry carries that loop's critique so you can trace how precision moved across repairs. See §1.4. |

### 1.3 `critique` (a batch critique object)

Returned by Agent B's `evaluate_payload_batch`. Describes the verdict over the
**whole list** of paths.

| Field | Type | Meaning |
|-------|------|---------|
| `approved` | bool | `True` iff **every** path passed (i.e. all per-path `approved` are `True`). |
| `mean_precision` | float | Mean of per-path `precision_score`, `0.0`–`1.0`. `0.0` if no paths. |
| `n_paths` | int | Number of paths evaluated. |
| `per_path` | list[object] | One verdict per path. See §1.5. |
| `feedback` | str | Human-readable critique. When not approved, lists offending paths + reasons; this string is fed verbatim into Agent A's repair prompt. When approved: a "all passed" message. |

### 1.4 `iterations[]` (sequential only)

Each entry is one loop of the deliberation. The exact shape mirrors the
per-iteration record the orchestrator accumulates; the key content is the
`critique` for that loop (same shape as §1.3), letting you plot precision vs.
loop index and see how repairs moved the needle.

### 1.5 `per_path[]` (a single-path verdict)

Returned by Agent B's `evaluate_single_path`, with the path text attached.

| Field | Type | Meaning |
|-------|------|---------|
| `path` | str | The natural-language reasoning path that was judged (Agent A's `llm_result`). |
| `approved` | bool | `True` iff `precision_score == 1.0` (all three dimensions passed). |
| `precision_score` | float | Fraction of the 3 dimensions that passed: `0.0`, `0.333`, `0.667`, or `1.0`. |
| `verdicts` | object | Per-dimension `Yes`/`No`. Keys: `"Cultural Group Alignment"`, `"Topic Alignment"`, `"Knowledge Path Alignment"`. |
| `feedback` | str | Per-path reason. On failure, names which dimension(s) failed and the model's raw response; on success, a validation message. |

#### The three critique dimensions

| Dimension | What Agent B checks |
|-----------|---------------------|
| **Cultural Group Alignment** | Does the path's `location` match the ground-truth cultural group? |
| **Topic Alignment** | Does the path's `sub_topic` match the ground-truth topic? |
| **Knowledge Path Alignment** | Does the path's `llm_result` align with the ground-truth `verified_points`? |

Each dimension is a separate LLM call returning a verdict; `"Yes"` in the
response counts as a pass. `precision_score` = passes / 3.

---

## 2. Summary record (`ablation_<timestamp>_summary.json`)

A single JSON object keyed by topology (`static`, `parallel`, `sequential`).
Each value aggregates that topology's runs across all queries.

> **Two harnesses emit slightly different summary keys.** `run_ablation.py`
> (mock/api) and `run_local.py` (cluster GPU) share most fields but differ on
> call-count reporting. The tables below mark which is which. Your cluster
> results come from `run_local.py`.

**Common to both harnesses:**

| Field | Type | Meaning |
|-------|------|---------|
| `n_runs` | int | Number of `(query, topology)` runs aggregated (= number of queries). |
| `approval_rate` | float | Fraction of runs where `final_approved` was `True`, `0.0`–`1.0`. |
| `avg_mean_precision` | float | Average of `final_mean_precision` across runs. |
| `avg_loops` | float | Average loop count. |
| `avg_repairs` | float | Average repair count. |

**`run_ablation.py` (mock/api) only:**

| Field | Type | Meaning |
|-------|------|---------|
| `avg_agent_b_calls` | float | Average Agent B backend calls per run. (Cost signal; `run_local.py` does not emit this.) |

> The precision key is `avg_mean_precision` (not `mean_precision`). `run_local.py`
> records carry no `llm_calls` block and its summary has no call-count field, so
> for cluster cost analysis count calls from the trace (loops × paths × 3
> dimensions) rather than expecting a logged total. `inspect_run.py` prints
> whatever keys are present, so it stays correct across both harnesses.

---

## 3. The path dict (Agent A output — context for `per_path.path`)

For reference, each path Agent A produces (and Agent B judges) has all six keys:

| Field | Type | Meaning |
|-------|------|---------|
| `event` | str | The action/situation (CCKG triple head). |
| `knowledge` | str | The cultural knowledge invoked. |
| `relation` | str | ATOMIC-style relation (e.g. `xEffect`, `xNeed`). Note: stored as `relation`, though Agent A's raw LLM output calls it `relation_type` before parsing. |
| `llm_result` | str | Natural-language reasoning path — the field Agent B critiques. |
| `location` | str | Cultural group / region. |
| `sub_topic` | str | Topic. |

---

## 4. Reading a run quickly

```bash
# Human-readable breakdown of the most recent run:
python inspect_run.py

# A specific run (timestamp or full path both work):
python inspect_run.py 20260616_135829
python inspect_run.py runs/ablation_20260616_135829.jsonl

# Just the summary table:
python inspect_run.py --summary-only

# Read the whole thing in pretty JSON: 
jq . runs/ablation_local__...143058.jsonl
```
