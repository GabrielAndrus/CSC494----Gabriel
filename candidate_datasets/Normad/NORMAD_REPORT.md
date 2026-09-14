# NormAd candidate-dataset report

**Date:** 2026-09-14
**Source:** [akhilayerukola/NormAd](https://huggingface.co/datasets/akhilayerukola/NormAd) (HF dataset, 2,615 rows, single `train` split, CSV-backed)
**What this covers:** an 18-row NormAd sample, a documented mapping onto the PluralTree eval-set schema, a complexity profile against the current GlobalCultureQA baseline, and a mock-backend run of the orchestrator over the normalized data.

**What this does NOT cover:** a real quality evaluation (actual model answers judged for correctness). That needs either a Gemini API key or the Narval GPU cluster — both flagged as open asks at the bottom of this report, per project convention of keeping cluster/credential steps under direct user oversight.

---

## 1. How the sample was pulled

Fetched via the public `datasets-server.huggingface.co` `/rows` API (no HF login needed; the dataset is public). `/info` first confirmed the row schema:

```
ID, Country, Background, Axis, Subaxis, Value, Rule-of-Thumb, Story, Explanation, Gold Label
```

The 2,615 rows are laid out in three contiguous blocks by `Gold Label` (`yes` ≈ rows 0–1300, `no` ≈ 1300–1950, `neutral` ≈ 1950–2615), so a naive `offset=0,length=18` pull would have been all-`yes` and mostly one country. Instead, 18 rows were hand-picked at spread-out offsets (18 separate single-row fetches) to get:

- **13 distinct countries**: Egypt, Canada, South Korea, France, Australia, Cyprus, South Sudan, Sweden, Thailand, Japan, Croatia, Papua New Guinea, Laos.
- **All three gold labels**: 7 `yes`, 6 `no`, 5 `neutral`.
- **All rows share `Axis = "Etiquette"`** — this wasn't a filtering choice, every row sampled across the full offset range came back as `Etiquette` with a varying `Subaxis` (`basic_etiquette`, `tipping`, `gifts`/`gift_giving`, `eating`, `visiting`). NormAd's public card describes additional axes (e.g. social-norm categories beyond etiquette); none turned up in this 18-row spread, so this sample should not be read as covering NormAd's full axis diversity — see §3.

Raw rows: [`normad_raw_sample.jsonl`](normad_raw_sample.jsonl) (18 lines, untouched HF fields). This is kept separately from the normalized file so every mapping decision below is auditable against the original.

## 2. Schema mapping

Target schema is the one `orchestration/build_eval_set.py` already produces for GlobalCultureQA:

```
{query, location, sub_topic,
 ground_truth: {location, sub_topic, verified_points: [str, ...]}}
```

| Target field | NormAd source | Transform |
|---|---|---|
| `query` | `Story` | verbatim — already a self-contained scenario ending in a yes/no acceptability question |
| `location` | `Country` | `snake_case` → `Title Case With Spaces` (e.g. `south_sudan` → `South Sudan`) |
| `sub_topic` | `Subaxis` | same title-casing (e.g. `basic_etiquette` → `Basic Etiquette`) |
| `ground_truth.verified_points` | `Rule-of-Thumb` + `Gold Label` + `Explanation` | see below — this is the one non-trivial decision |

`ground_truth.verified_points` construction (implemented in [`normalize_normad.py`](normalize_normad.py)):

1. **`Rule-of-Thumb` verbatim** — for `yes`/`no` rows only (see the neutral-label caveat in §3, this is deliberately dropped for `neutral` rows).
2. **One derived sentence encoding the Gold Label**, e.g. *"In Egypt, the behavior described is considered socially acceptable given this norm."* NormAd's label itself (yes/no/neutral) isn't a "knowledge point" in HF's data — it's a classification target — so this sentence is synthesized, not copied, to give Agent B something to check the candidate answer's conclusion against, mirroring how GlobalCultureQA's `grounded_answer.answer` prose contributes the "conclusion" fact.
3. **Each sentence of `Explanation`**, split on `.`/`!`/`?`, deduplicated against the Rule-of-Thumb line. `Explanation` is NormAd's human-written justification tying the specific story to the specific rule — it's the closest analogue to GlobalCultureQA's hand-decomposed `knowledge_points`, just not pre-split into atoms by the original authors, so this script does the splitting.

Original fields are preserved losslessly per record under an extra `_normad_source` key (`id`, `country_raw`, `axis`, `subaxis`, `value`, `gold_label`, `background`). Per [`AGENT_CONTRACT.md`](../../orchestration/AGENT_CONTRACT.md) §1, extra keys on a record are allowed and ignored by the orchestrator/agents — this key is for traceability only, e.g. to look up which original HF row a given `verified_points` list came from, or to re-derive `Gold Label` for the by-label breakdown in §5.

Output: [`normad_eval_set.jsonl`](normad_eval_set.jsonl), 18/18 rows mapped successfully (no rows dropped).

## 3. Nuances and what's lost in this mapping

This is the section to read before trusting any number below.

**a) NormAd is a binary/ternary judgment task; GlobalCultureQA is open QA.** GlobalCultureQA asks "what/how/why" and grades a free-text answer against several independent knowledge points. NormAd asks "is this acceptable?" and the entire ground truth pivots on ONE rule. Reusing the same `verified_points` container works mechanically (Agent B's precision/recall check doesn't care what the strings mean), but it quietly repurposes a "how much of the rich answer did you cover" metric into a "did you correctly restate the one rule and its verdict" metric. A model that gets the yes/no verdict right but never repeats the rule in the rule's exact phrasing could score low; a model that pattern-matches "yes, socially acceptable" boilerplate could score deceptively well. This is a task-type mismatch, not just a formatting one — a report finding of "PluralTree does/doesn't help on NormAd" would need this caveat attached everywhere.

**b) `neutral` rows are adversarial pairings, and this was caught mid-normalization.** NormAd's `neutral` label doesn't mean "socially ambiguous behavior" — it means the annotators deliberately paired a story with a `Rule-of-Thumb`/`Background` from a country that does **not** govern that story, to test whether a system notices the rule doesn't apply. Concretely, the raw row for the Croatia/neutral example has `Rule-of-Thumb = "It is correct to express gratitude for a meal before beginning to eat"` attached to a story about chopsticks left upright in a rice bowl — a rule with zero bearing on the story, by design. The first draft of `normalize_normad.py` included `Rule-of-Thumb` unconditionally, which would have handed Agent B a **wrong** fact to grade candidate answers against for 5/18 rows. Fixed by omitting `Rule-of-Thumb` from `verified_points` when `Gold Label == "neutral"` (kept only the derived ambiguity sentence + `Explanation` sentences, which correctly describe the mismatch). **If this dataset is extended past 18 rows, re-check this logic against a bigger neutral sample** — the fix generalizes from one clear case, but wasn't verified against NormAd's full neutral population.

**c) `location` means something different for `neutral` rows.** For `yes`/`no` rows, `Country` is the story's actual cultural setting. For `neutral` rows, `Country` is the country the *distractor* rule was drawn from — the story's real cultural context is deliberately unspecified in NormAd. So in the normalized file, `"location": "Japan"` on a neutral row does not mean "this story is about Japan"; it means "Japan's norms were the (inapplicable) foil offered." Nothing in the schema marks this distinction — a downstream consumer that assumes `location` is always ground-truth-relevant (which is true for the other 13/18 rows and for 100% of GlobalCultureQA) will misread these 5.

**d) Single rule vs. multi-fact ground truth changes what "complexity" means here.** See §4 — NormAd's `verified_points` count is structurally capped low (a rule + a verdict + 1-2 explanation sentences) no matter how long or elaborate the `Story` is, because there is fundamentally one norm being tested. GlobalCultureQA's point count scales with how much the source Wikipedia article said. A shared "≥8 verified points" complexity threshold (from `analyze_eval_complexity.py`) will never fire on NormAd as normalized here — not because NormAd items are simple, but because the schema conflates "richness of ground truth" with "number of independent facts," and NormAd's richness lives in the *story*, not in a decomposed answer.

**e) `Subaxis` slugs make a thin `sub_topic` taxonomy compared to GlobalCultureQA's free-text topics.** GlobalCultureQA's `topic` values are article-level and highly specific (e.g. "Jirga", "Summer Festival, Albania"). NormAd's `Subaxis` values are a small closed vocabulary (`basic_etiquette`, `tipping`, `gifts`, `eating`, `visiting`, …) shared across many countries — in this 18-row sample there are 13 locations but only 6 distinct `sub_topic` values. Group/Topic-alignment checks in Agent B (which key off `location`/`sub_topic` per `AGENT_CONTRACT.md` §3) will see much less topic diversity per row than they do on GlobalCultureQA.

**f) Naming collision to flag explicitly, not a data problem but a real gotcha:** NormAd contains a country literally named **"Tonga"** (the Pacific island nation) in its `yes`-label block. This project's Agent A is built on an unrelated codebase nicknamed "Tonga" (the CCKG generator, see `bootstrap.py`/`PROJECT_HANDOFF.md`). No Tonga(-country) rows made it into this 18-row sample, but a larger pull will include some — worth a second's pause in logs/discussion so "Tonga" isn't misread as referring to the codebase.

**g) Sample size and curation.** 18 rows is enough to validate the mapping and smoke-test the pipeline, not enough to say anything statistically about NormAd as a benchmark. The 18 were also hand-picked for label/country diversity rather than drawn uniformly at random — appropriate for this exploratory pass, but a real candidate-dataset decision should re-sample at n≥100 with a documented random seed (mirroring `build_eval_set.py`'s `--seed` pattern) before any performance number is taken seriously.

**h) `Background` (multi-bullet country etiquette notes) was deliberately NOT split into extra `verified_points`.** It's the closest NormAd field to GlobalCultureQA's `cultural_knowledge` (broad source material) rather than to `grounded_answer_knowledge_points` (answer-specific atoms) — most of its bullets are irrelevant to the one story being judged, and injecting them as "verified points" would reward a candidate answer for mentioning unrelated facts. It's preserved under `_normad_source.background` for anyone who wants it later.

## 4. Complexity profile vs. the current baseline

Ran `orchestration/analyze_eval_complexity.py` (stdlib-only, no model calls) on the normalized NormAd sample, and on an equal-size (n=18, same `--seed 13` as the project default) draw from GlobalCultureQA for a fair side-by-side:

| Metric | NormAd (n=18, this sample) | GlobalCultureQA (n=18, seed=13) |
|---|---:|---:|
| Question length, mean (words) | **35.8** | 24.9 |
| Question length, range | 21–66 | 19–35 |
| Verified points per item, mean | **3.4** | 8.4 |
| Verified points per item, range | 2–4 | 5–11 |
| Verified-points total words, mean | 61.3 | 108.5 |
| Vocabulary size (unique words across all verified points) | 312 | 1,089 |
| Unique `sub_topic` values | 6 | 18 |
| "Sufficiently complex" (≥8 VP AND ≥30-word question, current project threshold) | **0 / 18 (0%)** | 3 / 18 (16.7%) |

Full machine-readable profile: [`normad_complexity.json`](normad_complexity.json).

**Reading this:** NormAd stories are *longer* than GlobalCultureQA questions (narrative scenario-setting takes words) but resolve to a *much shallower* ground truth (a single rule + verdict, vs. a decomposed multi-fact answer). Under the project's original complexity threshold (VP≥8 AND Q_len≥30, tuned for GlobalCultureQA), **zero NormAd items as normalized here would be classified "sufficiently complex."** This isn't evidence NormAd items are cognitively easy — the "is this behavior acceptable given this one specific norm" judgment can require real cultural reasoning — it's evidence that **the original metric measures ground-truth decomposition depth, which is the wrong axis for a single-rule judgment task.**

**Update (2026-09-14): `analyze_eval_complexity.py` was augmented, not replaced,** to fix exactly this. Rather than reduce "cultural complexity" to one smarter scalar (deliberately rejected — see the script's new module docstring, "WHY THIS WAS AUGMENTED," for the reasoning: complexity here is multi-dimensional and partly subjective, and one number just relocates the same judgment calls one level down while looking more authoritative than it is), the script now auto-detects task shape from query phrasing and reports **two classification views side by side** — the original `open_qa` rule (unchanged, still correct for GlobalCultureQA) and a new `judgment` rule for NormAd-shaped data: *narrative_beats ≥ 2 (a multi-step scenario) AND (lexical_specificity ≥ this batch's own median OR the ground truth carries a hedging/contested-ness marker)*. Four new schema-agnostic signals feed this (`narrative_beats`, `action_density`, `lexical_specificity`, `ambiguity_flag`) — none hardcode anything NormAd-specific (e.g. `ambiguity_flag` looks for generic hedge language in the text, not NormAd's `Gold Label` field), so the judgment view should generalize to whatever the next judgment-shaped candidate dataset turns out to be.

Re-run with the augmented script:

| View | Threshold | NormAd (n=18) | GlobalCultureQA (n=18, seed=13) |
|---|---|---:|---:|
| `open_qa` (original, unchanged) | VP≥8 AND Q_len≥30 | 0 / 18 (0%) | 3 / 18 (16.7%) — unchanged from §4's original numbers, confirming the rewrite is backward-compatible |
| `judgment` (new) | beats≥2 AND (spec≥median OR ambiguous) | **14 / 18 (77.8%)** | 0 / 18 (0%) |

The `judgment` view finds real structure in NormAd that the `open_qa` view was blind to (78% vs. 0%) — largely because most sampled stories here bundle ≥2 narrative beats (someone arrives, is offered something, then acts) and invoke specific enough norms (median `lexical_specificity` 0.578) to clear the adaptive bar. Symmetrically, `judgment` correctly reads GlobalCultureQA as 0% complex — its questions are single-beat ("what/how/why is X"), so the judgment view isn't secretly a better universal metric, it's a different, task-matched one. Full per-item signals (`narrative_beats`, `action_density`, `lexical_specificity`, `ambiguity_flag`) for every one of the 18 rows are in the refreshed [`normad_complexity.json`](normad_complexity.json). Treat the judgment thresholds as a first pass calibrated on 18 rows, not a settled number — the script's docstring says as much and asks for retuning once real model performance on judgment items is observed.

## 5. Pipeline run

**What ran:** [`run_pipeline_normad_mock.py`](run_pipeline_normad_mock.py), which feeds `normad_eval_set.jsonl` through `orchestration.run_ablation`'s existing `build_orchestrator("mock", ...)` (same `MockBackend` + same canned `mock_agent_a_responder`/`mock_agent_b_responder` heuristics already in the repo — no pipeline code was modified) across all three topologies (static/parallel/sequential). Environment: the project's own `venv/` locally (Python 3.14), no GPU, no API key, no network calls once the dataset was already fetched.

Setup check passed first (`python -m orchestration.check_setup` → `PASS`), confirming this machine already has both vendored repos (`CulFiT/`, `Cultural_Commonsense_Knowledge_Graph/`) and the project venv correctly wired, so this required no environment changes.

**Result:** all 18 items × 3 topologies = 54 runs completed without errors — the orchestrator, Agent A adapter, and Agent B critique engine all accept NormAd-shaped `query`/`location`/`sub_topic`/`ground_truth.verified_points` records without any schema friction. Full per-run trace: `normad_mock_run_20260914_160751.jsonl`; aggregate: `normad_mock_run_20260914_160751_summary.json`.

```json
{
  "by_topology": {
    "static":     {"n_runs": 18, "approval_rate": 0.0, "avg_mean_precision": 0.667, "avg_loops": 0, "avg_repairs": 0},
    "parallel":   {"n_runs": 18, "approval_rate": 0.0, "avg_mean_precision": 0.667, "avg_loops": 1, "avg_repairs": 0},
    "sequential": {"n_runs": 18, "approval_rate": 0.0, "avg_mean_precision": 0.667, "avg_loops": 3, "avg_repairs": 2}
  },
  "by_gold_label_static_only": {
    "yes":     {"n": 7, "approval_rate": 0.0},
    "no":      {"n": 6, "approval_rate": 0.0},
    "neutral": {"n": 5, "approval_rate": 0.0}
  }
}
```

**How to read this — and its real limit:** this is a **structural/integration check, not a quality evaluation.** `MockBackend` returns the same two or three canned strings regardless of what query it's asked (see `orchestration/run_ablation.py`'s `mock_agent_a_responder`/`mock_agent_b_responder` — they're hardcoded around an "Indonesian breakfast / Bubur Ayam" example and a "does the path mention history/tradition/culinary" keyword check). That's why `avg_mean_precision` is a flat 0.667 across every topology and every NormAd gold label — the mock backend is, by construction, blind to NormAd's actual content, so approval rate cannot and should not track `yes`/`no`/`neutral` here. What this run *does* establish: the schema mapping in §2 is load-bearing correctly — the orchestrator ran all three topologies on all 18 NormAd-derived items to completion, Agent A generated paths, Agent B critiqued them against `location`/`sub_topic`/`verified_points`, and the sequential loop's repair/collapse-guard logic engaged normally (`avg_repairs=2` before hitting `max_loops=3`, same failure-to-converge pattern seen on the existing 2-item mock fixture).

**What did NOT run, and why:** an actual quality read (does the real model get NormAd's yes/no/neutral judgments right, does agentic deliberation help or hurt on it) needs a real LLM behind Agent A/B. Per `orchestration/llm_backend.py`, that means either:
- **`local` mode** — Llama/Qwen on a GPU, which per `orchestration/NARVAL.md` requires the Digital Research Alliance of Canada's Narval cluster (SSH login with the project's `def-enaskt` account, then `salloc`/`sbatch` commands on that remote system), or
- **`gemini` mode** — just needs a `GEMINI_API_KEY` and runs anywhere (no cluster) per `llm_backend.py`'s `GeminiBackend`; no key exists on this machine currently, and `run_ablation.py`'s `build_orchestrator()` didn't actually expose `gemini` as a mode until this session (it only had `mock`/`api`/`local`) — that gap is now fixed (`--mode gemini` works in both `orchestration/run_ablation.py` and `run_pipeline_normad_mock.py --mode gemini`), so a real run is one API key away. See §8.

Both require the user directly — see §7/§8.

## 6. Files in this folder

| File | What it is |
|---|---|
| `normad_raw_sample.jsonl` | 18 untouched rows pulled from HF's `datasets-server` API |
| `normalize_normad.py` | the mapping script (§2/§3), run as `python normalize_normad.py --in normad_raw_sample.jsonl --out normad_eval_set.jsonl` |
| `normad_eval_set.jsonl` | the 18 normalized records in PluralTree's eval-set schema |
| `normad_complexity.json` | full `analyze_eval_complexity.py` output for the normalized set (§4) |
| `run_pipeline_normad_mock.py` | the integration runner (§5); `--mode mock` (default) or `--mode gemini` for a real run once a key is set (§8) |
| `normad_mock_run_20260914_160751.jsonl` / `..._summary.json` | its output |
| `NORMAD_REPORT.md` | this file |

## 7. Open asks (need you, not me)

1. **A real quality run needs a backend I can't stand up myself.** Either:
   - You provide a `GEMINI_API_KEY` (project already has a `GeminiBackend` wired up in `orchestration/llm_backend.py` for exactly this — no cluster needed, runs locally), **or**
   - We follow `orchestration/NARVAL.md`'s Step 0–3 on the actual Narval cluster, which needs **you to log into the Digital Research Alliance of Canada** (`def-enaskt` account) and **run the `salloc`/`sbatch`/`module load` commands yourself in that remote terminal** — I don't have credentials or SSH access to do this.
2. Let me know if you want the sample re-drawn larger (e.g. n=100, seeded/random rather than hand-picked) once the mapping approach in §2/§3 looks right to you — the current 18 were chosen for diversity to stress-test the normalization logic, not as a final candidate set.
3. §3(b)'s neutral-label fix (omitting `Rule-of-Thumb` for `neutral` rows) was verified against exactly the 5 neutral rows in this sample — worth a second look before trusting it at scale.

## 8. Next steps: running a real (non-mock) test pipeline

Three paths, ranked by how soon they'd give a real signal. All three end with a credential/login/terminal step that has to be you, per project convention — everything before that step is already done or ready to go.

### 8a. Gemini mode (fastest — no cluster, runs on this laptop)

This machine already has the project venv (`venv/`) with `pandas` installed; it's missing `openai` and `python-dotenv`, and `run_ablation.py` only just gained a `gemini` mode this session (previously it silently only supported `mock`/`api`/`local` even though `llm_backend.py`'s `GeminiBackend` already existed).

1. **Get a key** — you'll need a Gemini API key (Google AI Studio, or whatever the project's existing shared Google Cloud project uses — `REPRODUCE.md` mentions Gemini credits are shared with another project, "use conservatively and report usage"). This is a credential step, so it's on you.
2. **Install the two missing packages:**
   ```bash
   venv/bin/pip install openai python-dotenv
   ```
3. **Set the key.** Heads up: despite `smoke_test_gemini.py`'s docstring saying it reads from a `.env` file, nothing in the codebase actually calls `load_dotenv()` yet — `GeminiBackend` reads `os.environ` directly. So either export it for real:
   ```bash
   export GEMINI_API_KEY=your-key-here
   ```
   or tell me and I'll add a two-line `load_dotenv()` call to the entrypoints so a `.env` file (already gitignored) works as the docs imply.
4. **Verify the backend before spending any real calls:**
   ```bash
   venv/bin/python3 -m orchestration.smoke_test_gemini
   ```
   Expect three `[ok]` lines. If it fails, the error message tells you which of key/package/network is the problem.
5. **Run the real NormAd pipeline:**
   ```bash
   venv/bin/python3 candidate_datasets/Normad/run_pipeline_normad_mock.py \
       candidate_datasets/Normad/normad_eval_set.jsonl --mode gemini
   ```
   This is 18 items × 3 topologies = 54 runs, each making several Gemini calls (Agent A generation + Agent B critique, more on the sequential topology if repairs happen) — modest but not free; a good first check before scaling to n=100. Compare the `by_gold_label_static_only` approval rates this produces against the flat 0.0/0.0/0.0 mock baseline in §5 — that comparison is the actual "does the pipeline get NormAd right" answer this report couldn't give yet.
6. Same `--mode gemini` flag now also works on `orchestration/run_ablation.py` directly, for the project's own 2-item fixture, if you want the fastest possible non-mock sanity check first:
   ```bash
   venv/bin/python3 -m orchestration.run_ablation --mode gemini
   ```

### 8b. Local mode — HF transformers (as built) vs. Ollama (as you guessed)

Checked both on this machine:

- **As currently built, `local` mode means HuggingFace `transformers` on a GPU** (`orchestration/llm_backend.py`'s `LocalBackend`, driven by `run_local.py`). Neither `torch` nor `transformers` is installed in this venv, and this Mac has 18GB unified memory — Llama-3.1-8B in bf16 needs ~16GB for weights alone, before activations, so running the actual pipeline's real model this way on this machine is impractical even after installing the packages. This mode is built for Narval's GPUs, not a laptop — see 8c.
- **Ollama is already installed on this Mac** (`/opt/homebrew/bin/ollama`, v0.32.1) but not currently running, and — importantly — **there is no `OllamaBackend` in `llm_backend.py` yet.** `REPRODUCE.md`/`NARVAL.md` don't mention Ollama; your instinct to double check was right; it isn't there. Given the 18GB RAM ceiling above, Ollama's quantized GGUF models (e.g. `llama3.1:8b` at ~4.7GB for a Q4 quant, vs. ~16GB unquantized) are actually the realistic way to get a real *local* model on this specific machine without a cloud key or the cluster. Building this would mean adding a small `OllamaBackend(LLMBackend)` class to `llm_backend.py` that POSTs to `http://localhost:11434/api/chat` — same `.chat(messages)` contract as every other backend (`AGENT_CONTRACT.md` §4), so no orchestrator/agent code would change. **I haven't built this — say the word and I will**, it's a scoped, low-risk addition (new class + one line in `make_backend()`), but it's a real pipeline change so I wanted to flag it as a decision rather than just doing it.

### 8c. Narval / Digital Research Alliance of Canada — the authoritative run, needs you end to end

This is the one that reproduces the project's actual reported numbers (`REPRODUCE.md`'s baseline: `culfit_baseline` F1 0.340 at n=100) and is the only path to a full Llama-3.1-8B run. Per your original instruction, this needs you throughout — I cannot SSH into an institutional cluster account on your behalf. Concretely, from `orchestration/NARVAL.md`:

1. **You log into the Digital Research Alliance of Canada** (Narval, account `def-enaskt`) and get the repo onto `~/projects/def-enaskt/hhpfiona/CSC494` (git pull, per `GIT_SETUP.md`'s laptop → GitHub → Narval flow).
2. **You run `NARVAL.md` Step 0 on a login node** — a one-time venv + SBERT pre-cache + `hf auth login` + Llama-3.1-8B download (~16GB, gated model, needs your HF token accepted for that model's license). Compute nodes are air-gapped, so this step has to happen here or not at all.
3. **You run Step 1** (`python -m orchestration.check_setup`) and **Step 2**, a cheap `salloc` smoke test (1 query, 1 topology, 1 loop) — this is the "fails in seconds, not after a 12h queue" check.
4. **You submit Step 3**, the real batch job (`sbatch orchestration/run_ablation.slurm`, one `ARMS=` value per job since both arms don't fit a 12h opportunistic slot).
5. Once results land in `runs/` on Narval, they come back to you via `scp` (per `PROJECT_HANDOFF.md`'s Windows/no-rsync note) — at that point I can pick back up and help analyze them (e.g., running `analyze_eval_complexity.py` against whatever comes back, including on the NormAd `--queries` set if you want that comparison run on the cluster too — see the drift note below on `analyze_taskeval`).

If you want to try NormAd specifically on Narval, the one piece to add there is a `run_generate.py --queries candidate_datasets/Normad/normad_eval_set.jsonl` pass — the file already loads any JSONL matching the shared schema (`load_items()` in `run_generate.py`), so the NormAd file from this report should work as-is once you're at that step.

**One thing worth fixing before you rely on it: `REPRODUCE.md`'s §3 example commands have drifted from the actual scripts.** Checked while writing this section — `REPRODUCE.md` shows `run_generate.py --arm ... --eval ... --out ...` and `run_judge.py --gen ... --judge-model ... --out ...`, but the real argparse in those files today is `run_generate.py --queries --systems --model` (no `--arm`, no `--eval`, no `--out` — it writes `runs/answers_<timestamp>.jsonl` on its own) and `run_judge.py --answers --judge_model` (underscore, no hyphen; also no `--out`). `REPRODUCE.md` also calls `python -m orchestration.analyze_taskeval`, which doesn't exist anywhere in `orchestration/` — there's a `run_task_eval.py` / `task_eval.py` pair that looks like it might be a consolidated one-pass alternative to the generate/judge split, but I haven't traced whether it's the intended replacement or a separate, older path. I didn't fix `REPRODUCE.md` itself (out of scope for this ask, and I'd rather you confirm which script is actually current before I edit the project's main reproduction guide) — but given `REPRODUCE.md`'s own "golden rule" is verifying before spending compute, **run `--help` on `run_generate.py`/`run_judge.py` on Narval and cross-check against `REPRODUCE.md` before trusting its exact §3 commands**, especially before submitting a real `sbatch` job. Happy to reconcile `REPRODUCE.md` with the real CLIs if you want that as a separate pass.
