# PluralTree — project handoff / context

Drop this in the repo root (e.g. `orchestration/PROJECT_HANDOFF.md`). It captures
the state, decisions, and rationale as of **2026-07-30** so any future session
has the full thread without re-deriving it. Update the "Current state" and "Next"
sections as work progresses; the "Decisions & rationale" section is the durable
record and should mostly grow, not change.

---

## What this project is

**PluralTree** — CSC494 research project (student: Fiona Hoang; supervisor:
Prof. Enas AlTarawneh, low-supervision style). Research question: does
multi-agent deliberation improve culturally-aware LLM question answering?

- **Agent A** — cultural reasoning-path generator (built on the Tonga CCKG codebase).
- **Agent B** — critique engine (built on CulFiT, ACL 2025).
- Orchestrated across static / parallel / sequential topologies.
- Teammate **Shawn** owns a separate neural-gating / hyperbolic-embedding
  workstream. Fiona's scope: the agentic system layer, evaluation, and ablations.

## Benchmark & metric

- **GlobalCultureQA** (CulFiT's eval set): 1,096 distinct cultural QA items
  (CSV has 1,104 rows, ~8 dupes). Columns: cultural_group, topic, source,
  cultural_knowledge, question, grounded_answer, grounded_answer_knowledge_points.
- **Metric**: cultural Precision / Recall / F1 — answers decomposed into atomic
  knowledge points, judged bidirectionally against a gold set. This is CulFiT's
  own `eval_method.py` logic, reusing their prompt strings
  (ANSWER_GENERATION, ANSWER_EXTRACT, EVAL_CULTURAL_POINTS) through the bootstrap.
- **Baseline to beat**: "CulFiT-at-inference" — base model answers one-shot with
  CulFiT's prompt.

## Current state (as of 2026-07-30)

- n=100 evaluation complete under a local Qwen2.5-7B judge:
  - `culfit_baseline`   : P=0.407 R=0.361 **F1=0.340**
  - `agentic_sequential`: P=0.360 R=0.355 **F1=0.310**  (ΔF1 = −0.030)
  - `agentic_sequential_terse`: F1=0.291 (ΔF1 = −0.049)
- **Headline**: multi-agent deliberation shows **no detectable effect** on cultural
  F1. ΔF1 = −0.030, 95% CI [−0.075, +0.016], p = 0.12, d_z = −0.13. CI crosses
  zero → claim is "no detectable effect," NOT "deliberation hurts." win/loss/tie
  = 34/48/18.
- **Terse ablation falsified the scaffolding-narration hypothesis**: terse cost
  recall (Δ=−0.069, p=0.019, Holm-sig) without recovering precision (Δ=−0.027,
  p=0.36). So augmentation adds volume, not correct facts.
- **THE key finding — conditional effect**: the null mean masks two opposing
  populations. Big wins are all on baseline=0 items (Eritrea 0→0.824, Israel
  0→0.571, Lithuania 0→0.500, China 0→0.429); big losses are all on
  strong-baseline items (Morocco 0.805→0.216, Latvia 0.635→0.182, Ethiopia
  0.545→0.121). **Augmentation helps where the base model is ignorant, hurts
  where it's competent.** Suggests confidence/competence-gated augmentation.
  (Pattern observed; interaction NOT yet formally tested — a regression of ΔF1 on
  baseline F1 would confirm it.)
- **Two measurement issues found & handled**:
  1. Metric bug: original `kp_f1` was structurally context-blind (no context arg
     → delta pinned at zero). Replaced with `rubric_mean_precision`, verified it
     responds to context (+0.333 on a targeted test) before trusting runs.
  2. Judge saturation: local Qwen judge approves ~everything — 89/100 items at
     ceiling → aggregate delta ~0 by construction. The judge, not the system, is
     the bottleneck.
- **Judge-reliability calibration**: 20 pairs dumped, not yet human-labeled.
- **SciPy note**: analysis used internal approximations (SciPy unavailable),
  validated to match SciPy to 5 decimals in this p-range. `pip install --no-index
  scipy` and re-run for belt-and-braces.

## The pivot (confirmed by supervisor, 2026-07)

Question shifts from "does deliberation help?" to "what kind of augmentation
helps, and when?" Supervisor's directive (paraphrased) + Fiona's design:

1. **Generate** cultural augmentation with **Gemini Flash**, using CulFiT's
   answer→critique→refine method. (Supervisor constraint: the augmentation
   generator "should have seen lots of cultural data" — Gemini Flash's broad
   pretraining is taken to satisfy this "for now"; a CulFiT-fine-tuned generator
   is the principled fallback if Gemini augmentation is thin.)
2. **Structure** that augmentation via an off-the-shelf KG-construction model
   (REBEL or an LLM triple-extractor) → relational triples (principles, values,
   associations, not just actions) injected back into answer context.
3. **Ablate** three arms on the same items: no augmentation / CulFiT-style flat
   augmentation / KG-structured augmentation.

**Judge & generator = Gemini Flash.** GPT-4o-mini access (CulFiT's original
judge, hardcoded in their `eval_method.py`) is unavailable — no free student key,
supervisor won't pay, Fiona won't pay. Gemini credits are available and
supervisor steered toward them.

### Why the 3-arm design is clean (verified)
- CulFiT's SFT/DPO training data IS on their Google Drive (SFT/, DPO/,
  original_data/ folders), so no GPT-4o-mini regeneration needed IF training.
- **Eval set is held out**: 0/1096 overlap (exact AND normalized) between the
  1,096 GlobalCultureQA questions and CulFiT's 42,193 distinct training questions.
  → can't extract cached augmentation for eval items; instead generate all three
  arms fresh with one model (Gemini) → isolates *structure* as the only variable.
  This is a feature, not a limitation: the augmentation-generating model is held
  constant across arms.
- Held-out also means: if a CulFiT-style generator is trained on the 42k, it
  won't have seen eval questions (no contamination).

### Reproduction status
- Full CulFiT reproduction is NOT the goal (no released checkpoint; only training
  data + LLaMA-Factory configs — llama3_lora_sft_ds3.yaml: base Meta-Llama-3.1-8B,
  LoRA r16, 20 epochs, max_samples 1000, ZeRO-3 [droppable for single-GPU]).
- Reproduction downgraded to "sanity check the eval harness," and even that is
  blocked without GPT-4o-mini. Their judge is hardcoded gpt-4o-mini; their answer
  generation serves a local Qwen via vllm.

## Next (priority order)

1. **[Step 0 — DONE/IN PROGRESS] Gemini backend.** GeminiBackend added to
   llm_backend.py, "gemini" mode in make_backend, GEMINI_API_KEY in .env. Verify
   with smoke_test_gemini.py.
2. **[Step 1 — highest info] Re-score existing `runs/answers_*.jsonl` with Gemini
   judge.** No regeneration (two-pass architecture). Check: does the 89/100
   ceiling break, and does the conditional win/loss effect survive a judge with
   dynamic range? Everything downstream is contingent on this.
3. **[Step 2] CulFiT-style augmentation with Gemini** (answer→critique→refine,
   their prompts). The "flat augmentation" arm.
4. **[Step 3] KG-construction stage** (REBEL / LLM triple-extractor) — the one
   genuinely new component. Decide triple-injection format; keep it a separate
   module so arms differ ONLY in this stage.
5. **[Step 4] Wire 3-arm ablation** in run_ablation.py: no_aug / culfit_style_aug
   / kg_aug. Same items, same Gemini generator, same Gemini judge.
6. **[Step 5] Run (n=100 first, then full 1,096) + analyze.** Re-run the
   win/loss-by-baseline-competence breakdown per arm. Formalize the interaction
   (regression of per-item Δ on baseline competence).

### Open questions for supervisor
- Is KG-structured augmentation the intended *contribution*, or a stepping stone
  to the confidence-gating idea? (Offered a one-page plan.)
- Does the pivot still need the topology comparison (static/parallel/sequential),
  or is that dropped? (Fiona's lean: drop it.)
- Is Aug-3 EACL deadline relevant to Fiona's timeline? Is DiverseSense (supervisor's
  related paper, "a mini version of what we're doing") shareable?

## Architecture / repo mechanics

- Two-pass eval: `run_generate.py` (pass 1, cheap, single GPU, writes
  `runs/answers_<ts>.jsonl`, judge-agnostic) → `run_judge.py` (pass 2, big judge,
  scores with CulFiT P/R/F1, writes `runs/taskeval_<ts>*.json`). Decoupling means
  re-scoring with a new judge needs NO regeneration — this is what makes the
  Gemini judge swap cheap.
- **`make_backend(mode, **kwargs)`** factory in `llm_backend.py`, modes
  mock / api / gemini / local. All agents call `.chat(messages)`. One-line switch;
  no agent code changes.
- **`bootstrap.py`** loads CulFiT + Tonga modules by explicit path under unique
  names (three colliding `utils/` dirs) and aliases them so upstream code runs
  UNMODIFIED. Nothing in either original repo is edited.
- APIBackend routes GPT via CulFiT's `openai_response` (chatanywhere / vllm).
  GeminiBackend deliberately bypasses that and hits Gemini's OpenAI-compatible
  endpoint directly, to avoid editing upstream CulFiT llm_utils.

### Environment & compute
- **Narval** (Digital Research Alliance), account `def-enaskt`, opportunistic GPU,
  project path `~/projects/def-enaskt/hhpfiona/CSC494`. SLURM; `$SLURM_TMPDIR` for
  model staging.
- Models: Llama-3.1-8B (generator), Qwen2.5-7B (local judge — saturated),
  **Gemini Flash** (new judge + augmentation generator).
- Code flow: laptop → GitHub → Narval via `git pull`; results Narval → laptop via
  `scp` (Windows PowerShell, no rsync). `runs/` is gitignored.
- Windows gotchas: open files with `encoding='utf-8'` (data is multilingual;
  cp1252 default crashes on byte 0x90); `head` doesn't exist in cmd.exe.

## Working principles (how Fiona operates)
- Delegates coding, retains judgment on direction/framing; wants the reasoning
  behind design decisions before implementation.
- Direct, honest framing — owns bugs/nulls/saturation as rigor, not confession.
- Null results have structure — item-level analysis beat the aggregate here.
- Judge quality is load-bearing — validate metric + judge before trusting runs.
- Low supervision → keep a written paper trail (progress emails, this file).
- Verify with import-check one-liners / smoke tests before SLURM submissions;
  prefer <30-min smoke tests or sbatch over interactive salloc for long runs.
- Supervisor said "make your own judgement" — shift from "is this right?" emails
  to "here's what I did and found" emails. Direction is confirmed; build.
