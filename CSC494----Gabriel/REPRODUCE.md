# Reproducing PluralTree — a guide for the next student

This is your entry point. It gets you from a fresh clone to **the baseline
number in the report** (`culfit_baseline`: P 0.407 / R 0.361 / **F1 0.340** at
n=100), and then to the agentic and stratified results. Read it top to bottom
once; after that, the four reference docs below are what you'll actually use.

> **What you're reproducing.** The headline of the Milestone 3 report is that
> cultural augmentation, applied *uniformly*, has no detectable aggregate effect
> (ΔF1 = −0.030, CI crosses zero), but **helps on hard items and hurts on easy
> ones**. The baseline (`culfit_baseline`) is the reference the whole story is
> measured against — reproduce it first, exactly, before touching anything else.

---

## 0. The five documents, and when to read each

| Doc | Read it when |
|---|---|
| **`REPRODUCE.md`** (this file) | First. The end-to-end path. |
| **`README.md`** | To understand the code in dataflow order (bootstrap → backend → agents → orchestrator → harness). |
| **`NARVAL.md`** | Before any cluster run. The air-gapped model/SBERT pre-cache, the smoke test, the batch job. This is the operational bible — follow it exactly. |
| **`GIT_SETUP.md`** | Once, for the laptop → GitHub → Narval sync model. |
| **`PROJECT_HANDOFF.md`** | For the *why*: decisions, rationale, and the full result history as of the handoff date. |

**Golden rule from the project:** verify before you commit compute. Every step
below has a cheap check that fails in seconds if something is wrong, so you never
burn a GPU slot on a broken setup.

---

## 1. Get the code (laptop)

```bash
git clone <your-fork-url> CSC494
cd CSC494
```

The repo vendors three sub-projects that must sit as direct children of `CSC494/`:

```
CSC494/
├── CulFiT/                                 # Agent B (critique) — do NOT edit
├── Cultural_Commonsense_Knowledge_Graph/   # Agent A (generator) — do NOT edit
└── orchestration/                          # our layer — all your work goes here
```

`bootstrap.py` loads the two vendored repos by absolute path so their three
colliding `utils/` folders never clash. **Never edit files inside `CulFiT/` or
`Cultural_Commonsense_Knowledge_Graph/`** — that's the invariant that keeps our
numbers comparable to CulFiT's published ones.

---

## 2. Prove the pipeline works offline (2 minutes, no GPU, no keys)

```bash
python -m orchestration.check_setup          # expect: PASS — setup looks good.
python -m orchestration.run_ablation --mode mock
```

Mock mode uses canned responders — no model, no network, no cost. If both
commands succeed you have a working pipeline and the plumbing is intact. If
`check_setup` fails, a repo folder is misplaced or renamed (see its output).

---

## 3. Reproduce the baseline number (the important part)

The evaluation is **two-pass by design**: a `generate` pass writes answers to
disk (judge-agnostic), then a `judge` pass scores them. This is why you can
re-score with a different judge later *without regenerating a single answer* —
remember that, it saves you days.

### 3a. Build the n=100 eval slice

```bash
python -m orchestration.build_eval_set --n 100 --seed 0 --out runs/eval_n100.jsonl
```

This samples 100 items from GlobalCultureQA (`cultural_group`-stratified, fixed
seed so it's identical run to run). The exact slice used for the report is
pinned by `--seed 0`; keep it to match the numbers.

### 3b. Generate answers — baseline and agentic (cluster)

Do the model/SBERT pre-cache in **`NARVAL.md` Step 0** first (login node,
air-gapped cluster — this is not optional). Then:

```bash
# baseline: base model answers one-shot with CulFiT's prompt
python -m orchestration.run_generate \
    --arm culfit_baseline \
    --eval runs/eval_n100.jsonl \
    --model $SCRATCH/models/qwen25-7b \
    --out runs/gen_baseline_n100.jsonl

# agentic: the full deliberation loop (uniform augmentation on every item)
python -m orchestration.run_generate \
    --arm agentic_sequential \
    --eval runs/eval_n100.jsonl \
    --model $SCRATCH/models/qwen25-7b \
    --out runs/gen_agentic_n100.jsonl
```

### 3c. Judge both, with the SAME scorer

```bash
python -m orchestration.run_judge \
    --gen runs/gen_baseline_n100.jsonl \
    --judge-model $SCRATCH/models/qwen25-7b \
    --out runs/judged_baseline_n100.jsonl

python -m orchestration.run_judge \
    --gen runs/gen_agentic_n100.jsonl \
    --judge-model $SCRATCH/models/qwen25-7b \
    --out runs/judged_agentic_n100.jsonl
```

### 3d. Aggregate and check against the target

```bash
python -m orchestration.analyze_taskeval runs/judged_baseline_n100.jsonl runs/judged_agentic_n100.jsonl
```

**Expected (must match to reproduce the report):**

| Arm | Precision | Recall | F1 |
|---|---|---|---|
| `culfit_baseline` | 0.407 | 0.361 | **0.340** |
| `agentic_sequential` | 0.360 | 0.355 | **0.310** |

ΔF1 = −0.030, 95% CI [−0.075, +0.016], p = 0.12, win/loss/tie = 34/48/18.

If your baseline F1 is off by more than ~0.01, stop and check: (1) same `--seed`,
(2) same judge model, (3) SBERT actually loaded offline (see NARVAL Step 0 — a
missing cache silently degrades node merging), (4) you didn't edit a vendored
`utils/`.

---

## 4. Reproduce the key finding (hard/easy stratification)

The report's headline is the *conditional* result. To regenerate it:

```bash
python -m orchestration.analyze_taskeval \
    runs/judged_baseline_n100.jsonl runs/judged_agentic_n100.jsonl \
    --stratify-by baseline_f1 --hard-threshold 0
```

`--hard-threshold 0` splits items into **hard** (baseline F1 = 0, the model
answered nothing correctly) and **easy** (baseline F1 > 0). The number to report
is the **hard-case help rate**: of the hard items, what fraction did augmentation
improve. (In the report this is estimated at ~80% of ~30 hard items, pending this
exact recomputation — running the command above is what replaces the estimate
with the measured value.)

---

## 5. What to do next (the handed-off research direction)

The project pivots to **Adaptive Cultural Contextualization**: gate augmentation
on competence instead of applying it uniformly. Concretely, in priority order:

1. **Re-judge under Gemini Flash.** The local Qwen judge is saturated (89/100 at
   ceiling). Point `--judge-model gemini-flash` at the *existing* generations
   (no regeneration needed) and re-run §3d and §4. Confirm the ceiling breaks and
   the hard/easy effect survives. **Do this before anything else** — everything
   downstream depends on a judge with real dynamic range.
2. **Rule-based gate.** Build the bag-of-cultures lookup from the hard set (§4)
   and augment only those items. Compare against the oracle gate (augment exactly
   where it helps) to see how much of the achievable gain it captures.
3. **Self-confidence gate + KG-structured augmentation.** See the report's §5–§6.

Gemini credits are on the shared Google Cloud project — **use conservatively and
report usage**, it's shared with another (mobility) project.

---

## Environment notes

- **Local:** Windows PowerShell. Use `scp` (not `rsync`) for results; `runs/` is
  gitignored — results come back outside git. Set `git config --global
  core.autocrlf input` once so Windows CRLF doesn't break cluster bash scripts.
- **Cluster:** Narval, account `def-enaskt`, opportunistic (low-priority) GPU.
  Stage model weights to `$SLURM_TMPDIR` (local SSD) inside the job — reading
  weights straight from Lustre scratch is painfully slow. Full runbook in
  `NARVAL.md`.
- **Models:** Llama-3.1-8B (generation), Qwen2.5-7B (local judge — saturating,
  being replaced), Gemini Flash (generator + judge going forward).

If a step fails, the fix is almost always in `NARVAL.md` (cluster) or the
"Decisions & rationale" section of `PROJECT_HANDOFF.md` (why something is the way
it is). Good luck.
