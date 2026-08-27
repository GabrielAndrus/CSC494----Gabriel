---
license: cc-by-4.0
language:
  - en
tags:
  - cultural-nlp
  - multi-agent
  - evaluation
  - culfit
  - globalcultureqa
pretty_name: PluralTree — Adaptive Cultural Contextualization (eval artifacts)
task_categories:
  - question-answering
---

# PluralTree — Adaptive Cultural Contextualization

Evaluation artifacts and run logs for the PluralTree project (CSC494, Summer
2026): a study of **when** cultural augmentation helps a language model, rather
than whether to apply it uniformly.

> **Note.** This repository hosts *reproducibility artifacts* — the n=100 eval
> slice, generation/judge run logs, and analysis outputs — not model weights.
> The underlying benchmark (GlobalCultureQA) and prompts belong to CulFiT; see
> Attribution below. Code lives in the companion GitHub repo (link in the repo
> metadata).

## What's here

| File / dir | Contents |
|---|---|
| `eval_n100.jsonl` | The pinned 100-item evaluation slice (`--seed 0`, `cultural_group`-stratified) sampled from GlobalCultureQA. |
| `judged_baseline_n100.jsonl` | Per-item scored answers for the `culfit_baseline` arm (base model, one-shot CulFiT prompt). |
| `judged_agentic_n100.jsonl` | Per-item scored answers for the `agentic_sequential` arm (uniform augmentation). |
| `analysis/` | Aggregate tables, the hard/easy stratification, and per-item ΔF1. |

Each judged record carries the question, the model answer, the atomic knowledge
units, the per-unit Yes/No judgements, and the resulting precision / recall / F1,
so every reported number is traceable to the item that produced it.

## Headline results (n=100, local Qwen2.5-7B judge)

| Arm | Precision | Recall | F1 |
|---|---|---|---|
| `culfit_baseline` | 0.407 | 0.361 | **0.340** |
| `agentic_sequential` | 0.360 | 0.355 | **0.310** |

Applied **uniformly**, augmentation is net-neutral: ΔF1 = −0.030, 95% CI
[−0.075, +0.016], p = 0.12, win/loss/tie = 34/48/18 — the interval crosses zero.

**The key finding is conditional.** The null mean masks two opposing
populations: augmentation helps strongly on **hard** items (baseline F1 = 0) and
hurts on **easy** ones (strong baseline). On the ~30% of items the model cannot
solve, augmentation helps ~80% of them (estimate, pending exact recomputation);
on easy items help and hurt roughly cancel. This motivates **Adaptive Cultural
Contextualization**: gate augmentation on estimated competence.

## Reproducing

See `REPRODUCE.md` in the GitHub repo for the end-to-end path. In brief:

```bash
python -m orchestration.build_eval_set --n 100 --seed 0 --out runs/eval_n100.jsonl
python -m orchestration.run_generate --arm culfit_baseline --eval runs/eval_n100.jsonl --model <model> --out runs/gen_baseline_n100.jsonl
python -m orchestration.run_judge   --gen runs/gen_baseline_n100.jsonl --judge-model <model> --out runs/judged_baseline_n100.jsonl
python -m orchestration.analyze_taskeval runs/judged_baseline_n100.jsonl runs/judged_agentic_n100.jsonl --stratify-by baseline_f1 --hard-threshold 0
```

The two-pass design (generate → judge) means you can re-score these same
generations under a different judge (e.g. Gemini Flash) without regenerating any
answers.

## Known limitations

- **n=100.** Representative for the overall hard/easy picture, but not for
  per-culture claims — roughly 9–11 items per cultural group.
- **Judge saturation.** The local Qwen judge approves ~everything (89/100 at
  ceiling), which compresses aggregate deltas. Re-scoring under a
  higher-dynamic-range judge is the first planned step.
- **Estimated stratification.** The hard-case help rate is an estimate pending
  the exact stratified re-run; the analysis command above produces the measured
  value.

## Attribution

- **Benchmark, metric, and prompts:** CulFiT (Feng et al., ACL 2025 Main).
  GlobalCultureQA and the cultural precision/recall/F1 protocol are theirs; this
  work reuses their prompt strings unmodified through a bootstrap layer.
- **Agent A generator:** Tonga et al., Cultural Commonsense Knowledge Graph.
- **KG extraction (planned):** REBEL (Cabot & Navigli, Findings of EMNLP 2021).

## Citation

If you use these artifacts, please cite the CulFiT benchmark and this project:

```bibtex
@misc{pluraltree2026,
  title  = {Adaptive Cultural Contextualization: When Does Cultural Augmentation Help a Language Model?},
  author = {Hoang, Fiona and AlTarawneh, Enas and Chatha, Divnoor and Duhart Pérez, Álvaro and Jiang, Shawn},
  year   = {2026},
  note   = {CSC494, York University. Under review.}
}
```
