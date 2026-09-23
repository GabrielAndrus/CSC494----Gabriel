# Narval runbook: Ki et al. single-agent + multi-agent replication

Complements `orchestration/NARVAL.md` (read its "compute nodes have no internet" and
"one job per allocation" notes first). Steps marked **YOU** cannot be done by Claude: they need
your credentials, browser or an interactive Narval shell. Everything else is already scripted.

**What runs**

| Iteration | What | Models | Script / job |
|---|---|---|---|
| 1. Single agent | Ki's Single Model baseline, without and with rule-of-thumb (paper Table 1) | Llama-3-8B-Instruct, Gemma-2-9B-it | `single_agent.slurm` (1 GPU) |
| 2. Multi-agent | Debate-Only, one feedback round, Gemma-2-9B + Llama-3-8B, then Gemma-2-27B judge on disagreements (paper Table 2, row LLAMA-3 + GEMMA-2) | + Gemma-2-27B | `debate.slurm` (2 GPUs) then `judge_27b.slurm` (2 GPUs) |
| reference | Gemma-2-27B single-agent (paper Table 1 reference row) | Gemma-2-27B | inside `judge_27b.slurm` |

Debate is run as two **variants**: `fixed` (the paper's prompts) and `released` (the public code's
actual behaviour, which never inserts the rule-of-thumb; see `common.py`). Comparing them tells us
whether the public code reproduces the paper.

Targets from the paper: Table 1 Llama-3 49.5 / 63.7, Gemma-2 50.7 / 68.9, Gemma-2-27B 55.8 / 79.2 (without / with rule);
Table 2 (Llama-3 + Gemma-2): D(Llama-3) 66.5, D(Gemma-2) 76.7, adjudicated D 79.7.

---

## Step 0 (YOU, once): accounts and access

1. **Alliance (CCDB) account with access to `def-enaskt` on Narval, and MFA set up.**
   `NARVAL.md` and the existing `.slurm` files use `~/projects/def-enaskt/hhpfiona/`, which is Fiona's
   directory. Yours will be `~/projects/def-enaskt/gandrus/`. If you are not yet in the
   `def-enaskt` group, ask Prof. AlTarawneh (the account sponsor) to add you in CCDB. Alliance login requires
   multi-factor authentication, so have your second factor ready.
2. **Hugging Face:** in a browser, logged in, open each model page and accept the license:
   - https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
   - https://huggingface.co/google/gemma-2-9b-it
   - https://huggingface.co/google/gemma-2-27b-it

   Then create a **read** token (Settings -> Access Tokens). Keep it private; you will paste it into
   `hf auth login` on Narval yourself. Never share it in chat.

## Step 1 (YOU): get this code onto Narval

Pick one.

**A. Via git** (needs the `ki_replication/` folder committed and pushed to GitHub first; ask Claude to do that):

```bash
ssh gandrus@narval.alliancecan.ca          # MFA prompt
mkdir -p ~/projects/def-enaskt/gandrus && cd ~/projects/def-enaskt/gandrus
git clone https://github.com/GabrielAndrus/CSC494----Gabriel.git CSC494     # first time
# later updates:  cd CSC494 && git pull
```

**B. Via scp from your Mac** (no git needed). Run this on your Mac:

```bash
ssh gandrus@narval.alliancecan.ca "mkdir -p ~/projects/def-enaskt/gandrus/CSC494/candidate_datasets/Normad"
scp -r candidate_datasets/Normad/ki_replication \
    gandrus@narval.alliancecan.ca:projects/def-enaskt/gandrus/CSC494/candidate_datasets/Normad/
```

Either way, then on Narval: `cd ~/projects/def-enaskt/gandrus/CSC494/candidate_datasets/Normad/ki_replication/narval`

## Step 2 (YOU, once, on a LOGIN node): download data and models (~90 GB)

Compute nodes are offline, so this must happen here. Models go in `$SCRATCH` (not home).

```bash
cd ~/projects/def-enaskt/gandrus/CSC494/candidate_datasets/Normad/ki_replication/narval
bash fetch_data.sh                          # Ki et al.'s 2,633-item NormAd-ETI file; expect "OK: 2633 items"

module load python/3.11
python -m venv $SCRATCH/precache_env && source $SCRATCH/precache_env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index huggingface_hub
hf auth login                               # paste your READ token when prompted
bash precache_models.sh                     # Llama-3-8B, Gemma-2-9B, Gemma-2-27B; resumable if interrupted
deactivate
ls $SCRATCH/models                          # expect: gemma2-27b-it gemma2-9b-it llama3-8b-instruct
```

If a download says 401/403 for a model, the license for that model has not been accepted on your account (Step 0.2).
If `hf` is not found, try `huggingface-cli` instead of `hf`.

## Step 3 (YOU): smoke test, about 30-60 min, catches problems cheaply

```bash
salloc --account=def-enaskt --gres=gpu:2 --cpus-per-task=6 --mem=128G --time=01:00:00
# wait for the prompt to change to a compute node (e.g. ng10104), then:
cd ~/projects/def-enaskt/gandrus/CSC494/candidate_datasets/Normad/ki_replication/narval
bash smoke.sh
exit                                        # releases the allocation
```

`smoke.sh` runs 8 items through every stage (single agent x2, both debate variants, the 27B judge). It first prints
`torch ... transformers ... gpus=2` and stops immediately if `transformers` is older than 4.42 (Gemma-2 needs it) or
no GPU is visible. **Paste the last ~40 lines of output back to Claude** (or any traceback): that is where wheel
versions, memory or Gemma-2 batching problems will show up. If the 2-GPU wait is long, `squeue -u $USER --start`
shows an estimate.

## Step 4 (YOU): submit the real jobs

```bash
cd ~/projects/def-enaskt/gandrus/CSC494/candidate_datasets/Normad/ki_replication/narval
mkdir -p logs
bash submit_all.sh          # queues single_agent, debate, and judge_27b (after debate finishes)
squeue -u $USER
```

Or one at a time: `sbatch single_agent.slurm`, `sbatch debate.slurm`, then `sbatch --dependency=afterok:<debate-jobid> judge_27b.slurm`.
Allocation is opportunistic (low priority), so queue times vary; jobs are **resumable**, so if one hits its wall time
just `sbatch` it again and it skips finished items.

Rough runtimes (estimates from the paper's timings and the batching, **not measured**): single agent under 1 h;
debate a few hours per variant; judge/27B 1-3 h. Logs: `logs/<jobname>_<jobid>.out`.

```bash
tail -f logs/ki_debate_<jobid>.out           # progress: "  <done>/<total>  (Ns elapsed, x.xs/item)"
seff <jobid>                                 # after it finishes: CPU/mem efficiency
```

## Step 5 (YOU): bring results back and score

Results land in `../results_narval/`. From your Mac:

```bash
scp -r gandrus@narval.alliancecan.ca:projects/def-enaskt/gandrus/CSC494/candidate_datasets/Normad/ki_replication/results_narval \
    candidate_datasets/Normad/ki_replication/
```

Scoring is already printed by the jobs (`summary_*` files), and can be recomputed locally with no GPU:

```bash
cd candidate_datasets/Normad/ki_replication
python3 score.py results_narval/*__without_rot.jsonl results_narval/*__with_rot.jsonl
python3 narval/score_debate.py results_narval/debate_fixed.jsonl
python3 narval/score_debate.py results_narval/debate_released.jsonl
```

Then tell Claude the runs are back and it will write the comparison into `NORMAD_SUMMARY.md`.

---

## Known risks (so they are not a surprise)

- **Not bit-identical to Ki et al.:** they generate one item at a time; we batch (left-padded greedy). Same algorithm, but rare
  bf16 near-ties can flip. Their exact library versions are unknown.
- **Gemma-2 is finicky:** we use eager attention and, as in their scripts, a raw (no chat template) prompt for Gemma. If
  the smoke test shows garbage from Gemma, that is the first thing to look at.
- **Debate runtime is the least certain number** (long generations can stall a batch; `--max-new-tokens 1024` is Ki's
  setting; `sbatch --export=ALL,BATCH=8 debate.slurm` or a lower cap are the levers).
- **Judge details are inferred:** the released repo has no judge code. We use the paper's Appendix A.3.4 prompt, a raw Gemma prompt,
  and label the agents Model1 = Llama-3, Model2 = Gemma-2 (paper says they are exchangeable).
