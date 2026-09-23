#!/bin/bash
# Optional convenience: queue everything with dependencies, from ki_replication/narval/ on Narval.
# (Submit the smoke test FIRST and inspect it -- see RUNBOOK.md.)
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p logs
J1=$(sbatch --parsable single_agent.slurm)
J2=$(sbatch --parsable debate.slurm)
J3=$(sbatch --parsable --dependency=afterok:$J2 judge_27b.slurm)
echo "single_agent=$J1  debate=$J2  judge_27b(after debate)=$J3"
squeue -u "$USER"
