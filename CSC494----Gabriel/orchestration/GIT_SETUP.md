# Version Control & Sync Setup (one-time)

**Model:** One GitHub repo is the source of truth. Laptop and Narval are
both working copies that push/pull to GitHub. Laptop never pushes code
directly to Narval. Cluster results come back outside git via `rsync`.

```
Code:    laptop  --push-->  GitHub  --pull-->  Narval
Results: Narval  --rsync-->  laptop            (runs/ is gitignored)
```

---

## A. One-time: collapse three repos into one (on LAPTOP)

Right now `CulFiT/` and `Cultural_Commonsense_Knowledge_Graph/` are separate git
repos (forks), and `orchestration/` tracks nothing. We flatten all three into a
single repo at the `CSC494/` level.

> One-way door: this drops the forks' git history. Fine for vendored baselines
> you've already modified, but you lose the clean path to pull upstream fixes.

```bash
cd ~/CSC494

# 1. Remove the inner git setups so they can be tracked by one top-level repo
rm -rf CulFiT/.git
rm -rf Cultural_Commonsense_Knowledge_Graph/.git
# (orchestration/ has no .git yet — nothing to remove)

# 2. Create the .gitignore BEFORE first commit (see section D)
#    ... create it, then:

# 3. Init one repo at the CSC494 level
git init
git add -A
git commit -m "Initial commit: CulFiT + CCKG + orchestration layer"
```

Then create an empty repo on GitHub (e.g. `CSC494` or `pluraltree`), and:

```bash
git remote add origin git@github.com:<you>/CSC494.git
git branch -M main
git push -u origin main
```

You can delete the old fork repos on GitHub afterward — their code now lives in
the new repo.

---

## B. One-time: clone onto NARVAL

```bash
ssh hhpfiona@narval.alliancecan.ca
cd ~/projects/def-enaskt/hhpfiona
git clone https://github.com/hhpfiona/CSC494.git
# -> ~/projects/def-enaskt/hhpfiona/CSC494
```

(If SSH keys to GitHub aren't set up on Narval, use the HTTPS clone URL and a
personal access token, or set up an SSH key on the login node.)

Then do the one-time model download (see NARVAL.md, Step 0) — git carries code,
NOT model weights or datasets.

---

## C. The everyday loop

```bash
# LAPTOP: after editing
cd ~/CSC494
git add -A && git commit -m "describe change" && git push

# NARVAL: get the change before running
cd ~/projects/def-enaskt/hhpfiona/CSC494
git pull

# ... run jobs (see NARVAL.md) ...

# LAPTOP: pull results back, OUTSIDE git
scp -r "hhpfiona@narval.alliancecan.ca:~/projects/def-enaskt/hhpfiona/CSC494/runs/*" "$HOME\Documents\CSC494\runs\"
```

Results live only where they're generated (Narval) and are copied down on
demand. Because `runs/` is gitignored, these never collide with code or bloat
history.

---

## D. .gitignore (place at CSC494/ root before first commit)

```
runs/
__pycache__/
*.pyc
.env
*.egg-info/
.venv/
venv/
# model weights / datasets never go in git:
*.bin
*.safetensors
*.gguf
```

Adjust the data patterns to your actual dataset file types. The rule of thumb:
git tracks code and small text configs only; anything large or machine-specific
(results, caches, weights, venvs) stays out.
```
