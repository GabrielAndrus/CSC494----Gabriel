#!/bin/bash
# LOGIN NODE (has internet). Downloads Ki et al.'s refined NormAd-ETI file (2,633 items, ~5MB).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
curl -fL https://raw.githubusercontent.com/dayeonki/cultural_debate/main/data/normad.jsonl -o data/normad_ki.jsonl
test "$(wc -l < data/normad_ki.jsonl)" -eq 2633 && echo "OK: 2633 items" || { echo "UNEXPECTED line count"; exit 1; }
