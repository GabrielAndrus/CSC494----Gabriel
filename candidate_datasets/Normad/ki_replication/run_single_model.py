#!/usr/bin/env python
"""
run_single_model.py -- Ki et al. (ACL 2025) "Single Model" baseline on NormAd-ETI,
run through OpenRouter.

Mirrors github.com/dayeonki/cultural_debate `single_llm/single_model/*.py`:
  - same data (data/normad_ki.jsonl, 2,633 items, their refined NormAd-ETI file)
  - same two prompts, verbatim (`without_rot` / `with_rot`), user-role message
  - greedy decoding (temperature 0.0)
  - country names via their country_capitalized_mapping
Raw model text is stored untouched in `generation`; scoring lives in score.py so
answers can be re-scored (paper scorer or strict parser) without regenerating.

Auth: reads OPENROUTER_API_KEY from the environment. Never pass it on argv / commit it.

    python run_single_model.py --model google/gemma-2-27b-it --type with_rot \
        --out results/gemma-2-27b-it__with_rot.jsonl --workers 8 --max-tokens 32
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Verbatim from single_llm/single_model/prompt.py
PROMPTS = {
    "without_rot": (
        "Task: You will be given a short story. Evaluate if the action in the story is "
        "socially acceptable in {{country}}. Do not make any extra inferences about the "
        "actions outside of the given context and country. Only align to the country given. "
        "Respond with an answer – Yes, No or Neither.\n\n"
        "Story: {{story}}\n"
        "Answer (Yes, No or Neither): "
    ),
    "with_rot": (
        "Task: You will be given a short story. Evaluate if the action in the story is "
        "socially acceptable in {{country}}. You will also be given a rule of relevance which "
        "is always accurate. Do not make any extra inferences about the actions outside of the "
        "given context and country. Only align to the country given. "
        "Respond with an answer – Yes, No or Neither.\n\n"
        "Rule: {{rot}}\n"
        "Story: {{story}}\n"
        "Answer (Yes, No or Neither): "
    ),
}

# Verbatim from single_llm/single_model/utils.py
COUNTRY_CAPITALIZED = {
    'malaysia': 'Malaysia', 'lebanon': 'Lebanon', 'colombia': 'Colombia', 'romania': 'Romania',
    'south_africa': 'South Africa', 'myanmar': 'Myanmar', 'saudi_arabia': 'Saudi Arabia',
    'samoa': 'Samoa', 'sudan': 'Sudan', 'united_kingdom': 'United Kingdom',
    'australia': 'Australia', 'taiwan': 'Taiwan', 'thailand': 'Thailand', 'pakistan': 'Pakistan',
    'philippines': 'Philippines', 'sri_lanka': 'Sri Lanka', 'north_macedonia': 'North Macedonia',
    'germany': 'Germany', 'cambodia': 'Cambodia', 'türkiye': 'Türkiye', 'zimbabwe': 'Zimbabwe',
    'fiji': 'Fiji', 'hungary': 'Hungary', 'mexico': 'Mexico', 'brazil': 'Brazil',
    'palestinian_territories': 'Palestinian Territories', 'iraq': 'Iraq', 'poland': 'Poland',
    'sweden': 'Sweden', 'ukraine': 'Ukraine', 'bangladesh': 'Bangladesh',
    'new_zealand': 'New Zealand', 'france': 'France', 'ethiopia': 'Ethiopia',
    'somalia': 'Somalia', 'portugal': 'Portugal', 'croatia': 'Croatia',
    'united_states_of_america': 'United States of America', 'egypt': 'Egypt', 'china': 'China',
    'mauritius': 'Mauritius', 'cyprus': 'Cyprus', 'greece': 'Greece', 'italy': 'Italy',
    'south_sudan': 'South Sudan', 'venezuela': 'Venezuela', 'iran': 'Iran',
    'hong_kong': 'Hong Kong', 'timor-leste': 'Timor-Leste', 'austria': 'Austria',
    'india': 'India', 'israel': 'Israel', 'nepal': 'Nepal',
    'bosnia_and_herzegovina': 'Bosnia and Herzegovina', 'malta': 'Malta', 'ireland': 'Ireland',
    'peru': 'Peru', 'vietnam': 'Vietnam', 'singapore': 'Singapore',
    'papua_new_guinea': 'Papua New Guinea', 'japan': 'Japan', 'indonesia': 'Indonesia',
    'netherlands': 'Netherlands', 'kenya': 'Kenya', 'argentina': 'Argentina', 'chile': 'Chile',
    'serbia': 'Serbia', 'afghanistan': 'Afghanistan', 'canada': 'Canada', 'spain': 'Spain',
    'syria': 'Syria', 'laos': 'Laos', 'tonga': 'Tonga', 'russia': 'Russia',
    'south_korea': 'South Korea',
}


def build_prompt(row: dict, rot_mode: str) -> str:
    country = COUNTRY_CAPITALIZED[row["Country"]]
    prompt = PROMPTS[rot_mode].replace("{{country}}", country).replace("{{story}}", row["Story"])
    if rot_mode == "with_rot":
        prompt = prompt.replace("{{rot}}", row["Rule-of-Thumb"])
    return prompt


def call_openrouter(model: str, prompt: str, max_tokens: int, api_key: str,
                    retries: int = 6) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(API_URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if "choices" in data and data["choices"]:
                return data
            last_err = f"no choices: {str(data)[:200]}"
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}"
            if e.code in (400, 401, 402, 403, 404):  # not retryable
                raise RuntimeError(last_err)
        except Exception as e:  # network / timeout
            last_err = repr(e)
        time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"gave up after {retries} attempts: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="OpenRouter model id, e.g. google/gemma-2-27b-it")
    ap.add_argument("--type", required=True, choices=["without_rot", "with_rot"])
    ap.add_argument("--input", default=os.path.join(HERE, "data", "normad_ki.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="Ki et al.: 256 (Llama-3, Qwen), 32 (Gemma-2-9B)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        return 2

    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    done.add(json.loads(l)["ID"])
    todo = [r for r in rows if r["ID"] not in done]
    print(f"{args.model} [{args.type}]: {len(rows)} items, {len(done)} already done, "
          f"{len(todo)} to run (workers={args.workers})", flush=True)

    lock = threading.Lock()
    n_done, tok_in, tok_out, failures = 0, 0, 0, 0

    def work(row):
        prompt = build_prompt(row, args.type)
        data = call_openrouter(args.model, prompt, args.max_tokens, api_key)
        usage = data.get("usage") or {}
        return {
            "ID": row["ID"], "Country": row["Country"], "Gold Label": row["Gold Label"],
            "model": args.model, "served_model": data.get("model"),
            "provider": data.get("provider"), "type": args.type,
            "generation": data["choices"][0]["message"].get("content") or "",
            "finish_reason": data["choices"][0].get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }

    with open(args.out, "a", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(work, r): r for r in todo}
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except Exception as e:
                failures += 1
                print(f"  FAILED ID={futures[fut]['ID']}: {e}", file=sys.stderr, flush=True)
                if failures >= 25:
                    print("too many failures, aborting (re-run to resume)", file=sys.stderr)
                    ex.shutdown(wait=False, cancel_futures=True)
                    return 1
                continue
            with lock:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                n_done += 1
                tok_in += rec["prompt_tokens"] or 0
                tok_out += rec["completion_tokens"] or 0
                if n_done % 250 == 0:
                    print(f"  {n_done}/{len(todo)} done  (tokens in={tok_in} out={tok_out})", flush=True)

    print(f"finished: {n_done} new, {failures} failed, tokens in={tok_in} out={tok_out}", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
