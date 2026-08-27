
1. This is in `agent_b_engine.py` and it's the thing most likely to silently break your numbers:
```python
if res and "Yes" in res:
    verdicts.append("Yes")
else:
    verdicts.append("No")
```
This is a substring check on free-form model output. It's brittle in a way that interacts badly with a model swap:

The substring `"Yes"` matches anywhere — including `"Yes and No"`, `"Yesterday"`, or a chatty Qwen preamble like `"Yes, here's my reasoning..."`. Qwen2.5 tends to be *more* verbose and explanatory than Llama at temperature 0 unless tightly constrained, so the rate of accidental `"Yes"` matches (or `"Yes"` buried in a "No" explanation) can shift. Your `precision_score`, `approved`, and `mean_precision` all derive from this check — so a model swap can move your headline metrics through parsing noise alone, not real capability differences. For an *ablation* whose entire point is comparing topologies, that's a measurement-validity problem worth fixing regardless of Qwen. 
Consider anchoring it: check whether the response *starts with* Yes/No after stripping, or normalize and look for a leading token, rather than substring-anywhere.


2. `agent_a_adapter.py` routes raw model text through CulFiT's `parse_llm_response`, and notably it swallows failures silently:
```python
except Exception as e:
    logger.warning("Agent A parse failed (%s); returning empty.", e)
    return []
```
If Qwen wraps its JSON array in ```json fences (it frequently does) and `parse_llm_response` wasn't written to strip them, 
every Agent A generation parses to `[]` — meaning zero paths, and run looks like it "works" (no crash) while producing empty results and misleading summaries. 
The `repair()` path's system prompt says "Output STRICT JSON only," which helps but doesn't guarantee fence-free output from Qwen. 
This is exactly what `--smoke` is for, but only if you *read the trace* rather than trusting the exit code. 
After the smoke run, check that `n_final_paths > 0` in `runs/ablation_smoke_*.jsonl` — if it's 0, this parser is the culprit.\



If you ever see Qwen running on past its answer, the fix is passing eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")],
but try it as-is first; current transformers + Qwen2.5's bundled config usually handle this.



run `--smoke` with Qwen and inspect the trace for `n_final_paths > 0` 
and sane `final_mean_precision` (not suspiciously all-1.0 or all-0.0, which would signal the `"Yes"` substring check misfiring). 
Third, consider hardening the Agent B verdict check before you trust ablation numbers across models.
