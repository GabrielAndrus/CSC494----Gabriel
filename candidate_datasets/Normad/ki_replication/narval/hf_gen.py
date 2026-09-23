"""
hf_gen.py -- Batched, greedy HuggingFace generation for the Ki et al. replication on Narval.

Ki et al. generate one item at a time. That is ~10h for the debate on 2,633 items, so we
batch (left-padded) instead. Greedy decoding is the same algorithm, but padding can flip
rare near-tie tokens in bf16, so results are "equivalent up to batching numerics", not
bit-identical.

Two prompt formats, matching what Ki's scripts do per model:
  chat : user-role message through the tokenizer's chat template (Llama-3)
  raw  : the prompt string tokenized as-is, no chat template (Gemma-2 in their scripts)

torch / transformers are imported lazily so the pure-logic parts of the package can be
unit-tested on a laptop without them.
"""

from __future__ import annotations


def _require_transformers(min_version=(4, 42)):
    import transformers
    parts = tuple(int(x) for x in transformers.__version__.split(".")[:2] if x.isdigit())
    if parts < min_version:
        raise RuntimeError(
            f"transformers {transformers.__version__} is too old for Gemma-2 "
            f"(need >= {'.'.join(map(str, min_version))}). Load a newer wheel: "
            f"`pip install --no-index --upgrade transformers` or check `pip download transformers`.")


class HFGenerator:
    def __init__(self, model_dir: str, fmt: str, device: str | dict = "auto",
                 dtype: str = "bfloat16", eager_attention: bool = False):
        assert fmt in ("chat", "raw"), fmt
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _require_transformers()
        self.torch = torch
        self.fmt = fmt
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kwargs = dict(torch_dtype=getattr(torch, dtype), device_map=device)
        if eager_attention:
            kwargs["attn_implementation"] = "eager"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
        except TypeError:  # newer transformers renamed torch_dtype -> dtype
            kwargs["dtype"] = kwargs.pop("torch_dtype")
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
        self.model.eval()
        self.eos_ids = self._eos_ids()

    def _eos_ids(self):
        ids = []
        gc = self.model.generation_config.eos_token_id
        if gc is not None:
            ids += list(gc) if isinstance(gc, (list, tuple)) else [gc]
        if self.tok.eos_token_id is not None:
            ids.append(self.tok.eos_token_id)
        eot = self.tok.convert_tokens_to_ids("<|eot_id|>")  # Llama-3 turn terminator
        if isinstance(eot, int) and eot >= 0 and eot != self.tok.unk_token_id:
            ids.append(eot)
        return sorted(set(ids))

    def _encode(self, prompts):
        if self.fmt == "chat":
            texts = [self.tok.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
                for p in prompts]
            return self.tok(texts, add_special_tokens=False, padding=True, return_tensors="pt")
        return self.tok(prompts, padding=True, return_tensors="pt")  # adds BOS where the tokenizer does

    def generate(self, prompts: list[str], max_new_tokens: int, batch_size: int) -> list[str]:
        """Greedy-generate one completion per prompt; returns texts in input order."""
        torch = self.torch
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # less padding
        out = [None] * len(prompts)
        dev = next(self.model.parameters()).device
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            enc = self._encode([prompts[i] for i in idx]).to(dev)
            with torch.inference_mode():
                gen = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    eos_token_id=self.eos_ids, pad_token_id=self.tok.pad_token_id)
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            texts = self.tok.batch_decode(new_tokens, skip_special_tokens=True)
            for i, t in zip(idx, texts):
                out[i] = t
        return out
