"""
Unified, model-agnostic LLM backend for the PluralTree agentic system.

One flag (`mode`) flips the entire pipeline between:
  - "mock"  : deterministic canned responses, runs anywhere, no GPU / no API key.
              Use this NOW while the Narval cluster is down to develop and debug
              the deliberation logic.
  - "api"   : OpenAI-compatible HTTP endpoint (GPT-4o via chatanywhere, or a
              vLLM server at 0.0.0.0:8001). Reuses agent_b.utils.llm_utils.
  - "local" : in-process HuggingFace transformers model on a GPU. Reuses
              agent_b.utils.llm_utils.lama_generation.

Both Agent A and Agent B receive ONE of these objects and call `.chat(messages)`.
Switching mock -> cluster is a one-line change in the ablation config; no agent
code changes.
"""

from __future__ import annotations
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class LLMBackend:
    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        """Take OpenAI-style [{'role','content'}, ...], return raw string."""
        raise NotImplementedError


class MockBackend(LLMBackend):
    """
    Deterministic backend for offline development.

    A `responder` callable receives the messages list and returns a string.
    The default responder is generic; the ablation harness installs richer
    scenario responders so the three topologies actually exercise their logic
    (e.g. Agent A "repairs" a draft, Agent B flips from No -> Yes).
    """

    def __init__(self, responder: Optional[Callable[[list[dict]], str]] = None,
                 name: str = "mock"):
        self.responder = responder or self._default_responder
        self.name = name
        self.call_count = 0

    def _default_responder(self, messages: list[dict]) -> str:
        return "Yes - Validated alignment."

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        self.call_count += 1
        return self.responder(messages)


def _import_from_culfit_llm_utils(name: str):
    """
    Import `name` from CulFiT's llm_utils (registered by bootstrap), turning the
    cryptic failure modes into one clear message. The usual causes are:
      - CulFiT/utils/llm_utils.py has a broken top-level import (e.g. the
        removed `openai.api_key`, or `openai`/`tenacity`/`pydantic` not installed
        in this venv), so the module loads partially and `name` is never defined.
    """
    from orchestration import bootstrap
    bootstrap.install()
    try:
        import culfit_llm_utils  # bootstrap-registered
    except Exception as e:
        raise ImportError(
            f"Could not import CulFiT's llm_utils. This is almost always a broken "
            f"top-level import in CulFiT/utils/llm_utils.py or a missing package in "
            f"this environment (openai / tenacity / pydantic). Original error: {e}"
        ) from e
    if not hasattr(culfit_llm_utils, name):
        raise ImportError(
            f"'{name}' is missing from CulFiT/utils/llm_utils.py. The module likely "
            f"imported only partially because a top-level import failed before "
            f"'{name}' was defined. Check the imports at the top of that file and "
            f"that openai / tenacity / pydantic are installed (pip install --no-index ...)."
        )
    return getattr(culfit_llm_utils, name)


class APIBackend(LLMBackend):
    """OpenAI-compatible endpoint. Lazily imports so mock mode never needs openai."""

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self._openai_response = _import_from_culfit_llm_utils("openai_response")

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        out = self._openai_response(
            model_name=self.model_name, messages=messages, temperature=temperature
        )
        return out or ""

class GeminiBackend(LLMBackend):
    """
    Gemini via its OpenAI-compatible endpoint. Used as BOTH the augmentation
    generator and the judge in the pivoted pipeline.

    Lazily imports `openai` so mock/local modes never require it. Reads the key
    from GEMINI_API_KEY (falls back to GOOGLE_API_KEY) so it never collides with
    OPENAI_API_KEY used by APIBackend.
    """

    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    def __init__(self, model_name: str = "gemini-2.0-flash",
                 api_key: str | None = None, base_url: str | None = None,
                 max_retries: int = 4):
        self.model_name = model_name
        import os
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "GeminiBackend needs an API key. Set GEMINI_API_KEY in your .env "
                "(or pass api_key=...). This is separate from OPENAI_API_KEY."
            )
        try:
            from openai import OpenAI  # lazy: only needed in gemini mode
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "GeminiBackend requires the `openai` package (>=1.0). "
                "pip install --no-index openai  (or with network: pip install openai). "
                f"Original error: {e}"
            ) from e
        self._client = OpenAI(
            api_key=key,
            base_url=base_url or self._BASE_URL,
            max_retries=max_retries,
        )

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "") if resp.choices else ""


class LocalBackend(LLMBackend):
    """In-process HF transformers model (cluster GPU). Lazily imports torch/transformers."""

    def __init__(self, model_obj, tokenizer_obj, model_name: str = "llama3"):
        self.model_obj = model_obj
        self.tokenizer_obj = tokenizer_obj
        self.model_name = model_name
        self._lama_generation = _import_from_culfit_llm_utils("lama_generation")

    def chat(self, messages: list[dict], temperature: float = 0.1) -> str:
        # HF's model.generate with do_sample=True rejects temperature=0.0
        # ("has to be a strictly positive float"). Agent B asks for 0.0 to mean
        # "be deterministic"; the closest valid sampling value is a tiny epsilon,
        # which is effectively greedy. Clamp here so callers can still pass 0.0.
        temp = max(temperature, 1e-2)
        out = self._lama_generation(
            model=self.model_obj, tokenizer=self.tokenizer_obj,
            input_messages=messages, temperature=temp,
        )
        return out or ""


def make_backend(mode: str, **kwargs) -> LLMBackend:
    """Factory. `mode` in {"mock","api","gemini","local"}."""
    mode = (mode or "mock").lower()
    if mode == "mock":
        return MockBackend(responder=kwargs.get("responder"),
                           name=kwargs.get("name", "mock"))
    if mode == "api":
        return APIBackend(model_name=kwargs.get("model_name", "gpt-4o"))

    if mode == "gemini":
        return GeminiBackend(
            model_name=kwargs.get("model_name", "gemini-2.0-flash"),
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url"),
        )
    
    if mode == "local":
        return LocalBackend(
            model_obj=kwargs["model_obj"],
            tokenizer_obj=kwargs["tokenizer_obj"],
            model_name=kwargs.get("model_name", "llama3"),
        )
    raise ValueError(f"Unknown backend mode: {mode!r} (expected mock/api/local)")
