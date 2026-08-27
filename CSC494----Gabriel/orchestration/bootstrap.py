"""
Bootstrap for in-place integration under CSC494/.

The problem: there are THREE directories named `utils` across the two repos:
  - CulFiT/utils/                       (Agent B: prompt_utils, llm_utils, json_fields)
  - Cultural_Commonsense_Knowledge_Graph/src/utils/   (Agent A: llm_query, prompt_templates, response_parser)
  - Cultural_Commonsense_Knowledge_Graph/utils/       (a COPY of CulFiT's utils)

If two of these sit on sys.path at once, `import utils.X` resolves against
whichever appears first -> silent wrong-module bugs.

The fix: never rely on a bare `utils` on sys.path. Instead load each file we
need by its explicit filesystem path under a unique module name, using
importlib. We also alias the names that the original files import internally
(e.g. agent_b_critique.py does `from utils.prompt_utils import ...`) so the
upstream code runs UNMODIFIED.

Nothing in either original repo is edited or copied.
"""

from __future__ import annotations
import importlib.util
import os
import sys
import types

# ---- Resolve repo roots relative to this file --------------------------------
# This file lives at CSC494/orchestration/bootstrap.py
_THIS = os.path.dirname(os.path.abspath(__file__))
CSC494 = os.path.dirname(_THIS)
CULFIT = os.path.join(CSC494, "CulFiT")
AGENT_A = os.path.join(CSC494, "Cultural_Commonsense_Knowledge_Graph")
AGENT_A_SRC = os.path.join(AGENT_A, "src")


def _load(module_name: str, file_path: str):
    """Load a single .py file as `module_name`, registering it in sys.modules."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ensure_pkg(name: str):
    """Create an empty namespace package entry so `pkg.sub` aliasing works."""
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = []  # mark as package
        sys.modules[name] = pkg
    return sys.modules[name]


def install():
    """
    Make both agents importable without collision. Call once before importing
    the adapters. Returns a dict of the key loaded modules for convenience.
    """
    # --- Agent B (CulFiT) ---------------------------------------------------
    # agent_b_critique.py does `from utils.prompt_utils import ...` and
    # `from utils.llm_utils import ...`. We satisfy that by registering a
    # `utils` package whose submodules point at CulFiT's files specifically.
    culfit_prompt_utils = _load("culfit_prompt_utils",
                                os.path.join(CULFIT, "utils", "prompt_utils.py"))

    # NOTE: llm_utils imports tenacity/openai (only needed for api/local modes).
    # Load it lazily so mock mode runs with zero extra deps.
    culfit_llm_utils = None
    try:
        culfit_llm_utils = _load("culfit_llm_utils",
                                 os.path.join(CULFIT, "utils", "llm_utils.py"))
    except ImportError as e:
        import logging
        logging.getLogger(__name__).info(
            "culfit_llm_utils not loaded (%s); fine for mock mode.", e)

    # Alias them under the `utils.*` names the upstream file expects.
    _ensure_pkg("utils")
    sys.modules["utils.prompt_utils"] = culfit_prompt_utils
    if culfit_llm_utils is not None:
        sys.modules["utils.llm_utils"] = culfit_llm_utils

    # --- Agent A (Tonga) ----------------------------------------------------
    # Tonga's modules import each other as `from utils.llm_query import *`
    # etc., AND `from config import ...`. Those names are relative to src/.
    # We load Tonga's files under explicit names, then alias the internal
    # references so the upstream code runs unmodified.
    #
    # config.py needs python-dotenv and is only used for multilingual prompt
    # selection / API keys -> load lazily so mock mode needs neither.
    a_config = None
    try:
        a_config = _load("agentA_config", os.path.join(AGENT_A_SRC, "config.py"))
    except ImportError as e:
        import logging
        logging.getLogger(__name__).info(
            "agentA_config not loaded (%s); fine for mock mode.", e)

    a_prompt_templates = _load("agentA_prompt_templates",
                               os.path.join(AGENT_A_SRC, "utils", "prompt_templates.py"))
    a_response_parser = _load("agentA_response_parser",
                              os.path.join(AGENT_A_SRC, "utils", "response_parser.py"))

    return {
        "culfit_prompt_utils": culfit_prompt_utils,
        "culfit_llm_utils": culfit_llm_utils,
        "agentA_config": a_config,
        "agentA_prompt_templates": a_prompt_templates,
        "agentA_response_parser": a_response_parser,
    }
