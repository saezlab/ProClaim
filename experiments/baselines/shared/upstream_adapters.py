from __future__ import annotations

import importlib
import json
import re
import sys
import types
from pathlib import Path

from baselines.shared.llm import LLMBackend
from baselines.shared.prompt_cache import GrowingPrefixBlocks

_SHARED_DIR = Path(__file__).resolve().parent
_BASELINES_DIR = _SHARED_DIR.parent
_EXPERIMENTS_DIR = _BASELINES_DIR.parent

FIRE_REPO_ROOT = _EXPERIMENTS_DIR / "fire"
FIRE_SYSTEM_PROMPT = "You are a fact-checking agent responsible for verifying the accuracy of claims."


def _clear_repo_modules(*prefixes: str) -> None:
    for module_name in list(sys.modules):
        if any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in prefixes
        ):
            sys.modules.pop(module_name, None)


def import_upstream_module(repo_root: Path, module_name: str):
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    _clear_repo_modules("common", "eval")
    common_pkg = importlib.import_module("common")

    modeling_stub = types.ModuleType("common.modeling")

    class Model:  # pragma: no cover - upstream only uses this for typing
        pass

    modeling_stub.Model = Model

    shared_config_stub = types.ModuleType("common.shared_config")
    shared_config_stub.openai_api_key = ""
    shared_config_stub.anthropic_api_key = ""
    shared_config_stub.serper_api_key = ""
    shared_config_stub.model_options = {
        "gpt_35_turbo": "OPENAI:gpt-3.5-turbo-0125",
    }

    utils_stub = types.ModuleType("common.utils")

    def strip_string(text: str) -> str:
        return text.strip()

    def extract_json_from_output(text: str):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def extract_first_code_block(text: str, ignore_language: bool = True):
        del ignore_language
        match = re.search(r"```(?:\w+)?\n?(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def extract_first_square_brackets(text: str):
        match = re.search(r"\[([^\]]+)\]", text)
        return match.group(1).strip() if match else ""

    def maybe_print_error(message):
        return None

    utils_stub.strip_string = strip_string
    utils_stub.extract_json_from_output = extract_json_from_output
    utils_stub.extract_first_code_block = extract_first_code_block
    utils_stub.extract_first_square_brackets = extract_first_square_brackets
    utils_stub.maybe_print_error = maybe_print_error

    sys.modules["common.modeling"] = modeling_stub
    sys.modules["common.shared_config"] = shared_config_stub
    sys.modules["common.utils"] = utils_stub
    common_pkg.modeling = modeling_stub
    common_pkg.shared_config = shared_config_stub
    common_pkg.utils = utils_stub
    return importlib.import_module(module_name)


# Search backend of the FIRE adapter. "web" is the upstream
# default (DuckDuckGo); "s2" and "pubmed_s2" reuse ReAct's paper-search
# functions so all three adaptive baselines share one retrieval path per
# backend. Set by FIREBaseline from its ``search_backend``.
SEARCH_BACKEND = "web"
VALID_SEARCH_BACKENDS = ("web", "s2", "pubmed_s2")


def name_search_engine(prompt: str, backend: str) -> str:
    """Replace upstream's "Google Search" in a query prompt with the paper-search backend."""
    if backend == "web":
        return prompt
    engine = {"s2": "Semantic Scholar", "pubmed_s2": "PubMed and Semantic Scholar"}[backend]
    return prompt.replace("Google Search", f"{engine} search").replace("Google search", f"{engine} search")


def set_search_backend(backend: str) -> None:
    global SEARCH_BACKEND
    if backend not in VALID_SEARCH_BACKENDS:
        raise ValueError(f"search_backend must be one of {VALID_SEARCH_BACKENDS}, got {backend!r}")
    SEARCH_BACKEND = backend


def do_search(query: str, k: int) -> str:
    from baselines.react_baseline import _do_search, _pubmed_search, _s2_search

    if SEARCH_BACKEND == "s2":
        return _s2_search(query, k=k)
    if SEARCH_BACKEND == "pubmed_s2":
        return "\n\n".join([_pubmed_search(query, k=k), _s2_search(query, k=k)])
    return _do_search(query, k=k)


class FireModelAdapter:
    def __init__(self, llm: LLMBackend) -> None:
        self._llm = llm
        self.prompts: list[str] = []
        # Prompt caching (opt-in): same text, sent as blocks so each step reuses the last.
        self._blocks = GrowingPrefixBlocks() if llm.prompt_cache else None
        self.cache_read = 0
        self.cache_write = 0

    def generate(self, context: str) -> tuple[str, dict[str, int]]:
        self.prompts.append(context)
        debug: dict = {}
        text, input_tokens, output_tokens = self._llm.complete_text(
            system=FIRE_SYSTEM_PROMPT,
            user=self._blocks(context) if self._blocks else context,
            debug=debug,
        )
        self.cache_read += debug.get("cache_read", 0)
        self.cache_write += debug.get("cache_write", 0)
        return text, {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
