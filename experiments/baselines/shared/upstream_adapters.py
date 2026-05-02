from __future__ import annotations

import importlib
import json
import re
import sys
import types
from pathlib import Path

from baselines.shared.llm import LLMBackend

_SHARED_DIR = Path(__file__).resolve().parent
_BASELINES_DIR = _SHARED_DIR.parent
_EXPERIMENTS_DIR = _BASELINES_DIR.parent

FIRE_REPO_ROOT = _EXPERIMENTS_DIR / "fire"
SAFE_REPO_ROOT = _EXPERIMENTS_DIR / "long-form-factuality"
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


def do_search(query: str, k: int) -> str:
    from baselines.react_baseline import _do_search

    return _do_search(query, k=k)


class FireModelAdapter:
    def __init__(self, llm: LLMBackend) -> None:
        self._llm = llm
        self.prompts: list[str] = []

    def generate(self, context: str) -> tuple[str, dict[str, int]]:
        self.prompts.append(context)
        text, input_tokens, output_tokens = self._llm.complete_text(
            system=FIRE_SYSTEM_PROMPT,
            user=context,
        )
        return text, {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


class SafeModelAdapter:
    def __init__(self, llm: LLMBackend) -> None:
        self._llm = llm
        self.prompts: list[str] = []
        self.input_tokens = 0
        self.output_tokens = 0

    def generate(
        self,
        prompt: str,
        do_debug: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_attempts: int = 1000,
        timeout: int = 60,
        retry_interval: int = 10,
    ) -> str:
        del do_debug, temperature, max_tokens, max_attempts, timeout, retry_interval
        self.prompts.append(prompt)
        text, input_tokens, output_tokens = self._llm.complete_text(system="", user=prompt)
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        return text