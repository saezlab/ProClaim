"""
Centralised configuration for the Evidence Verification system.

Uses ``pydantic-settings`` to unify:
  - Local YAML parameter files (for reproducible experiments)
  - Environment variables (from ``.env`` at project root)
  - CLI arguments (``--config``, ``--claim``, etc.)
  - Programmatic overrides (constructor kwargs)

Resolution order (highest priority first):
  1. Explicit constructor kwargs / CLI flags
  2. Environment variables (reads ``.env`` automatically)
  3. YAML config file (``--config path/to/config.yaml``)
  4. Field defaults

Usage — library code::

    from pkevolve.verification.config import get_settings
    cfg = get_settings()
    print(cfg.api_key, cfg.model)

Usage — CLI script::

    from pkevolve.verification.config import VerificationSettings
    cfg = VerificationSettings.from_cli()   # parses sys.argv

Usage — from YAML file::

    cfg = VerificationSettings.from_yaml("experiments/mapk1_h3.yaml")

Usage — save config for reproducibility::

    cfg.save_yaml("experiments/mapk1_h3.yaml")

Usage — programmatic override::

    cfg = VerificationSettings(model="gpt-4o", max_iterations=12)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# ---------------------------------------------------------------------------
# Path constants (not configurable — derived from source tree)
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent.parent  # repo root

# ---------------------------------------------------------------------------
# Settings models
# ---------------------------------------------------------------------------


class APISettings(BaseSettings):
    """LLM / external-service API keys and endpoints.

    Reads from env vars automatically (case-insensitive).
    The ``.env`` file at the project root is loaded first.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API keys -----------------------------------------------------------
    glm_api_key: Optional[str] = Field(
        default=None,
        description="Primary API key for Z.AI / GLM models.",
    )
    zai_api_key: Optional[str] = Field(
        default=None,
        description="Alias for glm_api_key (fallback).",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="Fallback OpenAI API key.",
    )

    # Full-text retrieval ------------------------------------------------
    unpaywall_email: str = Field(
        default="pkevolve@example.com",
        description="Email address for Unpaywall API requests.",
    )
    elsevier_api_key: Optional[str] = Field(
        default=None,
        description="Elsevier API key for INDRA full-text via ScienceDirect.",
    )

    # Derived: resolved API key ------------------------------------------
    @property
    def api_key(self) -> str:
        """Resolve the best available API key (GLM > ZAI > OpenAI > EMPTY)."""
        return (
            self.glm_api_key
            or self.zai_api_key
            or self.openai_api_key
            or "EMPTY"
        )


class LLMSettings(BaseSettings):
    """LLM endpoint and model configuration."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
        populate_by_name=True,
    )

    # Agent endpoint (Anthropic-compatible, used by Claude Agent SDK)
    agent_base_url: str = Field(
        default="https://api.z.ai/api/anthropic",
        validation_alias="ANTHROPIC_BASE_URL",
        description="Anthropic-compatible base URL for the outer agent (Claude Agent SDK).",
    )

    # Subagent endpoint (OpenAI-compatible, used by REPL subagents / vLLM)
    subagent_base_url: str = Field(
        default="http://localhost:8000/v1/",
        validation_alias="LLM_BASE_URL",
        description="OpenAI-compatible base URL for subagent LLM calls (vLLM, etc.).",
    )

    # Model identifiers
    model: str = Field(
        default="glm-5",
        description="Model identifier for the outer/main agent.",
    )
    subagent_model: Optional[str] = Field(
        default=None,
        description="Model identifier for inner subagent calls. "
                    "Defaults to `model` when None.",
    )

    # Generation parameters
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature.",
    )

    @property
    def effective_subagent_model(self) -> str:
        """Return subagent_model if set, else fall back to model."""
        return self.subagent_model or self.model


class VerificationSettings(BaseSettings):
    """Top-level configuration merging API, LLM, and workflow settings.

    This is the single settings object that scripts and library code should use.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Nested settings (populated via env vars) ──────────────────────
    api: APISettings = Field(default_factory=APISettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    # ── Verification workflow ─────────────────────────────────────────
    claim: str = Field(
        default="",
        description="Scientific claim to verify. Always provided via CLI "
                    "(--claim).",
    )
    mode: Literal["sdk", "repl"] = Field(
        default="sdk",
        description="Orchestration mode: sdk (Claude Agent SDK + nb_execute) "
                    "or repl (standalone REPL).",
    )

    max_iterations: int = Field(
        default=8,
        ge=1,
        description="Maximum sufficiency-check iterations.",
    )
    sufficiency_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for early stopping.",
    )
    max_turns: int = Field(
        default=30,
        ge=1,
        description="Hard ceiling on LLM turns in REPL mode.",
    )
    max_output_chars: int = Field(
        default=12_000,
        description="Truncate kernel output fed back to LLM.",
    )

    # ── Paths ─────────────────────────────────────────────────────────
    output_dir: Optional[Path] = Field(
        default=None,
        description="Output workspace directory. "
                    "Defaults to results/verification/notebook_demo.",
    )
    notebook_path: Optional[Path] = Field(
        default=None,
        description="Path for the output notebook (Mode A only).",
    )

    # ── Logging ───────────────────────────────────────────────────────
    verbose: bool = Field(
        default=False,
        description="Enable debug logging.",
    )

    # ── Convenience accessors ─────────────────────────────────────────

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def resolved_output_dir(self) -> Path:
        """Return output_dir or the default."""
        if self.output_dir is not None:
            return self.output_dir
        return PROJECT_ROOT / "results" / "verification" / "notebook_demo"

    @property
    def resolved_workspace(self) -> Path:
        return self.resolved_output_dir / "workspace"

    @property
    def resolved_notebook_path(self) -> Path:
        if self.notebook_path is not None:
            return self.notebook_path
        return self.resolved_output_dir / "evidence_report.ipynb"

    # ── Shortcut properties (delegate to nested) ──────────────────────

    @property
    def model(self) -> str:  # type: ignore[override]
        return self.llm.model

    @property
    def subagent_model(self) -> str:
        return self.llm.effective_subagent_model

    @property
    def api_key(self) -> str:
        return self.api.api_key

    @property
    def agent_base_url(self) -> str:
        return self.llm.agent_base_url

    @property
    def subagent_base_url(self) -> str:
        return self.llm.subagent_base_url

    @property
    def temperature(self) -> float:
        return self.llm.temperature

    @property
    def unpaywall_email(self) -> str:
        return self.api.unpaywall_email

    # ── Builders ──────────────────────────────────────────────────────

    def build_sdk_env(self) -> dict[str, str]:
        """Build the environment dict for the Claude Agent SDK process."""
        if self.api_key == "EMPTY":
            raise RuntimeError(
                "GLM_API_KEY not set. Add it to .env at project root.\n"
                "  echo 'GLM_API_KEY=<your-key>' >> .env"
            )
        env = {
            **os.environ,
            "API_TIMEOUT_MS": os.getenv("API_TIMEOUT_MS", "3000000"),
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1",
            "ANTHROPIC_AUTH_TOKEN": self.api_key,
            "ANTHROPIC_BASE_URL": self.agent_base_url,
            "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT": "300000",
        }
        env.pop("ANTHROPIC_API_KEY", None)
        return env

    # ── Factory: from YAML file ───────────────────────────────────────

    @classmethod
    def from_yaml(
        cls, path: str | Path, **overrides,
    ) -> "VerificationSettings":
        """Load settings from a YAML config file.

        Values in the YAML file are merged with env vars and defaults.
        Explicit ``overrides`` keyword arguments take highest priority.

        Note: ``claim`` is intentionally excluded from YAML — it varies
        per-run and should be passed as a keyword override or via CLI.

        Usage::

            cfg = VerificationSettings.from_yaml(
                "experiments/config.yaml",
                claim="Does MAPK1 directly activate H3-3A?",
            )
        """
        if not _YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML config files.  "
                "Install it with: uv pip install pyyaml"
            )
        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path) as fh:
            raw: dict = yaml.safe_load(fh) or {}

        # claim is a per-run argument, not a config parameter — strip it
        # from YAML to avoid accidental reuse across different runs.
        if "claim" in raw:
            import warnings
            warnings.warn(
                "'claim' found in YAML config file and will be ignored. "
                "Pass --claim via CLI or as a keyword override instead.",
                UserWarning,
                stacklevel=2,
            )
            raw.pop("claim")

        # Resolve nested dicts into sub-model instances
        if "llm" in raw and isinstance(raw["llm"], dict):
            raw["llm"] = LLMSettings(**raw["llm"])
        if "api" in raw and isinstance(raw["api"], dict):
            raw["api"] = APISettings(**raw["api"])

        # Convert path strings
        for key in ("output_dir", "notebook_path"):
            if key in raw and isinstance(raw[key], str):
                raw[key] = Path(raw[key])

        # Overrides trump YAML
        raw.update(overrides)

        return cls(**raw)

    # ── Save config snapshot ──────────────────────────────────────────

    def save_yaml(self, path: str | Path) -> Path:
        """Dump current settings to a YAML file for reproducibility.

        API keys are redacted.  The file can be reloaded with ``from_yaml()``.
        """
        if not _YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML config files.  "
                "Install it with: uv pip install pyyaml"
            )
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump(
            mode="json",
            exclude={
                "claim": True,  # per-run CLI argument, not a config param
                "api": {"glm_api_key", "zai_api_key",
                        "openai_api_key", "elsevier_api_key"},
            },
        )

        # Convert Path objects to strings for YAML
        for key in ("output_dir", "notebook_path"):
            if data.get(key) is not None:
                data[key] = str(data[key])

        with open(out_path, "w") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False)

        return out_path

    # ── Factory: from CLI args ────────────────────────────────────────

    @classmethod
    def from_cli(cls, args: Optional[list[str]] = None) -> "VerificationSettings":
        """Parse CLI arguments and env vars into a VerificationSettings.

        This combines argparse (for familiarity / --help) with pydantic-settings
        (for env-var merging).  CLI flags take highest priority.
        """
        import argparse

        parser = argparse.ArgumentParser(
            description="Evidence Verification — RLM dual-mode orchestrator.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "--config", "-c", default=None,
            help="Path to a YAML config file. Values are merged with env vars "
                 "and defaults. CLI flags override config file values.",
        )
        parser.add_argument(
            "--claim", required=True,
            help="Scientific claim to verify.",
        )
        parser.add_argument(
            "--mode", choices=["sdk", "repl"], default=None,
            help="Orchestration mode (default: sdk).",
        )
        parser.add_argument("--model", default=None, help="Model identifier for the outer agent.")
        parser.add_argument(
            "--subagent-model", default=None,
            help="Model for inner subagent calls (defaults to --model).",
        )
        parser.add_argument("--threshold", type=float, default=None, help="Sufficiency confidence threshold.")
        parser.add_argument("--max-iterations", type=int, default=None, help="Max sufficiency iterations.")
        parser.add_argument("--output-dir", default=None, help="Output workspace directory.")
        parser.add_argument("--notebook-path", default=None, help="Notebook path (Mode A only).")
        parser.add_argument("--subagent-base-url", default=None, help="OpenAI-compatible base URL for subagent LLM.")
        parser.add_argument("--agent-base-url", default=None, help="Anthropic-compatible base URL for outer agent.")
        parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Debug logging.")

        parsed = parser.parse_args(args)

        # ── Start from YAML base if provided ──────────────────────────
        yaml_base: dict = {}
        if parsed.config is not None:
            if not _YAML_AVAILABLE:
                raise ImportError(
                    "PyYAML is required for --config.  "
                    "Install it with: uv pip install pyyaml"
                )
            config_path = Path(parsed.config)
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Config file not found: {config_path}"
                )
            with open(config_path) as fh:
                yaml_base = yaml.safe_load(fh) or {}

            # claim is a per-run argument — strip from YAML
            yaml_base.pop("claim", None)

            # Resolve nested dicts into sub-model instances
            if "llm" in yaml_base and isinstance(yaml_base["llm"], dict):
                yaml_base["llm"] = LLMSettings(**yaml_base["llm"])
            if "api" in yaml_base and isinstance(yaml_base["api"], dict):
                yaml_base["api"] = APISettings(**yaml_base["api"])
            for key in ("output_dir", "notebook_path"):
                if key in yaml_base and isinstance(yaml_base[key], str):
                    yaml_base[key] = Path(yaml_base[key])

        # ── Build CLI overrides (only explicitly-provided values) ─────
        overrides: dict = {"claim": parsed.claim}
        if parsed.verbose:
            overrides["verbose"] = True
        if parsed.mode is not None:
            overrides["mode"] = parsed.mode
        if parsed.threshold is not None:
            overrides["sufficiency_threshold"] = parsed.threshold
        if parsed.max_iterations is not None:
            overrides["max_iterations"] = parsed.max_iterations
        if parsed.output_dir is not None:
            overrides["output_dir"] = Path(parsed.output_dir)
        if parsed.notebook_path is not None:
            overrides["notebook_path"] = Path(parsed.notebook_path)

        # LLM overrides
        llm_overrides: dict = {}
        if parsed.model is not None:
            llm_overrides["model"] = parsed.model
        if parsed.subagent_model is not None:
            llm_overrides["subagent_model"] = parsed.subagent_model
        if parsed.subagent_base_url is not None:
            llm_overrides["subagent_base_url"] = parsed.subagent_base_url
        if parsed.agent_base_url is not None:
            llm_overrides["agent_base_url"] = parsed.agent_base_url

        if llm_overrides:
            # Merge with YAML-level llm if present
            if "llm" in yaml_base and isinstance(yaml_base["llm"], LLMSettings):
                base_llm = yaml_base["llm"].model_dump()
                base_llm.update(llm_overrides)
                overrides["llm"] = LLMSettings(**base_llm)
            else:
                overrides["llm"] = LLMSettings(**llm_overrides)

        # Merge: yaml_base is the floor, overrides are the ceiling
        merged = {**yaml_base, **overrides}

        return cls(**merged)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> VerificationSettings:
    """Return the cached global settings instance.

    Reads from env vars / ``.env`` only.  For CLI usage, prefer
    ``VerificationSettings.from_cli()`` and pass the result explicitly.
    """
    return VerificationSettings()
