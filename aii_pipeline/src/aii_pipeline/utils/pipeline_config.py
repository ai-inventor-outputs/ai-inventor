"""
Typed pipeline configuration using Pydantic.

Usage:
    from aii_pipeline.utils import PipelineConfig, rel_path

    # Load from YAML file
    config = PipelineConfig.from_yaml("config/")

    # Access with typed attributes (no .get() chains!)
    api_key = config.api_keys.openrouter
    model = config.gen_hypo.llm_client.model
    timeout = config.gen_hypo.llm_client.llm_timeout

    # Still works with raw dict access if needed
    config.raw["custom_key"]

    # Use rel_path for logging
    rel_path("/home/aii/projects/ai-inventor/runs/foo")  # -> "runs/foo"
"""

from pathlib import Path
from typing import Any, ClassVar

from aii_lib.remote.contracts import DEFAULT_MAX_FILE_SIZE_MB
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config_models import *  # noqa: F403 — all Pydantic config models

# Hard cap per artifact on OpenRouter LLM API spend (USD). Used by both
# PipelineConfig and prompts/components/resources.py — keep them aligned.
DEFAULT_MAX_USD_OPENROUTER_PER_ART = 10.0

# Pipeline-local config files (paths relative to ``aii_config/pipeline/``).
# Each merges at root. ``io/{sources,sinks}.yaml`` carry their own single
# top-level key (``sources:`` / ``sinks:``) so root-merging puts them in
# the right place.
PIPELINE_CONFIG_FILES = (
    "pipeline.yaml",
    "io/sources.yaml",
    "io/sinks.yaml",
)

# Backend config files (relative to ``aii_config/pipeline/harness/``). Each
# file's contents merge under a top-level key matching its filename (without
# the ``.yaml`` suffix). The key is also the canonical PipelineConfig field
# name — so ``harness/execute_env.yaml`` content lands at
# ``cfg.raw["execute_env"]`` and feeds the ``ExecuteEnvConfig`` Pydantic model.
BACKEND_CONFIG_FILES = (
    "agent_backend.yaml",
    "llm_backend.yaml",
    "execute_env.yaml",
)


# =============================================================================
# Path Utilities
# =============================================================================


def get_project_root() -> Path:
    """Get the ai-inventor project root directory."""
    # From aii_pipeline/src/aii_pipeline/utils/pipeline_config.py -> ai-inventor
    return Path(__file__).parent.parent.parent.parent.parent


def rel_path(path: str | Path | None) -> str | None:
    """Convert path to be relative to ai-inventor directory for logging."""
    if path is None:
        return None
    path = Path(path)
    try:
        return str(path.relative_to(get_project_root()))
    except ValueError:
        # If path is not relative to project root, return as-is
        return str(path)


class PipelineConfig(BaseModel):
    """
    Main pipeline configuration with typed access.

    Usage:
        config = PipelineConfig.from_yaml("config/")
        config.api_keys.openrouter
        config.gen_hypo.llm_client.model
    """

    max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB
    max_usd_openrouter_per_art: float = DEFAULT_MAX_USD_OPENROUTER_PER_ART
    # User's research prompt for fresh runs. Settable via the ``--prompt``
    # CLI flag. For fork/resume the same flag carries the per-task user
    # message instead — it doesn't land on this field.
    prompt: str = ""
    # APIKeysConfig is the dataclass from aii_lib.config (single source).
    # Default reads from env; YAML-provided dicts merge with env via from_dict.
    api_keys: APIKeysConfig = Field(default_factory=APIKeysConfig.from_env)

    @field_validator("api_keys", mode="before")
    @classmethod
    def _build_api_keys(cls, v: Any) -> Any:
        if v is None or v == {}:
            return APIKeysConfig.from_env()
        if isinstance(v, APIKeysConfig):
            return v
        if isinstance(v, dict):
            return APIKeysConfig.from_dict(v)
        return v

    sinks: SinksConfig = Field(default_factory=SinksConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    init: InitConfig = Field(default_factory=InitConfig)
    execute_env: ExecuteEnvConfig = Field(default_factory=ExecuteEnvConfig)
    seed_hypo: SeedHypoConfig = Field(default_factory=SeedHypoConfig)
    gen_hypo_loop: GenHypoLoopConfig = Field(default_factory=GenHypoLoopConfig)
    audit_hypo: AuditHypoConfig = Field(default_factory=AuditHypoConfig)
    invention_loop: InventionLoopConfig = Field(default_factory=InventionLoopConfig)
    gen_paper_repo: GenPaperConfig = Field(default_factory=GenPaperConfig)

    @property
    def gen_hypo(self) -> GenHypoConfig:
        """Shortcut: config.gen_hypo → config.gen_hypo_loop.gen_hypo."""
        return self.gen_hypo_loop.gen_hypo

    @property
    def review_hypo(self) -> ReviewHypoConfig:
        """Shortcut: config.review_hypo → config.gen_hypo_loop.review_hypo."""
        return self.gen_hypo_loop.review_hypo

    # Keep raw dict for any custom/unknown keys
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)

    model_config = ConfigDict(extra="allow")

    @property
    def outputs_directory(self) -> str:
        """Resolve outputs directory based on exec env mode."""
        return self.init.resolve_outputs_directory(self.execute_env.mode)

    @classmethod
    def from_yaml(cls, *config_dirs: str | Path) -> "PipelineConfig":
        """Load configuration from N config directories, deep-merged in order.

        Layers, applied left to right (later wins, deep-merged):

          0. ``aii_config/pipeline/`` — canonical defaults (always layer 0).
          1..N. ``*config_dirs`` — additional dirs supplied by the caller.
                Each must mirror the canonical layout: a ``pipeline.yaml``,
                an ``io/{sources,sinks}.yaml``, and/or
                ``harness/{agent_backend,llm_backend,execute_env}.yaml``.
                Missing files are skipped (no error). ``.private.yaml``
                siblings deep-merge on top of each public file via
                ``load_config_with_overrides``.

        Typical use:
            PipelineConfig.from_yaml()                         # canonical only
            PipelineConfig.from_yaml(user_config_dir)          # + per-user
            PipelineConfig.from_yaml(user_config_dir,          # + per-user
                                     experiment_dir)           # + experiment overlay
        """
        from aii_lib.utils.config_overrides import load_config_with_overrides

        repo_root = get_project_root()
        canonical = repo_root / "aii_config" / "pipeline"

        layers: list[Path] = [canonical]
        for d in config_dirs:
            if d is None or d == "":
                continue
            p = Path(d)
            if p.is_dir():
                layers.append(p)
            elif p:
                import warnings

                warnings.warn(
                    f"PipelineConfig.from_yaml: {p!s} is not a directory; ignoring",
                    stacklevel=2,
                )

        raw_config: dict[str, Any] = {}

        for layer_dir in layers:
            # harness/<name>.yaml — each lands at a top-level key.
            for filename in BACKEND_CONFIG_FILES:
                path = layer_dir / "harness" / filename
                if not path.exists():
                    continue
                data = load_config_with_overrides(path) or {}
                top_key = filename.removesuffix(".yaml")
                raw_config[top_key] = cls._deep_merge(raw_config.get(top_key, {}) or {}, data)
            # pipeline.yaml + io/*.yaml — deep-merge at root.
            for filename in PIPELINE_CONFIG_FILES:
                path = layer_dir / filename
                if not path.exists():
                    continue
                data = load_config_with_overrides(path) or {}
                raw_config = cls._deep_merge(raw_config, data)

        # ---------------------------------------------------------------------
        # Apply defaults to every per-step ``claude_agent:`` block.
        #
        # Three merge sources, applied in this priority order (later loses
        # to earlier, i.e. step values > llm_backend defaults > agent_backend
        # defaults):
        #
        #   1. The block's own keys (highest precedence).
        #   2. ``llm_backend.active`` filled in if the block didn't pick one.
        #   3. ``llm_backend.<that>.defaults`` (model, effort, …) merged in.
        #   4. ``agent_backend.<active>.defaults`` (timeouts, retries, tools, …)
        #      merged in (lowest precedence).
        #
        # Steps that override ``llm_backend`` automatically pick up that
        # backend's default ``model`` etc., so swapping a step from claude_max
        # to openrouter doesn't require also rewriting the model name.
        # ---------------------------------------------------------------------
        agent_backends = raw_config.get("agent_backend", {})
        llm_backends = raw_config.get("llm_backend", {})
        active_agent_backend = agent_backends.pop("active", "claude_agent_sdk")
        active_llm_backend = llm_backends.pop("active", "claude_max")

        agent_defaults: dict = (
            agent_backends.get(active_agent_backend, {}).pop("defaults", {}) or {}
        )
        llm_defaults_by_name: dict[str, dict] = {
            name: (cfg.pop("defaults", {}) or {})
            for name, cfg in llm_backends.items()
            if isinstance(cfg, dict)
        }

        cls._apply_backend_defaults_to_claude_agent(
            raw_config,
            active_llm_backend=active_llm_backend,
            agent_defaults=agent_defaults,
            llm_defaults_by_name=llm_defaults_by_name,
        )

        cls._validate_backend_pairings(raw_config)

        # Apply pod defaults from execute_env.runpod to every executor's claude_agent
        runpod_cfg = raw_config.get("execute_env", {}).get("runpod", {})
        pod_defaults = {}
        if runpod_cfg.get("pod_start_retries") is not None:
            pod_defaults["pod_start_retries"] = runpod_cfg["pod_start_retries"]
        if runpod_cfg.get("pod_timeout") is not None:
            pod_defaults["pod_timeout"] = runpod_cfg["pod_timeout"]
        if pod_defaults:
            cls._apply_defaults_recursive(raw_config, "claude_agent", pod_defaults)

        instance = cls.model_validate(raw_config)
        instance.raw = raw_config
        return instance

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PipelineConfig":
        """Load configuration from a pre-built dict (used by tests + tools)."""
        instance = cls.model_validate(config_dict)
        instance.raw = config_dict
        return instance

    @classmethod
    def _apply_backend_defaults_to_claude_agent(
        cls,
        d: dict,
        *,
        active_llm_backend: str,
        agent_defaults: dict,
        llm_defaults_by_name: dict[str, dict],
    ) -> None:
        """Fill every per-step ``claude_agent:`` block with backend defaults.

        Applies to ``claude_agent:`` and any ``*_claude_agent:`` block.
        First pulls in ``llm_backend`` + the matching backend's defaults,
        then the agent_backend's defaults. Block-level fields (incl. an
        explicit ``llm_backend:``) win; per-step ``llm_backend`` overrides
        the active default before its backend's defaults get pulled in.
        """
        for key, value in d.items():
            if isinstance(value, dict):
                is_claude_agent_block = key == "claude_agent" or key.endswith("_claude_agent")
                if is_claude_agent_block:
                    # Step's effective llm_backend: explicit field wins,
                    # otherwise fall back to active. Fill the field if missing
                    # so downstream code can read it without ambiguity.
                    eff_llm = value.setdefault("llm_backend", active_llm_backend)
                    # Per-llm_backend model defaults (lower precedence than
                    # block fields, higher than agent_backend defaults).
                    for dk, dv in (llm_defaults_by_name.get(eff_llm) or {}).items():
                        value.setdefault(dk, dv)
                    # Per-agent_backend transport defaults (lowest precedence).
                    for dk, dv in agent_defaults.items():
                        value.setdefault(dk, dv)
                cls._apply_backend_defaults_to_claude_agent(
                    value,
                    active_llm_backend=active_llm_backend,
                    agent_defaults=agent_defaults,
                    llm_defaults_by_name=llm_defaults_by_name,
                )
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        cls._apply_backend_defaults_to_claude_agent(
                            item,
                            active_llm_backend=active_llm_backend,
                            agent_defaults=agent_defaults,
                            llm_defaults_by_name=llm_defaults_by_name,
                        )

    # Supported pairings between agent_backend (the orchestrator that drives
    # the per-step tool loop) and llm_backend (the model transport). Anything
    # not listed here is rejected at config load with a clear error rather
    # than failing mid-run with confusing translation/proxy errors.
    #
    # Today: claude_agent_sdk only works against api.anthropic.com — i.e.
    # claude_max. Bridging it to OpenAI-style endpoints via the LiteLLM
    # proxy looks superficially fine but Anthropic's server-side tools
    # (``web_search_20250305`` etc.) don't translate to OpenAI's function
    # call schema, every WebSearch call returns 400, and the agent runs out
    # of retries without ever calling its structured-output tool.
    #
    # The openrouter llm_backend has no compatible agent_backend yet —
    # callers go through ``_run_task_openrouter`` (direct OpenRouter client
    # with response_format) which doesn't use ``claude_agent:`` blocks at
    # all. Per-step ``use_claude_agent: false`` flips substeps onto that
    # path; this validator catches the case where someone leaves
    # ``claude_agent:`` blocks in place but flips llm_backend to
    # openrouter, which won't work today.
    _SUPPORTED_BACKEND_PAIRS: ClassVar[set[tuple[str, str]]] = {
        ("claude_agent_sdk", "claude_max"),
    }

    @classmethod
    def _validate_backend_pairings(cls, raw: dict) -> None:
        """Reject any per-step claude_agent block with an unsupported llm_backend.

        Walks the tree looking for ``claude_agent:`` (or
        ``*_claude_agent:``) blocks; each carries an ``llm_backend`` field
        post-defaults. The agent_backend driving them is
        ``agent_backend.active`` (already popped by the time we run, so we
        re-read from raw — the active backend is implicit:
        ``claude_agent_sdk`` is the only one today).
        """
        agent_backend = "claude_agent_sdk"  # only one supported today
        bad: list[tuple[str, str]] = []  # (path, llm_backend)

        def _walk(node: object, path: str) -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    if (k == "claude_agent" or k.endswith("_claude_agent")) and isinstance(v, dict):
                        llmb = v.get("llm_backend")
                        if (
                            isinstance(llmb, str)
                            and (agent_backend, llmb) not in cls._SUPPORTED_BACKEND_PAIRS
                        ):
                            bad.append((f"{path}.{k}", llmb))
                    _walk(v, f"{path}.{k}" if path else k)
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    _walk(item, f"{path}[{i}]")

        _walk(raw, "")
        if bad:
            supported = ", ".join(f"{ab}+{lb}" for (ab, lb) in cls._SUPPORTED_BACKEND_PAIRS)
            offenders = "\n".join(f"  - {p}: llm_backend={lb}" for p, lb in bad)
            raise ValueError(
                f"Unsupported agent_backend / llm_backend pairing.\n"
                f"Today {agent_backend} only works with: {supported}.\n"
                f"Offending claude_agent: blocks:\n{offenders}\n\n"
                f"To use openrouter for these steps, drop the claude_agent: "
                f"block and configure the openrouter direct path "
                f"(``use_claude_agent: false`` + step-level model/effort) "
                f"instead. The openrouter llm_backend has no compatible "
                f"agent_backend yet."
            )

    @classmethod
    def _apply_defaults_recursive(cls, d: dict, block_name: str, defaults: dict) -> None:
        """Apply defaults to every nested dict named or suffixed with ``block_name``.

        Walks the config tree.  When a dict key equals ``block_name`` or
        ends with ``_block_name`` (e.g. claude_agent, review_claude_agent)
        and its value is a dict, any missing keys are filled from ``defaults``.
        Existing values are NOT overwritten.
        """
        for key, value in d.items():
            if isinstance(value, dict):
                if key == block_name or key.endswith(f"_{block_name}"):
                    for dk, dv in defaults.items():
                        value.setdefault(dk, dv)
                cls._apply_defaults_recursive(value, block_name, defaults)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        cls._apply_defaults_recursive(item, block_name, defaults)

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Deep merge override dict into base dict."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = PipelineConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def to_dict(self) -> dict:
        """Export configuration as dict."""
        return self.model_dump(exclude={"raw"})
