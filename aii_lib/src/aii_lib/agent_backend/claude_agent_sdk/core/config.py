"""Configuration building and initialization."""

import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions

from aii_lib.run import emit

from ..models import AgentOptions, SessionType

# Per-task suppression for system_prompt and config re-emission. Multiple
# non-resume prompts inside one task otherwise re-emit identical
# system_prompt/config events; these sets track which task_ids have already
# emitted each event type. Entries linger after the task ends but the
# universe of task_ids is bounded per pipeline run.
_emitted_sprompt_for_task: set[str] = set()
_emitted_for_task_lock = threading.Lock()


def _mark_first_emit(task_id: str | None, store: set[str]) -> bool:
    """Return True if this task_id has not yet emitted; mark as emitted.

    Returns False if the task_id has already emitted (caller should suppress).
    Empty/None task_ids always return True (no dedup possible — caller
    proceeds with the emit).
    """
    if not task_id:
        return True
    with _emitted_for_task_lock:
        if task_id in store:
            return False
        store.add(task_id)
        return True


def initialize_agent(options: AgentOptions) -> ClaudeAgentOptions:
    """Initialize agent (one-time setup when Agent is created).

    Builds SDK options by:
    - Parsing agent markdown files to AgentDefinition dataclasses
    - Preparing workspace (.claude/ directories)
    - Setting up MCP configurations
    - Converting AgentOptions to ClaudeAgentOptions

    Args:
        options: Agent configuration options

    Returns:
        SDK-ready ``ClaudeAgentOptions`` with workspace, agents, skills,
        and MCP servers wired in.
    """
    # Prepare workspace and parse agents/skills/MCPs
    cwd_path = Path(options.cwd).resolve() if options.cwd else Path.cwd()
    rid = options.run_id  # Shorthand for run_id

    # Convert selected agents to programmatic definitions (no copying to workspace)
    # This prevents SDK from discovering unwanted agents from parent directories
    if options.selected_agents:
        from ..utils.init_helpers import get_agent
        from ..utils.init_helpers.agent_parser import parse_agent_markdown

        emit.status_private_debug(
            f"Processing selected_agents: {options.selected_agents}", run_id=rid
        )
        emit.status_private_debug(f"options.agents before: {options.agents}", run_id=rid)

        if not options.agents:
            options.agents = {}

            for agent in options.selected_agents:
                if isinstance(agent, str):
                    # Resolve string name to AgentDefinition from registry
                    agent_def_obj = get_agent(agent)
                    emit.status_private_debug(
                        f"get_agent('{agent}') returned: {agent_def_obj}", run_id=rid
                    )

                    if agent_def_obj:
                        # Parse the source .md file to get SDK AgentDefinition
                        agent_def = parse_agent_markdown(agent_def_obj.path)
                        emit.status_private_debug(
                            f"Parsed agent '{agent}': {agent_def}", run_id=rid
                        )
                        options.agents[agent] = agent_def
                        emit.status_public_success(
                            f"Added programmatic agent '{agent}' to options.agents",
                            run_id=rid,
                        )
                    else:
                        emit.status_public_warning(
                            f"Agent '{agent}' not found in registry, skipping",
                            run_id=rid,
                        )
                else:
                    # Already an SDK AgentDefinition object
                    options.agents[agent.name] = agent
                    emit.status_public_success(
                        f"Added programmatic agent '{agent.name}' (already SDK object)",
                        run_id=rid,
                    )

        emit.status_private_info(
            f"Final options.agents: {len(options.agents) if options.agents else 0} agents",
            run_id=rid,
        )

    # Prepare MCPs to workspace
    mcp_config_path = None
    if options.selected_mcps:
        from ..utils.init_helpers import prepare_mcps

        mcp_config_path = prepare_mcps(
            options.selected_mcps,
            cwd=cwd_path,
            run_id=options.run_id,
        )

    # Auto-set mcp_servers if MCPs were prepared
    if mcp_config_path and not options.mcp_servers:
        options.mcp_servers = str(mcp_config_path)

    # Determine session parameters based on session_type enum
    if options.session_type == SessionType.NEW:
        resume_id = None
        fork = False
    elif options.session_type == SessionType.RESUME:
        resume_id = options.resume_session_id
        fork = False
    elif options.session_type == SessionType.FORK:
        resume_id = options.resume_session_id
        fork = True
    else:
        raise ValueError(f"Invalid session_type: {options.session_type}")

    # Handle custom tools
    mcp_servers = options.mcp_servers
    if options.custom_tool_files:
        from ..utils.init_helpers.mcp_tools import setup_custom_tools

        # Load custom tools and create SDK MCP server
        custom_server_config = setup_custom_tools(
            options.custom_tool_files,
            run_id=rid,
        )

        # Merge with existing mcp_servers
        if isinstance(mcp_servers, dict):
            mcp_servers = {**mcp_servers, **custom_server_config}
        else:
            # If mcp_servers is a path string, keep it and log warning
            emit.status_public_warning(
                "custom_tool_files specified but mcp_servers is a path string. "
                "Custom tools will be ignored. Use dict format for mcp_servers.",
                run_id=rid,
            )

    agents = options.agents

    # Build SDK options dict
    # Convert cwd to absolute path for Skills to work
    cwd_path = Path(options.cwd).resolve() if options.cwd else Path.cwd()

    # SDK telemetry on/off switch — flip CLAUDE_CODE_ENABLE_TELEMETRY in the
    # subprocess env. The OTLP transport vars (endpoint, headers, protocol,
    # exporter selection) come from process-wide setup in
    # ``aii_lib.agent_backend.claude_agent_sdk.sdk_telemetry.configure_sdk_telemetry`` so the
    # only per-call decision is "emit or don't." Caller-provided env entries
    # win over our default, in case a step explicitly sets ``=0``.
    sdk_env = dict(options.env)
    if options.telemetry:
        sdk_env.setdefault("CLAUDE_CODE_ENABLE_TELEMETRY", "1")

    # Per-step llm_backend routing. Today only ``claude_max`` is supported
    # for the SDK path; ``PipelineConfig._validate_backend_pairings``
    # rejects other combos at config load. ``get_adapter`` raises for
    # unsupported llm_backends — we treat that as "skip env injection"
    # so direct programmatic AgentOptions construction (tests, ad-hoc
    # scripts) can still build the options without exploding here, with
    # the validator catching real misuse upstream. claude_max's adapter
    # returns an empty dict (CLI's own OAuth flow handles auth), so this
    # branch is effectively dormant unless a future llm_backend wires in.
    from aii_pipeline.utils.context import get_pipeline_config

    from aii_lib.llm_backend._adapter import get_adapter

    pcfg = get_pipeline_config()
    if pcfg is not None:
        try:
            adapter = get_adapter(options.llm_backend)
        except (ValueError, NotImplementedError):
            adapter = None
        if adapter is not None:
            llm_cfg = pcfg.raw.get("llm_backend", {}).get(options.llm_backend, {}) or {}
            for k, v in adapter.env_for_sdk(llm_cfg).items():
                sdk_env.setdefault(k, v)

    # continue_conversation must be True when forking/resuming a session,
    # otherwise the SDK CLI crashes with exit code 1
    # NOTE: system_prompt is NOT passed to SDK - it's emitted as S_PROMPT in initialize_execution
    options_dict = {
        "allowed_tools": options.allowed_tools,
        # system_prompt intentionally omitted - emitted as S_PROMPT before PROMPT
        "permission_mode": options.permission_mode,
        "continue_conversation": True
        if fork or resume_id
        else None,  # Required for fork/resume; None filtered out for new sessions
        "max_turns": options.max_turns,
        "model": options.model,
        "cwd": str(cwd_path),
        "resume": resume_id,
        "fork_session": fork,
        "disallowed_tools": options.disallowed_tools,
        "mcp_servers": mcp_servers,
        "permission_prompt_tool_name": options.permission_prompt_tool_name,
        "settings": options.settings,
        "add_dirs": [str(d) for d in options.add_dirs],
        "env": sdk_env,
        "effort": options.effort,
        "extra_args": options.extra_args,
        "max_buffer_size": options.max_buffer_size,
        "include_partial_messages": options.include_partial_messages,
        "agents": agents,
        "setting_sources": options.setting_sources,
        "output_format": options.output_format,
        "hooks": options.hooks,
    }

    # Filter out None values
    options_dict = {key: value for key, value in options_dict.items() if value is not None}

    # Create SDK options
    sdk_options = ClaudeAgentOptions(**options_dict)

    # Note: agent_config emission is owned by the SDK-side stream
    # (claude_agent_tel_adapter.py maps the SDK's ``system`` message to
    # AgentConfigMessage). The previous pipeline-side emit ran a few
    # seconds before that, producing a duplicate row per task. The
    # adapter version carries the live session_id + tool counts, which
    # the pipeline-side snapshot can't know yet — so it's the more
    # informative single source of truth.
    sys.stdout.flush()
    return sdk_options


def initialize_execution(
    options: AgentOptions,
    prompt: str,
    prompt_index: int,
) -> dict:
    """Initialize execution for a single prompt.

    Emits S_PROMPT (system prompt) followed by PROMPT (user prompt),
    matching the pattern used by other LLM backends (OpenRouter, etc.).

    Args:
        options: Agent configuration options
        prompt: The prompt text to execute
        prompt_index: Index of this prompt in the sequence

    Returns:
        execution_state dict
    """
    # Execution state (tracks model, timing, etc.)
    # Use options.model as initial model (fallback if SDK doesn't provide it)
    execution_state = {
        "prompt_index": prompt_index,
        "current_model": options.model,  # Initialize with options.model as fallback
        "module_start_time": datetime.now(UTC).isoformat(),
        "message_count": 0,
        "custom_metadata": options.custom_metadata or {},
        "run_id": options.run_id,  # For sequenced parallel execution
        "agent_context": options.agent_context,  # Display name for logs (e.g., "data-0")
    }

    # Emit S_PROMPT (system prompt) before PROMPT - matches OpenRouter pattern.
    # Skip on resumed/forked sessions: the system prompt was already emitted
    # in the original conversation and is part of the SDK session being
    # resumed, so re-emitting it would double the SYSTEM PROMPT pill in the
    # feed for what is functionally one continuous conversation.
    # Also skip if this task has already emitted the system prompt — protects
    # against multi-prompt sequences re-emitting it for prompt_index 0 on a
    # second call (which can happen after retries/forks within the same task).
    is_resuming = options.session_type in (
        SessionType.FORK,
        SessionType.RESUME,
    ) and bool(options.resume_session_id)
    # System + user prompt emission goes through the SDK tel_adapter
    # so every agent_* event in the sinks/clone/clone_log.jsonl stream lands via
    # the same single path: build a raw dict, run it through ``adapt``,
    # emit the typed result via ``run._on``. This matches what the SDK
    # message_callback does for ``claude_msg`` / ``thinking`` / etc.
    from aii_lib.agent_backend.claude_agent_sdk.claude_agent_tel_adapter import (
        adapt as _adapt_claude,
    )
    from aii_lib.run import get_current_run

    _task_id = options.run_id or ""
    _task_name = options.agent_context or ""

    def _emit_via_adapter(raw: dict) -> None:
        run = get_current_run()
        if run is None:
            return
        run._on(_adapt_claude(raw, _task_id, _task_name))

    _sp_task_id = options.run_id or ""
    _first_for_task = _mark_first_emit(_sp_task_id, _emitted_sprompt_for_task)
    if options.system_prompt and prompt_index == 0 and not is_resuming and _first_for_task:
        # Only emit system prompt on first prompt of a sequence
        system_prompt_text = options.system_prompt
        if isinstance(system_prompt_text, dict):
            # Handle preset dict format
            system_prompt_text = str(system_prompt_text)

        _emit_via_adapter(
            {
                "type": "s_prompt",
                "text": system_prompt_text,
                "prompt_index": prompt_index,
                "backend": "claude_agent",
                "extras": {"model": options.model},
            }
        )
        execution_state["message_count"] += 1

    # User-prompt ``agent_user_prompt`` events are emitted by the SDK echo
    # path in :mod:`.utils.execution.sdk_client` — the SDK echoes each
    # user message back through ``receive_response`` (with
    # ``replay-user-messages`` enabled), at which point we know the API
    # has actually received and processed it. We still bump
    # ``message_count`` here so accounting elsewhere stays consistent.
    execution_state["message_count"] += 1

    # Reset ``prompt_source`` after the inject's first prompt — it's a
    # single-shot tag attached to the AgentOptions; without this clamp
    # later SDK calls would re-tag their initial prompt as user-written.
    if prompt_index == 0 and options.prompt_source != "pipeline":
        options.prompt_source = "pipeline"

    return execution_state


def prepare_sdk_options(
    options: AgentOptions,
    session_id: str | None,
    last_failure_reason: str | None,
    prompt_index: int,
    original_prompt: str,
    sdk_options_cache: Any,
    with_output_format: bool,
) -> tuple:
    """Prepare SDK options based on prompt index and session state.

    Returns (sdk_options, effective_prompt, sdk_options_cache).
    """
    from dataclasses import replace

    from .prompts import build_continue_prompt

    if prompt_index == 0:
        if session_id:
            # RETRY: fork from timed-out session
            if sdk_options_cache is None:
                saved = options.output_format
                options.output_format = None
                sdk_options = initialize_agent(options)
                options.output_format = saved
                sdk_options_cache = sdk_options
            sdk_options = replace(
                sdk_options_cache,
                resume=session_id,
                fork_session=False,
                continue_conversation=True,
            )
            effective_prompt = build_continue_prompt(
                original_prompt,
                last_failure_reason,
                options,
            )
            from aii_lib.run import get_current_run

            run = get_current_run()
            if run is not None:
                emit.status_private_debug(
                    f"Resuming from session {session_id[:12]}... | "
                    f"model={sdk_options.model} | "
                    f"max_turns={sdk_options.max_turns} | "
                    f"permission_mode={sdk_options.permission_mode}",
                )
        else:
            # First attempt: fresh SDK options
            saved = options.output_format
            options.output_format = None
            sdk_options = initialize_agent(options)
            options.output_format = saved
            sdk_options_cache = sdk_options
            effective_prompt = original_prompt
    else:
        # Subsequent prompts: plain resume from previous session (no fork)
        from dataclasses import replace

        resume_id = session_id if options.continue_seq_item else None
        sdk_options = replace(
            sdk_options_cache,
            resume=resume_id,
            fork_session=False,
            continue_conversation=bool(resume_id),
        )
        effective_prompt = original_prompt

    if with_output_format and options.output_format:
        sdk_options = replace(sdk_options, output_format=options.output_format)

    return sdk_options, effective_prompt, sdk_options_cache
