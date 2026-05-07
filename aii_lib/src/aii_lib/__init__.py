"""
aii_lib - GenAI toolkit for ai-inventor pipeline.

Structure:
├── run/                - Run-object architecture (typed messages, sinks, sources)
├── utils/              - General utilities (ForkingServer, call_server, server_available)
├── workflows/          - Multi-step orchestrations (research_workflow, gen_kg)
├── llm_backend/        - LLM clients (OpenAI, Anthropic, Gemini, OpenRouter)
├── agent_backend/      - Claude Agent SDK wrapper
└── abilities/          - @aii_ability decorator, ability server, skills

Lazy imports:
- Subpackages can be imported directly without triggering the full
  import chain: `from aii_lib.utils import call_server`
- Top-level imports are loaded lazily on first access
"""

import importlib
from typing import TYPE_CHECKING

# For type checking, import everything statically
if TYPE_CHECKING:
    from .abilities.aii_ability import (
        abilities_to_openai_tools,
        ability_to_openai_tool,
    )
    from .agent_backend import (
        Agent,
        AgentOptions,
        AgentResponse,
        ExpectedFile,
        SessionType,
    )
    from .agent_backend.utils import (
        build_options,
        chain_validators,
        check_oversized_files,
        copy_dependencies,
        end_task,
        end_task_error,
        end_task_failure,
        end_task_success,
        end_task_timeout,
        ensure_servers,
        gen_dependency_prompt,
        generate_requirements,
        get_oversized_files_prompt,
        make_file_size_validator,
        read_metadata,
        setup_workspace,
        start_task,
    )
    from .llm_backend import (
        ConversationStats,
        OpenRouterClient,
        ToolLoopResult,
        chat,
    )
    from .prompts import (
        LLMPromptModel,
        LLMStructOutModel,
    )
    from .utils import (
        ClaudeAgentToLLMStructOut,
        ClaudeAgentToLLMStructOutResult,
        call_server,
        cleanup_run_caches,
        get_model_short,
        server_available,
    )
    from .workflows import (
        RESEARCH_TOOLS,
        GenKGConfig,
        GenKGResult,
        ResearchWorkflowConfig,
        ResearchWorkflowResult,
        generate_kg_triples,
        research_workflow,
        verify_wikipedia_urls,
    )
    # aii_runpod is an optional extra (``pip install ai-inventor[runpod]``).
    # Public/open-source builds don't ship that package, so don't pretend
    # the names exist for type checkers either — runtime callers reach
    # them via ``_LAZY_IMPORTS`` below, which raises ImportError when the
    # extra isn't installed.


# Lazy import mapping: attribute name -> (module, attribute)
_LAZY_IMPORTS = {
    # utils
    "call_server": (".utils", "call_server"),
    "server_available": (".utils", "server_available"),
    "cleanup_run_caches": (".utils", "cleanup_run_caches"),
    "get_model_short": (".utils", "get_model_short"),
    "LLMPromptModel": (".prompts", "LLMPromptModel"),
    "LLMStructOutModel": (".prompts", "LLMStructOutModel"),
    "ClaudeAgentToLLMStructOut": (".utils", "ClaudeAgentToLLMStructOut"),
    "ClaudeAgentToLLMStructOutResult": (".utils", "ClaudeAgentToLLMStructOutResult"),
    # workflows - Research OR
    "research_workflow": (".workflows", "research_workflow"),
    "ResearchWorkflowConfig": (".workflows", "ResearchWorkflowConfig"),
    "ResearchWorkflowResult": (".workflows", "ResearchWorkflowResult"),
    "RESEARCH_TOOLS": (".workflows", "RESEARCH_TOOLS"),
    # workflows - Gen KG
    "GenKGConfig": (".workflows", "GenKGConfig"),
    "GenKGResult": (".workflows", "GenKGResult"),
    "generate_kg_triples": (".workflows", "generate_kg_triples"),
    "verify_wikipedia_urls": (".workflows", "verify_wikipedia_urls"),
    # abilities → OpenAI tool schema
    "abilities_to_openai_tools": (
        ".abilities.aii_ability",
        "abilities_to_openai_tools",
    ),
    "ability_to_openai_tool": (".abilities.aii_ability", "ability_to_openai_tool"),
    # llm_backend exports — OpenRouter direct client only (per-provider clients
    # have been removed). Two pipeline consumption paths:
    #   - claude_max: Claude Agent SDK against api.anthropic.com (full agentic loop)
    #   - openrouter: direct OpenRouterClient.chat (no agent loop)
    # The claude_agent_sdk + openrouter combo is rejected at config load —
    # see aii_pipeline.utils.pipeline_config._validate_backend_pairings.
    "OpenRouterClient": (".llm_backend", "OpenRouterClient"),
    "ConversationStats": (".llm_backend", "ConversationStats"),
    "chat": (".llm_backend", "chat"),
    "ToolLoopResult": (".llm_backend", "ToolLoopResult"),
    # agent_backend
    "Agent": (".agent_backend", "Agent"),
    "AgentOptions": (".agent_backend", "AgentOptions"),
    "ExpectedFile": (".agent_backend", "ExpectedFile"),
    "AgentResponse": (".agent_backend", "AgentResponse"),
    "SessionType": (".agent_backend", "SessionType"),
    # agent utilities (module-level helpers, formerly AgentInitializer/AgentFinalizer)
    "setup_workspace": (".agent_backend.utils", "setup_workspace"),
    "copy_dependencies": (".agent_backend.utils", "copy_dependencies"),
    "gen_dependency_prompt": (".agent_backend.utils", "gen_dependency_prompt"),
    "ensure_servers": (".agent_backend.utils", "ensure_servers"),
    "build_options": (".agent_backend.utils", "build_options"),
    "start_task": (".agent_backend.utils", "start_task"),
    "end_task": (".agent_backend.utils", "end_task"),
    "end_task_success": (".agent_backend.utils", "end_task_success"),
    "end_task_failure": (".agent_backend.utils", "end_task_failure"),
    "end_task_timeout": (".agent_backend.utils", "end_task_timeout"),
    "end_task_error": (".agent_backend.utils", "end_task_error"),
    "check_oversized_files": (".agent_backend.utils", "check_oversized_files"),
    "get_oversized_files_prompt": (
        ".agent_backend.utils",
        "get_oversized_files_prompt",
    ),
    "read_metadata": (".agent_backend.utils", "read_metadata"),
    "generate_requirements": (".agent_backend.utils", "generate_requirements"),
    "chain_validators": (".agent_backend.utils", "chain_validators"),
    "make_file_size_validator": (".agent_backend.utils", "make_file_size_validator"),
    # remote execution
    "RunPodClient": ("aii_runpod", "RunPodClient"),
    "WorkerPod": ("aii_runpod", "WorkerPod"),
    "OrchestratorClient": ("aii_runpod", "OrchestratorClient"),
    "ComputeProfile": ("aii_runpod", "ComputeProfile"),
    "PodInfo": ("aii_runpod", "PodInfo"),
    "PodFailureInfo": ("aii_runpod", "PodFailureInfo"),
    "PodExecutionError": ("aii_runpod", "PodExecutionError"),
}


def __getattr__(name: str) -> object:
    """Lazy import handler for top-level attributes."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package="aii_lib")
        value = getattr(module, attr_name)
        # Cache the imported value
        globals()[name] = value
        return value
    raise AttributeError(f"module 'aii_lib' has no attribute {name!r}")


__all__ = [
    # Utils
    "call_server",
    "server_available",
    "cleanup_run_caches",
    "get_model_short",
    "LLMPromptModel",
    "LLMStructOutModel",
    "ClaudeAgentToLLMStructOut",
    "ClaudeAgentToLLMStructOutResult",
    # Workflows - Research OR (OpenRouter + tools)
    "research_workflow",
    "ResearchWorkflowConfig",
    "ResearchWorkflowResult",
    "RESEARCH_TOOLS",
    # Workflows - Gen KG (Knowledge Graph Generation)
    "GenKGConfig",
    "GenKGResult",
    "generate_kg_triples",
    "verify_wikipedia_urls",
    # Abilities → OpenAI tool schema
    "abilities_to_openai_tools",
    "ability_to_openai_tool",
    # LLM Clients
    "OpenRouterClient",
    "ConversationStats",
    "chat",
    "ToolLoopResult",
    # Agent
    "Agent",
    "AgentOptions",
    "ExpectedFile",
    "AgentResponse",
    "SessionType",
    # Agent utilities (module-level helpers, formerly AgentInitializer/AgentFinalizer)
    "setup_workspace",
    "copy_dependencies",
    "gen_dependency_prompt",
    "ensure_servers",
    "build_options",
    "start_task",
    "end_task",
    "end_task_success",
    "end_task_failure",
    "end_task_timeout",
    "end_task_error",
    "check_oversized_files",
    "get_oversized_files_prompt",
    "read_metadata",
    "generate_requirements",
    "chain_validators",
    "make_file_size_validator",
    # Remote execution
    "RunPodClient",
    "WorkerPod",
    "OrchestratorClient",
    "ComputeProfile",
    "PodInfo",
    "PodFailureInfo",
    "PodExecutionError",
]
