"""
Workflows - Multi-step orchestrations using base operations.

Workflows combine base operations (chat, chat_tools, agent) with
telemetry to implement higher-level patterns.
"""

from .gen_kg import (
    GenKGConfig,
    GenKGResult,
    generate_kg_triples,
    verify_wikipedia_urls,
)
from .research_workflow import (
    RESEARCH_TOOLS,
    ResearchWorkflowConfig,
    ResearchWorkflowResult,
    research_workflow,
)

__all__ = [
    # Research Workflow (agnostic)
    "research_workflow",
    "ResearchWorkflowConfig",
    "ResearchWorkflowResult",
    "RESEARCH_TOOLS",
    # Gen KG - Knowledge Graph Generation
    "GenKGConfig",
    "GenKGResult",
    "generate_kg_triples",
    "verify_wikipedia_urls",
]
