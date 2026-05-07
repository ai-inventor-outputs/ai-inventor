#!/usr/bin/env python3
"""
Core logic for extracting triples from a single research paper.

Uses the gen_kg workflow from aii_lib for:
1. Initial prompt to extract triples
2. Wikipedia URL verification
3. Retry loop with conversation continuity for failed URLs
"""

import json
from pathlib import Path
from typing import Any

from aii_lib import GenKGConfig, generate_kg_triples

# Import prompt, schema, and system prompt from standard location
from aii_pipeline.prompts.steps._1_seed_hypo._invention_kg import (
    Triples,
    build_retry_prompt,
    get_system_prompt,
    triples_prompt,
)


async def get_triples_for_paper(
    paper_id: int,
    paper_index: int,
    title: str,
    abstract: str,
    parent_run_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Extract knowledge graph triples from a single paper.

    Uses gen_kg workflow which:
    1. Runs agent to extract triples
    2. Verifies Wikipedia URLs exist
    3. Retries with conversation continuity if URLs are invalid

    Args:
        paper_id: Paper ID within the year
        paper_index: Global paper index across all years
        title: Paper title
        abstract: Paper abstract
        parent_run_dir: Parent run directory where this paper's folder will be created
        config: Agent configuration from config.yaml (get_triples section with claude_agent nested)

    Returns:
        Dict with analysis results or None if failed:
        {
            "paper_id": int,
            "paper_index": int,
            "title": str,
            "cost": float,
            "run_dir": str,
            "analysis": dict
        }
    """
    # Create run directory: parent_run_dir/paper_{index:05d}/
    run_name = f"paper_{paper_index:05d}"
    run_dir = parent_run_dir / run_name
    agent_cwd_dir = run_dir / "agent_cwd"

    # Setup workspace
    run_dir.mkdir(parents=True, exist_ok=True)
    agent_cwd_dir.mkdir(parents=True, exist_ok=True)

    # Create prompt
    prompt = triples_prompt(title, abstract)

    # Extract claude_agent config (nested under get_triples in config.yaml)
    claude_cfg = config.get("claude_agent", {})

    # Build workflow config
    # Note: gen_kg workflow uses aii_web_tools__search and aii_web_tools__fetch MCP tools automatically
    # TODO: thread ``parent_module_id`` through ``get_triples_for_paper`` /
    # ``_4_get_triples.main`` so ``GenKGConfig`` can wire its task to the
    # owning module. Pre-existing gap exposed by the v26-object-first
    # parent_module_id requirement on ``GenKGConfig``.
    kg_config = GenKGConfig(  # ty: ignore[missing-argument]
        paper_id=paper_id,
        paper_index=paper_index,
        title=title,
        abstract=abstract,
        prompt=prompt,
        system_prompt=get_system_prompt(),  # From prompts module
        model=claude_cfg.get("model", "claude-sonnet-4-6"),
        max_turns=claude_cfg.get("max_turns"),
        agent_timeout=claude_cfg.get("agent_timeout"),
        agent_retries=claude_cfg.get("agent_retries", 3),
        seq_prompt_timeout=claude_cfg.get("seq_prompt_timeout"),
        seq_prompt_retries=claude_cfg.get("seq_prompt_retries", 3),
        cwd=str(agent_cwd_dir),
        response_schema=Triples,
        verify_retries=config.get("url_verification_retries", 2),
        min_valid_urls=config.get("min_valid_urls", 0),
        build_retry_prompt_fn=build_retry_prompt,
        disallowed_tools=claude_cfg.get("disallowed_tools"),
        allowed_tools=claude_cfg.get("allowed_tools"),
    )

    # Run workflow
    result = await generate_kg_triples(kg_config)

    # Check result
    if result.error:
        return None

    if result.triples is None:
        return None

    # Write structured output to disk for downstream validation
    # (Previously written by StructJsonOutConfig, now written by caller)
    output_data = {"paper_type": result.paper_type, "triples": result.triples}
    (agent_cwd_dir / "triples_output.json").write_text(
        json.dumps(output_data, indent=2), encoding="utf-8"
    )

    # Build analysis dict from result
    analysis = {
        "paper_type": result.paper_type,
        "triples": result.triples,
    }

    return {
        "paper_id": paper_id,
        "paper_index": paper_index,
        "title": title,
        "run_dir": str(run_dir),
        "analysis": analysis,
        "verified": result.verified,
    }
