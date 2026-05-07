#!/usr/bin/env python3
"""
_1_seed_hypo.py - Hypothesis Seed Generation Module.

Two-step process: gen_seeds → sample_seeds

gen_seeds:
  If invention_kg.resume_file is set: loads seeds from that file
  Otherwise: runs the invention_kg pipeline (first_step → last_step)

  The invention_kg pipeline steps:
  1. sel_topics     - Select topics from OpenAlex
  2. get_papers     - Fetch papers for topics
  3. clean_papers   - Extract minimal paper data
  4. get_triples    - Extract triples using agents
  5. add_wikidata   - Enrich triples with Wikidata
  6. link_to_papers - Combine papers + triples
  7. gen_hypo_seeds - Generate blind spots/breakthroughs
  8. gen_hypo_prompt - Format seeds as prompts
  9. gen_graphs     - Generate co-occurrence/ontology graphs

sample_seeds:
  1. Select topics (BM25 match to aii_prompt or manual list)
  2. Build sampling pool per topic (top N by score_percentile)
  3. Assign topics to agents (round-robin)
  4. Sample prompts for each agent from their topics' pools
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from aii_lib.run import emit
from aii_lib.run.context import ctx_scope, current_ctx
from aii_lib.run.mdgroup import SeqMdGroup

from aii_pipeline.prompts.steps._1_seed_hypo.out_schema import SeedHypoOut
from aii_pipeline.steps._1_seed_hypo.sampling import (
    assign_topics_to_agents,
    build_sampling_pools,
    load_hypothesis_prompts_from_file,
    match_topics_bm25,
    sample_seeds_for_agents,
)
from aii_pipeline.utils import PipelineConfig, rel_path

if TYPE_CHECKING:
    from aii_pipeline.steps.base import StepContext


@dataclass
class SeedHypoCtx:
    """Phase context for seed_hypo — the first phase.

    The only upstream input is the run-level config + run dir.
    Listed as NEW in REFACTOR_PLAN §3.2. seed_hypo has no
    phase-specific upstream inputs (no prior phase) and no extra
    control state, so this is a thin narrowing of ``StepContext`` —
    kept for symmetry with the other phase ctxs and so any future
    seed_hypo-specific configuration has a typed home.
    """

    config: PipelineConfig
    run_dir: Path


class SeedHypoGroup(SeqMdGroup):
    """Phase 1 — seed_hypo.

    Wraps the seed_hypo phase's sample_seeds + invention_kg sub-pipeline.
    Substep accessors aren't defined here because seed_hypo's tree
    structure depends on dynamic config branches that aren't
    pre-scaffolded.
    """

    kind: Literal["seed_hypo_group"] = "seed_hypo_group"
    """Per-subclass discriminator (see ``HypoLoopGroup.kind``)."""

    def get_context(self) -> "SeedHypoCtx":
        parent: StepContext = current_ctx()
        return SeedHypoCtx(config=parent.config, run_dir=parent.run_dir)

    async def execute(self) -> Any:
        with ctx_scope(self.get_context()) as ctx:
            return await run_seed_hypo_module(
                ctx.config,
                run_dir=ctx.run_dir,
            )


# Valid step names for invention_kg pipeline (order matches execution)
KG_STEPS = [
    "sel_topics",
    "get_papers",
    "clean_papers",
    "get_triples",
    "add_wikidata",
    "link_to_papers",
    "gen_hypo_seeds",
    "gen_hypo_prompt",
    "gen_graphs",
]


def _kg_step_name_to_number(step_name: str) -> int:
    """Convert kg step name to step number (1-indexed)."""
    try:
        return KG_STEPS.index(step_name) + 1
    except ValueError as e:
        raise ValueError(f"Invalid KG step name '{step_name}', valid steps: {KG_STEPS}") from e


async def run_invention_kg_pipeline(
    config: PipelineConfig,
    output_dir: Path,
    run_dir: Path,
) -> tuple[list[dict], str, Path]:
    """
    Run the invention_kg sub-pipeline (9 steps) to generate hypothesis seeds.

    The 9 kg steps are run inline here as a simple loop; each step is called
    directly with the typed PipelineConfig + run_id + base_dir. There is no
    longer a separate invention_kg.Config singleton or kg-level pipeline.py.

    Args:
        config: Pipeline configuration (typed)
        output_dir: Directory to store top-level seed_hypo outputs (run_dir / "1_seed_hypo")
        run_dir: Main pipeline run directory. kg outputs land at
            run_dir / "1_seed_hypo" / "_N_step/" — derived via base_dir = run_dir.parent
            and run_id = run_dir.name so the kg sub-pipeline nests under the main run.

    Returns:
        Tuple of (prompts list, source description, kg_data_dir)
    """
    import inspect

    from aii_pipeline.steps._1_seed_hypo.invention_kg.constants import (
        STEP_1_SEL_TOPICS,
        STEP_2_PAPERS,
        STEP_3_PAPERS_CLEAN,
        STEP_8_SEED_PROMPT,
    )
    from aii_pipeline.steps._1_seed_hypo.invention_kg.steps._1_sel_topics import (
        run_sel_topics,
    )
    from aii_pipeline.steps._1_seed_hypo.invention_kg.steps._2_get_papers import (
        run_get_papers,
    )
    from aii_pipeline.steps._1_seed_hypo.invention_kg.steps._3_clean_papers import (
        run_clean_papers,
    )
    from aii_pipeline.steps._1_seed_hypo.invention_kg.steps._4_get_triples import (
        main as get_triples_main,
    )
    from aii_pipeline.steps._1_seed_hypo.invention_kg.steps._5_add_wikidata import (
        main as add_wikidata_main,
    )
    from aii_pipeline.steps._1_seed_hypo.invention_kg.steps._6_link_to_papers import (
        main as link_main,
    )
    from aii_pipeline.steps._1_seed_hypo.invention_kg.steps._7_gen_hypo_seeds import (
        main as hypo_seeds_main,
    )
    from aii_pipeline.steps._1_seed_hypo.invention_kg.steps._8_gen_seed_prompt import (
        main as seed_prompt_main,
    )
    from aii_pipeline.steps._1_seed_hypo.invention_kg.steps._9_gen_graphs import (
        main as gen_graphs_main,
    )
    from aii_pipeline.steps._1_seed_hypo.invention_kg.utils import get_run_dir

    kg_cfg = config.seed_hypo.invention_kg

    # Convert step names to numbers
    start_step_num = _kg_step_name_to_number(kg_cfg.first_step)
    end_step_num = _kg_step_name_to_number(kg_cfg.last_step)

    # base_dir + run_id route kg outputs to <run_dir>/1_seed_hypo/_N_step/.
    base_dir = run_dir.parent
    run_id = run_dir.name

    # Resume validation: when starting past step 1, the user must have set the
    # corresponding *_out_dir on the typed config and that dir must exist
    # under the same run scope (so resume picks up files at the expected path).
    step_to_prev_out_dir = {
        2: ("sel_topics_out_dir", kg_cfg.sel_topics_out_dir),
        3: ("get_papers_out_dir", kg_cfg.get_papers_out_dir),
        4: ("clean_papers_out_dir", kg_cfg.clean_papers_out_dir),
        5: ("get_triples_out_dir", kg_cfg.get_triples_out_dir),
        6: ("add_wikidata_out_dir", kg_cfg.add_wikidata_out_dir),
        7: ("link_to_papers_out_dir", kg_cfg.link_to_papers_out_dir),
        8: ("gen_hypo_seeds_out_dir", kg_cfg.gen_hypo_seeds_out_dir),
        9: ("gen_hypo_prompt_out_dir", kg_cfg.gen_hypo_prompt_out_dir),
    }

    if start_step_num > 1:
        out_dir_key, out_dir_value = step_to_prev_out_dir.get(start_step_num, (None, ""))
        out_dir_path = (out_dir_value or "").strip()
        if not out_dir_path:
            emit.status_public_error(
                f"Starting from step {start_step_num} requires {out_dir_key} to be set"
            )
            return [], "", output_dir
        out_dir = Path(out_dir_path)
        if not out_dir.is_absolute():
            emit.status_public_error(f"{out_dir_key} must be an absolute path: {out_dir_path}")
            return [], "", output_dir
        if not out_dir.exists():
            emit.status_public_error(f"{out_dir_key} does not exist: {out_dir}")
            return [], "", output_dir
        # Re-derive run_id from path to ensure resume reads from the right run.
        # Path structure: <base>/<run_id>/1_seed_hypo/_N_step
        parts = out_dir.parts
        try:
            seed_hypo_idx = parts.index("1_seed_hypo")
            run_id = parts[seed_hypo_idx - 1]
            base_dir = Path(*parts[: seed_hypo_idx - 1])
        except (ValueError, IndexError) as e:
            emit.status_public_error(f"Could not extract run_id from path: {out_dir}")
            raise ValueError(f"Could not extract run_id from path: {out_dir}") from e
        emit.status_public_progress(f"Resuming from {out_dir_key}: {out_dir}")
        emit.status_public_info(f"Extracted run_id: {run_id}")

    emit.status_public_info("🔬 Running invention_kg pipeline...")
    emit.status_private_info(
        f"Steps: {kg_cfg.first_step} ({start_step_num}) → {kg_cfg.last_step} ({end_step_num})"
    )
    emit.status_public_info(f"Using run_id: {run_id}, base_dir: {base_dir}")

    # Build a plain-dict view of get_triples (the agent wrapper still expects
    # dict access for nested claude_agent fields). Same for blind_spots /
    # temporal_windows below — kept narrowly scoped per-step.
    get_triples_dict = {
        "get_triples": {
            "max_papers": kg_cfg.get_triples.max_papers,
            "max_concurrent_agents": kg_cfg.get_triples.max_concurrent_agents,
            "stagger_delay": kg_cfg.get_triples.stagger_delay,
            "url_verification_retries": kg_cfg.get_triples.url_verification_retries,
            "min_valid_urls": kg_cfg.get_triples.min_valid_urls,
            "claude_agent": {
                "model": kg_cfg.get_triples.claude_agent.model,
                "max_turns": kg_cfg.get_triples.claude_agent.max_turns,
                "agent_timeout": kg_cfg.get_triples.claude_agent.agent_timeout,
                "agent_retries": kg_cfg.get_triples.claude_agent.agent_retries,
                "seq_prompt_timeout": kg_cfg.get_triples.claude_agent.seq_prompt_timeout,
                "seq_prompt_retries": kg_cfg.get_triples.claude_agent.seq_prompt_retries,
                "disallowed_tools": kg_cfg.get_triples.claude_agent.disallowed_tools,
                "allowed_tools": kg_cfg.get_triples.claude_agent.allowed_tools,
            },
        }
    }
    blind_spots_dict = {
        "min_shared_concepts": kg_cfg.gen_hypo_seeds.blind_spots.min_shared_concepts,
        "max_similarity": kg_cfg.gen_hypo_seeds.blind_spots.max_similarity,
        "entity_types": kg_cfg.gen_hypo_seeds.blind_spots.entity_types,
    }

    # Per-step thunks. Each returns True on success / non-False on success.
    # Step 1 wraps run_sel_topics (which signature differs); steps 2/3 wrap
    # run_get_papers/run_clean_papers; steps 4-9 are direct main() calls.
    def step_1():
        out = get_run_dir(STEP_1_SEL_TOPICS, run_id, base_dir)
        run_sel_topics(
            topic_names=kg_cfg.sel_topics.topics,
            output_dir=out,
            email=kg_cfg.get_papers.email,
        )
        return True

    def step_2():
        topics_file = get_run_dir(STEP_1_SEL_TOPICS, run_id, base_dir) / "topics.json"
        if not topics_file.exists():
            emit.status_public_error(f"Topics file not found: {topics_file}")
            return False
        out = get_run_dir(STEP_2_PAPERS, run_id, base_dir)
        run_get_papers(
            topics_file=topics_file,
            output_dir=out,
            start_year=kg_cfg.get_papers.year_range["start"],
            end_year=kg_cfg.get_papers.year_range["end"],
            papers_per_topic_per_year=kg_cfg.get_papers.papers_per_year,
            sort_by=kg_cfg.get_papers.sort_by,
            email=kg_cfg.get_papers.email,
        )
        return True

    def step_3():
        in_dir = get_run_dir(STEP_2_PAPERS, run_id, base_dir)
        out = get_run_dir(STEP_3_PAPERS_CLEAN, run_id, base_dir)
        run_clean_papers(papers_dir=in_dir, output_dir=out)
        return True

    async def step_4():
        exit_code = await get_triples_main(run_id, base_dir, get_triples_dict)
        return exit_code == 0

    async def step_5():
        exit_code = await add_wikidata_main(run_id, base_dir)
        return exit_code == 0

    def step_6():
        exit_code = link_main(run_id, base_dir)
        return exit_code == 0

    def step_7():
        exit_code = hypo_seeds_main(run_id, base_dir, blind_spots_dict)
        return exit_code == 0

    def step_8():
        exit_code = seed_prompt_main(run_id, base_dir)
        return exit_code == 0

    def step_9():
        exit_code = gen_graphs_main(run_id, base_dir, kg_cfg.gen_graph.temporal_windows)
        return exit_code == 0

    steps = {
        1: ("Select Topics", step_1),
        2: ("Get Papers", step_2),
        3: ("Clean Papers", step_3),
        4: ("Get Triples", step_4),
        5: ("Add Wikidata", step_5),
        6: ("Link to Papers", step_6),
        7: ("Gen Hypo Seeds", step_7),
        8: ("Gen Seed Prompt", step_8),
        9: ("Generate Graphs", step_9),
    }

    # The KG sub-pipeline runs as telemetry-only steps — they don't
    # appear in the v26 structured tree (they're not part of the formal
    # gen_hypo module set).

    success = True
    try:
        for step_num in range(start_step_num, end_step_num + 1):
            if step_num not in steps:
                emit.status_public_error(f"Invalid step number: {step_num}")
                continue
            step_label, step_fn = steps[step_num]
            emit.status_public_progress(f"Step {step_num}: {step_label} (run_id: {run_id})")
            try:
                if inspect.iscoroutinefunction(step_fn):
                    ok = await step_fn()
                else:
                    ok = step_fn()
                if ok is False:
                    emit.status_public_error(f"Pipeline stopped at step {step_num} due to error")
                    success = False
                    break
                emit.status_public_success(f"Step {step_num} ({step_label}) completed")
            except Exception as e:
                emit.status_public_error(f"Error in step {step_num} ({step_label}): {e}")
                raise

        if not success:
            return [], "", output_dir

        emit.status_public_success("INVENTION_KG pipeline completed successfully!")
    finally:
        # KG steps are telemetry-only; no v26 group to close here.
        pass

    # Load prompts from step 8 output (regardless of how far the loop went,
    # as long as step 8 was in range).
    prompts: list[dict] = []
    kg_data_dir = get_run_dir(STEP_8_SEED_PROMPT, run_id, base_dir)
    prompts_file = kg_data_dir / "blind_spot_prompts.json"
    if prompts_file.exists():
        with open(prompts_file, encoding="utf-8") as f:
            prompts = json.load(f)
        emit.status_public_info(f"Loaded {len(prompts)} prompts from {prompts_file}")

    # Copy prompts file to seed_hypo output directory
    output_prompts = output_dir / "hypo_seed_prompts.json"
    with open(output_prompts, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    source = "invention_kg pipeline"
    emit.status_public_success(f"✅ Generated {len(prompts)} prompts from {source}")

    return prompts, source, kg_data_dir


async def run_seed_hypo_module(
    config: PipelineConfig,
    run_dir=None,
):
    """
    Run the hypothesis seed generation module.

    Steps: gen_seeds → sample_seeds
    - gen_seeds: If resume_file is set, load from file; otherwise run invention_kg pipeline
    - sample_seeds: Sample seeds for agents based on AII prompt

    Args:
        config: Typed pipeline configuration
        run_dir: Optional run directory for outputs

    Returns:
        Dictionary with agent_prompts and metadata
    """
    # Create output directory
    if run_dir:
        output_dir = run_dir / "1_seed_hypo"
    else:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"{config.outputs_directory}/{timestamp}_seed_hypo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get config
    aii_prompt = config.prompt
    seed_hypo_cfg = config.seed_hypo
    kg_cfg = seed_hypo_cfg.invention_kg
    sampling_cfg = seed_hypo_cfg.sampling

    # Determine which steps to run
    first_step = seed_hypo_cfg.first_step
    last_step = seed_hypo_cfg.last_step
    run_gen_seeds = first_step == "gen_seeds"
    run_sample_seeds = last_step == "sample_seeds"

    emit.status_private_info(f"Research Area: {aii_prompt}")
    emit.status_private_info(f"Steps: {first_step} → {last_step}")
    emit.status_private_info(f"Output: {rel_path(output_dir)}")

    # Start the seed_hypo seq group (wraps the seed_hypo + sample_seeds
    # substeps). Capture the auto-generated group id; pipeline ceremony
    # references this single var instead of re-typing canonical strings
    # everywhere.
    gen_hypo_gid = emit.start_seq_group(name="seed_hypo")

    all_prompts = []

    # Step 1: gen_seeds (telemetry-only; no v26 group ceremony)
    if run_gen_seeds:
        # Check if we should load from existing output dir
        if kg_cfg.gen_hypo_prompt_out_dir:
            # Load from gen_hypo_prompt output directory
            emit.status_public_info("\n💡 Loading seeds from gen_hypo_prompt_out_dir...")
            prompts_file = Path(kg_cfg.gen_hypo_prompt_out_dir)
            if not prompts_file.is_absolute():
                # Navigate to project root: steps/ -> aii_pipeline/ -> src/ -> aii_pipeline/ -> project_root
                prompts_file = Path(__file__).parent.parent.parent.parent.parent / prompts_file
            # Look for prompts file (hypo_seed_prompts.json, blind_spot_prompts.json, etc.)
            prompts_files = [
                "hypo_seed_prompts.json",
                "blind_spot_prompts.json",
                "_1_hypo_seed_prompts.json",
            ]
            found_file = None
            for fname in prompts_files:
                if (prompts_file / fname).exists():
                    found_file = prompts_file / fname
                    break

            if found_file:
                all_prompts, _hypo_source = load_hypothesis_prompts_from_file(
                    file_path=str(found_file),
                )
            else:
                emit.status_public_error(f"❌ No prompts file found in {prompts_file}")
                return None
        else:
            # Run invention_kg pipeline
            emit.status_public_info("\n🔬 Running invention_kg pipeline...")
            all_prompts, _hypo_source, _kg_data_dir = await run_invention_kg_pipeline(
                config=config,
                output_dir=output_dir,
                run_dir=run_dir if run_dir else output_dir.parent,
            )

    else:
        # Skip gen_seeds - load from invention_kg_seed_out_dir
        seed_out_dir = seed_hypo_cfg.invention_kg_seed_out_dir
        if seed_out_dir:
            emit.status_public_info("\n💡 Loading seeds from invention_kg_seed_out_dir...")
            prompts_file = Path(seed_out_dir)
            if not prompts_file.is_absolute():
                # Navigate to project root: steps/ -> aii_pipeline/ -> src/ -> aii_pipeline/ -> project_root
                prompts_file = Path(__file__).parent.parent.parent.parent.parent / prompts_file
            if (prompts_file / "hypo_seed_prompts.json").exists():
                all_prompts, _hypo_source = load_hypothesis_prompts_from_file(
                    file_path=str(prompts_file / "hypo_seed_prompts.json"),
                )
            else:
                emit.status_public_error(f"❌ No prompts file found in {prompts_file}")
                return None
        else:
            emit.status_public_warning("⚠️  Skipping gen_seeds but no invention_kg_seed_out_dir set")

    # Handle case with no prompts
    if not all_prompts:
        emit.status_public_warning("⚠️  No hypothesis prompts available - continuing without")
        if config.gen_hypo.use_claude_agent:
            num_agents = config.gen_hypo.seeded_hypos_per_llm
        else:
            num_agents = config.gen_hypo.seeded_hypos_per_llm * len(
                config.gen_hypo.llm_client.models
            )
        module_output = SeedHypoOut(
            output_dir=str(output_dir),
            agent_prompts=[[] for _ in range(num_agents)],
            agent_topics=[[] for _ in range(num_agents)],
        )
        emit.end_group(gen_hypo_gid)
        return module_output

    # Copy prompts file to output directory if not already there
    output_prompts = output_dir / "hypo_seed_prompts.json"
    if not output_prompts.exists():
        with open(output_prompts, "w", encoding="utf-8") as f:
            json.dump(all_prompts, f, indent=2, ensure_ascii=False)

    # Step 2: sample_seeds
    if run_sample_seeds:
        sample_seeds_mid = emit.start_single_module(
            name="sample_seeds",
            parent_id=gen_hypo_gid,
        )
        sample_seeds_task_id = emit.start_task(
            name="sample_seeds",
            parent_module_id=sample_seeds_mid,
        )

        # Extract available topics from prompts
        available_topics = set()
        for p in all_prompts:
            available_topics.update(p.get("topics", []))
        available_topics = sorted(available_topics)
        emit.status_public_info(
            f"\n   Available topics ({len(available_topics)}): {available_topics}"
        )

        # Select topics (BM25 or manual)
        emit.status_public_info("\n📋 Selecting topics for sampling...")
        sel_topics_cfg = sampling_cfg.sel_topics

        if sel_topics_cfg == "auto":
            # BM25 match
            selected_topics = match_topics_bm25(
                aii_prompt,
                available_topics,
                top_k=sampling_cfg.aii_prompt_topic_match_k,
            )
            emit.status_public_info(
                f"BM25 matched {len(selected_topics)} topics: {selected_topics}"
            )
        else:
            # Manual list
            selected_topics = [t for t in sel_topics_cfg if t in available_topics]
            emit.status_public_info(f"Manual selection: {selected_topics}")

        if not selected_topics:
            emit.status_public_warning("   No topics selected - using all available")
            selected_topics = list(available_topics)

        # Build sampling pools
        emit.status_public_info("\n🎯 Building sampling pools...")
        pools = build_sampling_pools(
            all_prompts, selected_topics, pool_size=sampling_cfg.seed_sampling_pool
        )
        for topic, pool in pools.items():
            emit.status_public_info(f"{topic}: {len(pool)} seeds in pool")

        # Assign topics to agents
        emit.status_public_info("\n👥 Assigning topics to agents...")
        if config.gen_hypo.use_claude_agent:
            num_agents = config.gen_hypo.seeded_hypos_per_llm
        else:
            num_agents = config.gen_hypo.seeded_hypos_per_llm * len(
                config.gen_hypo.llm_client.models
            )
        agent_topics = assign_topics_to_agents(
            selected_topics, num_agents, sampling_cfg.topics_per_agent
        )
        for i, topics in enumerate(agent_topics):
            emit.status_public_info(f"Agent {i + 1}: {topics}")

        # Sample seeds for each agent
        emit.status_public_info("\n🎲 Sampling seeds for agents...")
        agent_prompts = sample_seeds_for_agents(pools, agent_topics, sampling_cfg.seeds_per_topic)

        total_unique = len({p["id"] for prompts in agent_prompts for p in prompts})
        emit.status_public_success(f"\n✅ Sampled seeds for {num_agents} agents")
        emit.status_private_info(
            f"Config: {sampling_cfg.topics_per_agent} topics/agent × {sampling_cfg.seeds_per_topic} seeds/topic"
        )
        emit.status_public_info(f"Total unique seeds: {total_unique}")

        for i, seeds in enumerate(agent_prompts):
            seed_ids = [p.get("id", "?")[:50] for p in seeds]
            emit.status_public_info(f"Agent {i + 1} ({len(seeds)} seeds): {seed_ids}")

        emit.end_task(sample_seeds_task_id, status="done", name="sample_seeds")
        emit.end_module(parent_id=gen_hypo_gid, module_id=sample_seeds_mid)
    else:
        # Skip sample_seeds - just return the prompts without sampling
        emit.status_public_info("\n⏭️  Skipping sample_seeds step")
        selected_topics = []
        pools = {}
        agent_topics = []
        agent_prompts = []
        num_agents = 0

    # Build module output
    module_output = SeedHypoOut(
        output_dir=str(output_dir),
        agent_prompts=agent_prompts,
        agent_topics=agent_topics,
        selected_topics=selected_topics,
        pools={t: [p["id"] for p in pool] for t, pool in pools.items()},
        all_hypo_prompts=all_prompts,
    )

    emit.end_group(gen_hypo_gid)

    return module_output
