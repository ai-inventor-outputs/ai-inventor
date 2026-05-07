"""Seed sampling utilities — BM25 topic matching, pool building, agent assignment."""

import json
import random
from pathlib import Path

from aii_lib.run import emit


def load_hypothesis_prompts_from_file(
    file_path: str,
) -> tuple[list[dict], str]:
    """Load hypothesis prompts from a JSON file."""
    prompts_file = Path(file_path)
    if not prompts_file.is_absolute():
        prompts_file = Path(__file__).parent.parent.parent.parent / file_path

    if not prompts_file.exists():
        emit.status_public_warning(f"Hypothesis prompts file not found: {prompts_file}")
        return [], ""

    with open(prompts_file) as f:
        prompts = json.load(f)

    emit.status_public_info(f"Loaded {len(prompts)} prompts from: {prompts_file}")
    return prompts, str(prompts_file)


def match_topics_bm25(aii_prompt: str, available_topics: list[str], top_k: int = 4) -> list[str]:
    """Match research area to most similar topics using BM25."""
    import bm25s

    if not available_topics:
        return []

    corpus_tokens = bm25s.tokenize(available_topics, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    query_tokens = bm25s.tokenize([aii_prompt], stopwords="en")
    results, _ = retriever.retrieve(
        query_tokens, corpus=available_topics, k=min(top_k, len(available_topics))
    )
    return results[0].tolist()


def build_sampling_pools(
    prompts: list[dict], selected_topics: list[str], pool_size: int = 10
) -> dict[str, list[dict]]:
    """Build sampling pools for selected topics (top-k by score_percentile)."""
    pools = {}
    for sel_topic in selected_topics:
        sel_blind = [p for p in prompts if p.get("blind_topic") == sel_topic]
        sel_blind.sort(key=lambda x: x.get("score_percentile", 0), reverse=True)
        pools[sel_topic] = sel_blind[:pool_size]
    return pools


def assign_topics_to_agents(
    selected_topics: list[str], num_agents: int, topics_per_agent: int
) -> list[list[str]]:
    """Assign topics to agents via round-robin distribution."""
    if not selected_topics:
        return [[] for _ in range(num_agents)]

    shuffled_topics = selected_topics.copy()
    random.shuffle(shuffled_topics)

    agent_topics = [[] for _ in range(num_agents)]
    topic_idx = 0
    for agent_idx in range(num_agents):
        for _ in range(topics_per_agent):
            if topic_idx >= len(shuffled_topics):
                topic_idx = 0
            agent_topics[agent_idx].append(shuffled_topics[topic_idx])
            topic_idx += 1
    return agent_topics


def sample_seeds_for_agents(
    pools: dict[str, list[dict]],
    agent_topics: list[list[str]],
    seeds_per_topic: int = 2,
) -> list[list[dict]]:
    """Sample seeds for each agent from their assigned topics' pools."""
    agent_seeds = []
    for topics in agent_topics:
        sampled = []
        for topic in topics:
            pool = pools.get(topic, [])
            if pool:
                n_select = min(seeds_per_topic, len(pool))
                selected = random.sample(pool, n_select)
                for s in selected:
                    if s not in sampled:
                        sampled.append(s)
        agent_seeds.append(sampled)
    return agent_seeds
