#!/usr/bin/env python3
"""
Topic Blind Spots Opportunity Finder (Concept-Centric).

Finds concepts that a "blind topic" is missing by comparing it to a
semantically dissimilar "reference topic" that shares some concepts.

Each blind spot concept is its own entry with rich metrics for ranking.

Logic:
1. Find topic pairs that are semantically dissimilar but share some concepts
2. For each pair, identify concepts the ref_topic uses that blind_topic doesn't
3. Each concept becomes an individual "blind spot" opportunity with metrics
"""

import json
import re
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aii_lib.run import emit

from ._scoring import (
    HAS_SENTENCE_TRANSFORMERS,
    batch_encode_concepts,
    calculate_topic_similarity,
    compute_bridge_potential,
    compute_citation_weight,
    compute_concept_novelty_score,
    compute_concept_ref_importance_score,
    compute_concept_transferability_score,
    compute_idf,
    compute_recency_score,
    compute_seed_score,
    compute_semantic_distance_to_topic,
    compute_tf_idf,
    compute_topic_centroid_distance,
    compute_topic_pair_score,
    extract_topic_data,
    get_semantic_model,
    zscore_sigmoid_normalize,
)

if TYPE_CHECKING:
    import numpy as np


def slugify(text: str) -> str:
    """Convert text to URL/ID-friendly slug."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s-]+", "_", text)
    return text.strip("_")


def find_blind_spots(
    topic_data: dict[str, dict[str, Any]],
    min_shared_concepts: int = 1,
    max_similarity: float = 1.0,
    min_blind_spot_count: int = 1,
) -> list[dict[str, Any]]:
    """
    Find concept-level blind spots with rich metrics.

    Args:
        topic_data: Topic data from extract_topic_data
        min_shared_concepts: Minimum shared concepts to consider pair (default 1)
        max_similarity: Maximum similarity (default 1.0 = no filter)
        min_blind_spot_count: Minimum times concept used in ref_topic (default 1)

    Returns:
        List of individual blind spot concepts with metrics
    """
    topics = list(topic_data.keys())
    blind_spots = []

    # Pre-compute: all topic concepts for IDF calculation
    all_topic_concepts = {topic: set(data["concepts"].keys()) for topic, data in topic_data.items()}

    # Load semantic model (required)
    embeddings_cache: dict[str, np.ndarray] = {}
    if not HAS_SENTENCE_TRANSFORMERS:
        raise ImportError(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        )

    semantic_model = get_semantic_model()
    if semantic_model is None:
        raise RuntimeError("Failed to load semantic model")

    # Batch encode topic names for topic pair distance
    emit.status_public_info(f"Encoding {len(topics)} topic names...")
    batch_encode_concepts(topics, embeddings_cache, semantic_model)

    # Batch encode all concepts for novelty score
    all_concepts = set()
    for concepts in all_topic_concepts.values():
        all_concepts.update(concepts)
    emit.status_public_info(f"Encoding {len(all_concepts)} concepts...")
    batch_encode_concepts(list(all_concepts), embeddings_cache, semantic_model)

    # Pre-compute: global year range for recency
    all_years = []
    for data in topic_data.values():
        for concept_info in data["concepts"].values():
            for paper in concept_info["papers"]:
                if paper.get("year", 0) > 0:
                    all_years.append(paper["year"])

    global_min_year = min(all_years) if all_years else 2000
    global_max_year = max(all_years) if all_years else 2024

    total_pairs = len(topics) * (len(topics) - 1) // 2
    pair_count = 0
    emit.status_public_info(f"Processing {total_pairs} topic pairs...")

    for i, blind_topic in enumerate(topics):
        blind_concepts = set(topic_data[blind_topic]["concepts"].keys())

        for ref_topic in topics[i + 1 :]:
            ref_concepts = set(topic_data[ref_topic]["concepts"].keys())
            pair_count += 1

            similarity, shared = calculate_topic_similarity(blind_concepts, ref_concepts)

            # Skip if not enough shared concepts
            if len(shared) < min_shared_concepts:
                continue

            # Skip if too similar
            if similarity > max_similarity:
                continue

            # Topic-level metrics
            # Use concept centroid distance if available, else fall back to Jaccard
            if semantic_model is not None:
                blind_ref_topic_sem_dist = compute_topic_centroid_distance(
                    blind_concepts, ref_concepts, embeddings_cache, semantic_model
                )
            else:
                blind_ref_topic_sem_dist = round(1 - similarity, 4)
            min_size = min(len(blind_concepts), len(ref_concepts))
            topic_pair_shared_ratio = round(len(shared) / min_size, 4) if min_size > 0 else 0

            # Process blind spots in both directions
            for direction_blind, direction_ref, direction_ref_concepts in [
                (blind_topic, ref_topic, ref_concepts - blind_concepts),
                (ref_topic, blind_topic, blind_concepts - ref_concepts),
            ]:
                ref_data = topic_data[direction_ref]

                # Compute total concepts in ref_topic for TF calculation
                total_concepts_in_ref_topic = sum(c["count"] for c in ref_data["concepts"].values())

                for concept in direction_ref_concepts:
                    concept_info = ref_data["concepts"][concept]

                    if concept_info["count"] < min_blind_spot_count:
                        continue

                    count = concept_info["count"]

                    # Compute IDF and TF-IDF
                    idf = compute_idf(concept, all_topic_concepts)
                    tf_idf = compute_tf_idf(count, total_concepts_in_ref_topic, idf)

                    # Compute citation weight
                    citation_weight = compute_citation_weight(concept_info["papers"])

                    # Compute recency
                    avg_year, recency_score = compute_recency_score(
                        concept_info["papers"], global_min_year, global_max_year
                    )

                    # Compute bridge potential
                    bridge_potential = compute_bridge_potential(
                        concept, shared, ref_data["concept_cooccurrence"]
                    )

                    # Compute semantic distance to blind topic
                    semantic_dist = None
                    if semantic_model is not None:
                        blind_topic_concepts = all_topic_concepts.get(direction_blind, set())
                        semantic_dist = compute_semantic_distance_to_topic(
                            concept,
                            blind_topic_concepts,
                            embeddings_cache,
                            semantic_model,
                        )

                    # Create blind spot entry with hierarchical structure
                    blind_spot_id = (
                        f"{slugify(concept)}__{slugify(direction_blind)}__{slugify(direction_ref)}"
                    )

                    blind_spot_entry = {
                        "id": blind_spot_id,
                        "concept": concept,
                        "entity_type": concept_info["entity_type"],
                        "blind_topic": direction_blind,
                        "ref_topic": direction_ref,
                        # Hierarchical metrics (scores added in second pass)
                        "topic_pair": {
                            "blind_ref_topic_sem_dist": blind_ref_topic_sem_dist,
                            "topic_pair_shared_ratio": topic_pair_shared_ratio,
                            "shared_concepts": list(shared),
                        },
                        "importance": {
                            "count": count,
                            "tf_idf": round(tf_idf, 6),
                            "idf": round(idf, 4),
                            "concept_citation": round(citation_weight, 1),
                            "avg_year": avg_year,
                            "paper_recency": recency_score,
                        },
                        "transferability": {
                            "blind_shared_concept_cooccur": bridge_potential,
                        },
                        "novelty": {
                            "min_concept_to_blind_dist": semantic_dist,
                        },
                        "relation_breakdown": {
                            "uses": concept_info["uses"],
                            "proposes": concept_info["proposes"],
                        },
                    }

                    blind_spots.append(blind_spot_entry)

            # Log progress every 5 pairs
            if pair_count % 5 == 0 or pair_count == total_pairs:
                emit.status_public_info(
                    f"Processed {pair_count}/{total_pairs} topic pairs ({len(blind_spots)} blind spots so far)"
                )

    # Second pass: compute hierarchical scores with normalization
    emit.status_public_info(f"Computing scores for {len(blind_spots)} blind spots...")
    if blind_spots:
        # =====================================================================
        # COLLECT VALUES FOR NORMALIZATION
        # =====================================================================

        # Per-ref_topic normalization (different topics have different patterns)
        tf_idf_by_ref_topic: dict[str, list[float]] = defaultdict(list)
        citation_weights_by_ref_topic: dict[str, list[float]] = defaultdict(list)
        for bs in blind_spots:
            tf_idf_by_ref_topic[bs["ref_topic"]].append(bs["importance"]["tf_idf"])
            citation_weights_by_ref_topic[bs["ref_topic"]].append(
                bs["importance"]["concept_citation"]
            )

        # Per-topic_pair normalization (max cooccurrence depends on shared concepts in pair)
        blind_shared_concept_cooccurs_by_pair: dict[tuple[str, str], list[float]] = defaultdict(
            list
        )
        for bs in blind_spots:
            pair_key = tuple(sorted([bs["blind_topic"], bs["ref_topic"]]))
            blind_shared_concept_cooccurs_by_pair[pair_key].append(
                bs["transferability"]["blind_shared_concept_cooccur"]
            )

        # Global normalization for remaining raw 0-1 scores
        # Even though they're already 0-1, their distributions may cluster differently
        all_paper_recency: list[float] = []
        all_topic_sem_dist: list[float] = []
        all_topic_shared_ratio: list[float] = []
        all_min_blind_concept_dist: list[float] = []

        for bs in blind_spots:
            all_paper_recency.append(bs["importance"]["paper_recency"])
            all_topic_sem_dist.append(bs["topic_pair"]["blind_ref_topic_sem_dist"])
            all_topic_shared_ratio.append(bs["topic_pair"]["topic_pair_shared_ratio"])
            if bs["novelty"]["min_concept_to_blind_dist"] is not None:
                all_min_blind_concept_dist.append(bs["novelty"]["min_concept_to_blind_dist"])

        # =====================================================================
        # COMPUTE HIERARCHICAL SCORES WITH Z-SCORE + SIGMOID NORMALIZATION
        # =====================================================================
        for bs in blind_spots:
            # --- Per-ref_topic normalization ---
            ref_topic_tf_idfs = tf_idf_by_ref_topic[bs["ref_topic"]]
            concept_ref_tfidf = zscore_sigmoid_normalize(
                bs["importance"]["tf_idf"], ref_topic_tf_idfs
            )

            ref_topic_citations = citation_weights_by_ref_topic[bs["ref_topic"]]
            concept_citation = zscore_sigmoid_normalize(
                bs["importance"]["concept_citation"], ref_topic_citations
            )

            # --- Per-topic_pair normalization ---
            pair_key = tuple(sorted([bs["blind_topic"], bs["ref_topic"]]))
            pair_blind_shared_concept_cooccurs = blind_shared_concept_cooccurs_by_pair[pair_key]
            blind_shared_concept_cooccur = zscore_sigmoid_normalize(
                bs["transferability"]["blind_shared_concept_cooccur"],
                pair_blind_shared_concept_cooccurs,
            )

            # --- Global normalization ---
            paper_recency = zscore_sigmoid_normalize(
                bs["importance"]["paper_recency"], all_paper_recency
            )
            topic_sem_dist = zscore_sigmoid_normalize(
                bs["topic_pair"]["blind_ref_topic_sem_dist"], all_topic_sem_dist
            )
            topic_shared_ratio = zscore_sigmoid_normalize(
                bs["topic_pair"]["topic_pair_shared_ratio"], all_topic_shared_ratio
            )

            # Novelty: handle None case (no embeddings available)
            raw_min_dist = bs["novelty"]["min_concept_to_blind_dist"]
            if raw_min_dist is not None and all_min_blind_concept_dist:
                min_blind_concept_dist = zscore_sigmoid_normalize(
                    raw_min_dist, all_min_blind_concept_dist
                )
            else:
                min_blind_concept_dist = 0.5  # Fallback

            # Level 1: Topic Pair Score
            topic_pair_score = compute_topic_pair_score(topic_sem_dist, topic_shared_ratio)
            bs["topic_pair"]["score"] = topic_pair_score

            # Level 2: Concept Ref Importance Score
            concept_ref_importance_score = compute_concept_ref_importance_score(
                concept_ref_tfidf, concept_citation, paper_recency
            )
            bs["importance"]["score"] = concept_ref_importance_score

            # Level 3: Concept Transferability Score
            concept_transferability_score = compute_concept_transferability_score(
                blind_shared_concept_cooccur,
            )
            bs["transferability"]["score"] = concept_transferability_score

            # Level 4: Concept Novelty Score
            concept_novelty_score = compute_concept_novelty_score(
                min_blind_concept_dist,
            )
            bs["novelty"]["score"] = concept_novelty_score

            # Final: Seed Score (combines all component scores)
            seed_score = compute_seed_score(
                topic_pair_score,
                concept_ref_importance_score,
                concept_transferability_score,
                concept_novelty_score,
            )
            bs["seed_score"] = seed_score

        # Sort by seed_score descending
        blind_spots.sort(key=lambda x: x["seed_score"], reverse=True)

        # Add percentiles for each score type using binary search (O(n log n) each)
        total = len(blind_spots)

        # Compute percentile for each score dimension
        score_types = [
            ("topic_pair", "score", "percentile"),
            ("importance", "score", "percentile"),
            ("transferability", "score", "percentile"),
            ("novelty", "score", "percentile"),
        ]

        for section, score_key, percentile_key in score_types:
            all_scores_asc = sorted(bs[section][score_key] for bs in blind_spots)
            for bs in blind_spots:
                below = bisect_left(all_scores_asc, bs[section][score_key])
                bs[section][percentile_key] = round(below / total * 100, 1)

        # Add percentile for final seed_score
        all_opp_scores_asc = sorted(bs["seed_score"] for bs in blind_spots)
        for bs in blind_spots:
            below = bisect_left(all_opp_scores_asc, bs["seed_score"])
            bs["score_percentile"] = round(below / total * 100, 1)

    return blind_spots


def generate_topic_blind_spots(
    papers: list[dict],
    output_file: Path,
    min_shared_concepts: int = 1,
    max_similarity: float = 1.0,
    min_count: int = 1,
    entity_types: list[str] | None = None,
) -> bool:
    """
    Generate concept-level blind spot opportunities and save to file.

    Each concept is its own entry with rich metrics for ranking.

    Args:
        papers: List of paper dictionaries
        output_file: Path to save opportunities JSON
        min_shared_concepts: Minimum shared concepts between topics (default 1)
        max_similarity: Maximum Jaccard similarity (default 1.0 = no filter)
        min_count: Minimum concept count in ref_topic (default 1)
        entity_types: List of entity types to include (e.g., ["method", "concept"]).
                      Empty list or None = include all types.
                      Valid types: method, concept, task, tool, artifact, data, other

    Returns:
        True if successful
    """
    emit.status_public_info("Finding concept-level blind spots...")

    # Extract comprehensive topic data
    topic_data = extract_topic_data(papers)
    emit.status_public_info(f"Found {len(topic_data)} topics")

    if len(topic_data) < 2:
        emit.status_public_warning("Need at least 2 topics to find blind spots")
        return False

    # Find blind spots
    blind_spots = find_blind_spots(
        topic_data,
        min_shared_concepts=min_shared_concepts,
        max_similarity=max_similarity,
        min_blind_spot_count=min_count,
    )

    if not blind_spots:
        emit.status_public_warning("No blind spot opportunities found")
        return False

    emit.status_public_info(f"Found {len(blind_spots)} concept-level blind spots")

    # Filter by entity types if specified
    if entity_types:
        pre_filter_count = len(blind_spots)
        blind_spots = [bs for bs in blind_spots if bs.get("entity_type") in entity_types]
        emit.status_public_info(
            f"Filtered to {len(blind_spots)} blind spots "
            f"(entity_types: {entity_types}, removed {pre_filter_count - len(blind_spots)})"
        )

    # Log top 5 for visibility
    emit.status_public_info("Top 5 blind spots:")
    for i, bs in enumerate(blind_spots[:5]):
        emit.status_public_info(
            f"  {i + 1}. {bs['concept']} ({bs['entity_type']}) score={bs['seed_score']:.3f}"
        )

    # Save
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(blind_spots, f, indent=2, ensure_ascii=False)
        emit.status_public_success(f"Saved {len(blind_spots)} blind spots to {output_file.name}")
        return True
    except Exception as e:
        emit.status_public_error(f"Failed to save: {e}")
        raise
