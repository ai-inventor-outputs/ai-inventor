#!/usr/bin/env python3
"""
Blind spots graph generator.

Creates a graph from topic_blind_spots.json (concept-centric format):
- Topic nodes (blind_topic and ref_topic)
- Concept nodes (blind spot concepts)
- Edges: blind_topic -> concept (concepts the topic is missing)

Shows concepts that one topic doesn't use but a related topic does.
Updated for concept-centric blind spots structure.
"""

import json
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from typing import Any

from aii_lib.run import emit

from ._common import generate_semantic_umap_layout, getPercentileColor


def load_blind_spots_data(hypo_seeds_dir: Path) -> list[dict]:
    """Load topic_blind_spots.json from hypo_seeds directory."""
    blind_spots_file = hypo_seeds_dir / "topic_blind_spots.json"

    if not blind_spots_file.exists():
        emit.status_public_warning(f"topic_blind_spots.json not found in {hypo_seeds_dir}")
        return []

    with open(blind_spots_file, encoding="utf-8") as f:
        return json.load(f)


def build_blind_spots_graph(blind_spots_data: list[dict]) -> dict[str, Any]:
    """
    Build graph from concept-centric blind spots data.

    Each entry in blind_spots_data is a single concept opportunity:
    {
        "id": "...",
        "concept": "...",
        "blind_topic": "Topic Name",
        "ref_topic": "Topic Name",
        "seed_score": 0.72,
        "score_percentile": 95.0,
        ...
    }
    """
    nodes = []
    edges = []

    # Track unique topics and concepts
    topics_seen: dict[str, dict[str, Any]] = {}
    concepts_seen: dict[str, dict[str, Any]] = {}

    # Track topic pairs for topic-topic edges
    topic_pairs = defaultdict(
        lambda: {
            "score": 0,
            "shared_concepts": [],
            "blind_spot_count": 0,
        }
    )

    for entry in blind_spots_data:
        # Extract data from concept-centric format (with hierarchical structure)
        blind_topic = entry.get("blind_topic", "")
        ref_topic = entry.get("ref_topic", "")
        concept_name = entry.get("concept", "")

        if not blind_topic or not ref_topic or not concept_name:
            continue

        seed_score = entry.get("seed_score", 0)
        score_percentile = entry.get("score_percentile", 50)

        # Extract from hierarchical sub-objects
        topic_pair = entry.get("topic_pair", {})
        blind_ref_topic_sem_dist = topic_pair.get("blind_ref_topic_sem_dist", 0)
        shared_concepts = topic_pair.get("shared_concepts", [])

        importance = entry.get("importance", {})
        count = importance.get("count", 1)

        # Extract component scores and percentiles
        importance_pct = importance.get("percentile", 50)
        transferability = entry.get("transferability", {})
        transferability_pct = transferability.get("percentile", 50)
        novelty = entry.get("novelty", {})
        novelty_pct = novelty.get("percentile", 50)
        topic_pair_pct = topic_pair.get("percentile", 50)

        entity_type = entry.get("entity_type", "concept")
        relation_breakdown = entry.get("relation_breakdown", {})

        # Add blind topic node
        if blind_topic not in topics_seen:
            topics_seen[blind_topic] = {
                "id": f"topic:{blind_topic}",
                "label": blind_topic,
                "type": "topic",
                "topic": blind_topic,
                "blind_spot_count": 0,
                "avg_seed_score": 0,
                "seed_scores": [],
                "color": "#3498db",
            }

        # Add ref topic node
        if ref_topic not in topics_seen:
            topics_seen[ref_topic] = {
                "id": f"topic:{ref_topic}",
                "label": ref_topic,
                "type": "topic",
                "topic": ref_topic,
                "blind_spot_count": 0,
                "avg_seed_score": 0,
                "seed_scores": [],
                "color": "#3498db",
            }

        # Track topic pair relationship
        pair_key = tuple(sorted([blind_topic, ref_topic]))
        topic_pairs[pair_key]["score"] = max(
            topic_pairs[pair_key]["score"], blind_ref_topic_sem_dist
        )
        topic_pairs[pair_key]["shared_concepts"] = shared_concepts[:10]
        topic_pairs[pair_key]["blind_spot_count"] += 1

        # Add or update concept node
        concept_id = f"concept:{concept_name}"
        # Per-topic score breakdown entry
        topic_score_entry = {
            "topic": blind_topic,
            "score_percentile": score_percentile,
            "importance_pct": importance_pct,
            "transferability_pct": transferability_pct,
            "novelty_pct": novelty_pct,
            "topic_pair_pct": topic_pair_pct,
        }

        if concept_name not in concepts_seen:
            concepts_seen[concept_name] = {
                "id": concept_id,
                "label": concept_name,
                "type": "blind_spot_concept",
                "entity_type": entity_type,
                "total_count": count,
                "ref_topics": [ref_topic],
                "blind_topics": [blind_topic],
                "relations": dict(relation_breakdown),
                "seed_scores": [seed_score],
                # Per-topic score breakdowns for tooltip
                "topic_scores": [topic_score_entry],
                "color": "#e74c3c",
            }
        else:
            concepts_seen[concept_name]["total_count"] += count
            if ref_topic not in concepts_seen[concept_name]["ref_topics"]:
                concepts_seen[concept_name]["ref_topics"].append(ref_topic)
            if blind_topic not in concepts_seen[concept_name]["blind_topics"]:
                concepts_seen[concept_name]["blind_topics"].append(blind_topic)
            # Merge relations
            for rel, cnt in relation_breakdown.items():
                concepts_seen[concept_name]["relations"][rel] = (
                    concepts_seen[concept_name]["relations"].get(rel, 0) + cnt
                )
            concepts_seen[concept_name]["seed_scores"].append(seed_score)
            # Add per-topic score breakdown
            concepts_seen[concept_name]["topic_scores"].append(topic_score_entry)

        # Add edge: blind_topic -> concept (shows what the topic is missing)
        edges.append(
            {
                "source": f"topic:{blind_topic}",
                "target": concept_id,
                "type": "blind_spot",
                "count": count,
                "relations": dict(relation_breakdown),
                "from_topic": ref_topic,  # Topic that has this concept
                "seed_score": seed_score,
                "score_percentile": score_percentile,
                # Component percentiles for tooltip
                "importance_pct": importance_pct,
                "transferability_pct": transferability_pct,
                "novelty_pct": novelty_pct,
                "topic_pair_pct": topic_pair_pct,
            }
        )

        # Update blind spot count and scores for blind topic
        topics_seen[blind_topic]["blind_spot_count"] += 1
        topics_seen[blind_topic]["seed_scores"].append(seed_score)

    # Calculate aggregated scores for topics
    for topic_data in topics_seen.values():
        scores = topic_data.pop("seed_scores", [])
        if scores:
            topic_data["avg_seed_score"] = round(sum(scores) / len(scores), 4)
        topic_data["size"] = 18 + min(topic_data.get("blind_spot_count", 0) // 10, 7)

    # Calculate aggregated scores for concepts (pass 1: collect scores)
    for concept_data in concepts_seen.values():
        scores = concept_data.pop("seed_scores", [])
        if scores:
            concept_data["avg_seed_score"] = round(sum(scores) / len(scores), 4)
            concept_data["max_seed_score"] = round(max(scores), 4)
        else:
            concept_data["max_seed_score"] = 0.5
        concept_data["topic_count"] = len(concept_data.get("ref_topics", []))

        # Sort topic_scores by score_percentile (highest first) for tooltip display
        topic_scores = concept_data.get("topic_scores", [])
        topic_scores.sort(key=lambda x: x.get("score_percentile", 0), reverse=True)

        # Aggregate component percentiles (max value for each) for node coloring
        if topic_scores:
            concept_data["importance_pct"] = round(
                max(ts["importance_pct"] for ts in topic_scores), 1
            )
            concept_data["transferability_pct"] = round(
                max(ts["transferability_pct"] for ts in topic_scores), 1
            )
            concept_data["novelty_pct"] = round(max(ts["novelty_pct"] for ts in topic_scores), 1)
            concept_data["topic_pair_pct"] = round(
                max(ts["topic_pair_pct"] for ts in topic_scores), 1
            )
        else:
            concept_data["importance_pct"] = 50
            concept_data["transferability_pct"] = 50
            concept_data["novelty_pct"] = 50
            concept_data["topic_pair_pct"] = 50

    # Calculate size and color based on max_seed_score percentile (pass 2)
    all_max_scores = sorted(c.get("max_seed_score", 0) for c in concepts_seen.values())
    total_concepts = len(all_max_scores)

    for concept_data in concepts_seen.values():
        score = concept_data.get("max_seed_score", 0.5)
        # Calculate percentile using binary search
        below = bisect_left(all_max_scores, score)
        score_percentile = round(below / total_concepts * 100, 1) if total_concepts > 0 else 50
        concept_data["score_percentile"] = score_percentile

        # Size based on percentile (6-18, same as concepts graph)
        concept_data["size"] = round(6 + (score_percentile / 100) * 12, 1)

        # Color based on percentile (blue->green->yellow->red)
        concept_data["color"] = getPercentileColor(score_percentile)

    nodes = list(topics_seen.values()) + list(concepts_seen.values())

    # Generate UMAP positions for all nodes (topics + concepts) based on semantic similarity
    concept_names = list(concepts_seen.keys())
    topic_names = list(topics_seen.keys())
    all_names = topic_names + concept_names

    if all_names:
        emit.status_public_info(
            f"Generating UMAP layout for {len(topic_names)} topics + {len(concept_names)} concepts..."
        )
        all_positions = generate_semantic_umap_layout(all_names)

        # Add positions to topic nodes
        for topic_name, topic_data in topics_seen.items():
            pos = all_positions.get(topic_name, (500, 500))
            topic_data["x"] = pos[0]
            topic_data["y"] = pos[1]

        # Add positions to concept nodes
        for concept_name, concept_data in concepts_seen.items():
            pos = all_positions.get(concept_name, (500, 500))
            concept_data["x"] = pos[0]
            concept_data["y"] = pos[1]

    # Add topic-topic edges
    topic_edges = []
    for (t1, t2), data in topic_pairs.items():
        topic_edges.append(
            {
                "source": f"topic:{t1}",
                "target": f"topic:{t2}",
                "type": "topic_pair",
                "blind_ref_topic_sem_dist": data["score"],
                "shared_concepts": data["shared_concepts"],
                "blind_spot_count": data["blind_spot_count"],
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "topic_edges": topic_edges,
        "metadata": {
            "topic_count": len(topics_seen),
            "concept_count": len(concepts_seen),
            "edge_count": len(edges),
            "source": "topic_blind_spots.json",
            "format": "concept_centric",
        },
    }


def generate_blind_spots_graph(hypo_seeds_dir: Path, output_file: Path) -> bool:
    """Generate and save blind spots graph."""
    emit.status_public_info("Generating blind spots graph from hypo_seeds")

    blind_spots_data = load_blind_spots_data(hypo_seeds_dir)

    if not blind_spots_data:
        emit.status_public_warning("No blind spots data found")
        return False

    graph_json = build_blind_spots_graph(blind_spots_data)

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(graph_json, f, indent=2, ensure_ascii=False)
        emit.status_public_success(
            f"Saved blind spots graph: {output_file.name} "
            f"({graph_json['metadata']['topic_count']} topics, "
            f"{graph_json['metadata']['concept_count']} concepts)"
        )
        return True
    except Exception as e:
        emit.status_public_error(f"Failed to save graph: {e}")
        return False
