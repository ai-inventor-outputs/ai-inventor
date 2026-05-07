#!/usr/bin/env python3
"""
Co-occurrence graph generator.

Creates a graph where:
- Nodes: Concepts from triples
- Edges: Concepts that co-occur in the same paper
- Edge weight: Number of co-occurrences
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from aii_lib.run import emit

from ._common import (
    calculate_pagerank,
    generate_umap_layout,
    get_temporal_window,
    log_scale_size,
    pagerank_to_colors,
)

# Typing
NodeData = dict[str, dict[str, Any]]
EdgeData = dict[tuple[str, str], dict[str, Any]]


def extract_cooccurrence_data(
    papers: list[dict], temporal_windows: list[list[int]] | None = None
) -> tuple[NodeData, EdgeData]:
    """
    Extract concept co-occurrence from papers.

    Returns:
        Tuple of (node_data, edge_data)
    """
    if temporal_windows is None:
        temporal_windows = [[2018, 2020], [2021, 2023], [2024, 2025]]

    def make_temporal_slices():
        return {
            f"{w[0]}-{w[1]}": {"count": 0, "papers": [], "citations": 0} for w in temporal_windows
        }

    node_data: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "papers": [],
            "total_citations": 0,
            "entity_type": None,
            "wikipedia_url": None,
            "wikidata_id": None,
            "relations": {"uses": 0, "extends": 0, "proposes": 0},
            "relevances": [],
            "temporal_slices": make_temporal_slices(),
            "topics": set(),  # Track which topics this concept appears in
        }
    )
    edge_data: EdgeData = defaultdict(lambda: {"count": 0, "papers": []})

    for paper in papers:
        triples_section = paper.get("triples", {}) or {}
        triples = triples_section.get("triples", [])
        if not triples:
            continue

        paper_idx = paper.get("index", -1)
        paper_data = paper.get("paper", {}) or {}
        cited_by_count = paper_data.get("cited_by_count", 0)
        publication_year = paper_data.get("publication_year")
        topic_name = paper_data.get("topic_name")

        temporal_window = None
        if publication_year and temporal_windows:
            temporal_window = get_temporal_window(publication_year, temporal_windows)

        concept_names = []
        for triple in triples:
            name = triple.get("name")
            if not name:
                continue

            concept_names.append(name)
            node_data[name]["count"] += 1
            node_data[name]["papers"].append(paper_idx)
            node_data[name]["total_citations"] += cited_by_count

            # Track topic
            if topic_name:
                node_data[name]["topics"].add(topic_name)

            if node_data[name]["entity_type"] is None and triple.get("entity_type"):
                node_data[name]["entity_type"] = triple.get("entity_type")
            if node_data[name]["wikipedia_url"] is None and triple.get("wikipedia_url"):
                node_data[name]["wikipedia_url"] = triple.get("wikipedia_url")
            if node_data[name]["wikidata_id"] is None and triple.get("wikidata_id"):
                node_data[name]["wikidata_id"] = triple.get("wikidata_id")

            relation = triple.get("relation", "uses")
            if relation in node_data[name]["relations"]:
                node_data[name]["relations"][relation] += 1

            relevance = triple.get("relevance")
            if relevance:
                node_data[name]["relevances"].append(relevance)

            if temporal_window and temporal_window in node_data[name]["temporal_slices"]:
                slice_data = node_data[name]["temporal_slices"][temporal_window]
                slice_data["count"] += 1
                slice_data["papers"].append(paper_idx)
                slice_data["citations"] += cited_by_count

        # Track co-occurrences
        for i, c1 in enumerate(concept_names):
            for c2 in concept_names[i + 1 :]:
                edge_key = tuple(sorted([c1, c2]))
                edge_data[edge_key]["count"] += 1
                edge_data[edge_key]["papers"].append(paper_idx)

    emit.status_public_info(f"Found {len(node_data)} unique concepts")
    emit.status_public_info(f"Found {len(edge_data)} co-occurrence edges")

    return dict(node_data), dict(edge_data)


def build_cooccurrence_graph(node_data: NodeData, edge_data: EdgeData) -> dict[str, Any]:
    """Build the co-occurrence graph JSON."""
    sorted_nodes = sorted(node_data.items(), key=lambda x: x[1]["count"], reverse=True)

    node_names_sorted = [name for name, _ in sorted_nodes]
    pagerank = calculate_pagerank(node_names_sorted, edge_data)
    colors = pagerank_to_colors(pagerank, node_names_sorted)

    # Get citations for size scaling
    all_citations = [data.get("total_citations", 0) for _, data in sorted_nodes]

    nodes = []
    positions = generate_umap_layout(node_names_sorted, edge_data, node_data)

    for idx, (name, data) in enumerate(sorted_nodes):
        x, y = positions.get(name, (0, 0))
        size = log_scale_size(data.get("total_citations", 0), all_citations)

        temporal_slices = {}
        for window, slice_data in data.get("temporal_slices", {}).items():
            if slice_data.get("count", 0) > 0:
                temporal_slices[window] = {
                    "count": slice_data["count"],
                    "citations": slice_data.get("citations", 0),
                }

        # Get topic (use first if multiple, or None)
        topics = data.get("topics", set())
        topic = next(iter(topics)) if topics else None

        nodes.append(
            {
                "x": x,
                "y": y,
                "id": name,
                "label": name,
                "size": round(size, 2),
                "color": colors[idx],
                "count": data["count"],
                "total_citations": data.get("total_citations", 0),
                "pagerank": round(pagerank.get(name, 0) * 1000, 3),
                "entity_type": data.get("entity_type"),
                "wikipedia_url": data.get("wikipedia_url"),
                "wikidata_id": data.get("wikidata_id"),
                "relations": data.get("relations", {}),
                "temporal_slices": temporal_slices if temporal_slices else None,
                "topic": topic,
                "topics": list(topics) if len(topics) > 1 else None,  # Only include if multi-topic
            }
        )

    # Build edges
    edge_counts = [e["count"] for e in edge_data.values()]
    min_count = min(edge_counts) if edge_counts else 1
    max_count = max(edge_counts) if edge_counts else 1

    edges = []
    for (source, target), edge_info in edge_data.items():
        co_occur_count = edge_info["count"]
        papers = edge_info.get("papers", [])

        if max_count > min_count:
            normalized = (co_occur_count - min_count) / (max_count - min_count)
            width = 0.3 + 4.7 * normalized
        else:
            width = 0.3

        edges.append(
            {
                "sourceID": source,
                "targetID": target,
                "width": round(width, 2),
                "count": co_occur_count,
                "papers": papers,
            }
        )

    return {"nodes": nodes, "edges": edges}


def generate_cooccurrence_graph(
    papers: list[dict],
    output_file: Path,
    temporal_windows: list[list[int]] | None = None,
) -> bool:
    """Generate and save co-occurrence graph."""
    emit.status_public_info(f"Generating co-occurrence graph ({len(papers)} papers)")

    node_data, edge_data = extract_cooccurrence_data(papers, temporal_windows)

    if not node_data:
        emit.status_public_warning("No concepts found")
        return False

    graph_json = build_cooccurrence_graph(node_data, edge_data)

    if temporal_windows:
        graph_json["metadata"] = {"temporal_windows": [f"{w[0]}-{w[1]}" for w in temporal_windows]}

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(graph_json, f, indent=2, ensure_ascii=False)
        emit.status_public_success(
            f"Saved co-occurrence graph: {output_file.name} "
            f"({len(graph_json['nodes'])} nodes, {len(graph_json['edges'])} edges)"
        )
        return True
    except Exception as e:
        emit.status_public_error(f"Failed to save graph: {e}")
        return False
