#!/usr/bin/env python3
"""
Semantic knowledge graph generator.

Uses semantic embeddings (sentence-transformers) + UMAP to position concepts
by meaning, then positions papers near their concepts.

This reveals:
- Semantic clusters of related concepts
- Which papers contribute to which semantic areas
- Potential blind spots (semantic areas with uses but no proposes)
"""

import json
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from aii_lib.run import emit

from ._common import (
    calculate_pagerank,
    generate_semantic_umap_layout,
    pagerank_to_colors,
)


def extract_semantic_data(papers: list[dict]) -> dict[str, Any]:
    """
    Extract concept and paper data for semantic graph.

    Returns:
        Dictionary with concept_nodes, paper_nodes, edges, concept_papers mapping
    """
    concept_nodes: dict[str, dict[str, Any]] = {}
    paper_nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    # Track which papers use each concept
    concept_papers: dict[str, list[int]] = defaultdict(list)

    for paper in papers:
        paper_idx = paper.get("index", -1)
        paper_data = paper.get("paper", {}) or {}

        paper_id = f"paper:{paper_idx}"
        title = paper_data.get("title", f"Paper {paper_idx}")

        # Add paper node
        paper_nodes[paper_id] = {
            "id": paper_id,
            "label": title[:50] + "..." if len(title) > 50 else title,
            "full_title": title,
            "type": "paper",
            "year": paper_data.get("publication_year"),
            "citations": paper_data.get("cited_by_count", 0),
            "topic": paper_data.get("topic_name"),
            "doi": paper_data.get("doi"),
            "concepts": [],  # Will populate below
            "relations": {"uses": 0, "proposes": 0},
        }

        # Extract triples
        triples_section = paper.get("triples", {}) or {}
        triples = triples_section.get("triples", [])

        for triple in triples:
            name = triple.get("name")
            if not name:
                continue

            relation = triple.get("relation", "uses")

            paper_nodes[paper_id]["concepts"].append(name)
            paper_nodes[paper_id]["relations"][relation] = (
                paper_nodes[paper_id]["relations"].get(relation, 0) + 1
            )
            concept_papers[name].append(paper_idx)

            # Add/update concept node
            if name not in concept_nodes:
                concept_nodes[name] = {
                    "id": name,
                    "label": name,
                    "type": "concept",
                    "entity_type": triple.get("entity_type"),
                    "wikipedia_url": triple.get("wikipedia_url"),
                    "wikidata_id": triple.get("wikidata_id"),
                    "count": 0,
                    "total_citations": 0,
                    "relations": {"uses": 0, "proposes": 0},
                }

            concept_nodes[name]["count"] += 1
            concept_nodes[name]["total_citations"] += paper_data.get("cited_by_count", 0)
            concept_nodes[name]["relations"][relation] = (
                concept_nodes[name]["relations"].get(relation, 0) + 1
            )

            # Add edge
            edges.append(
                {
                    "source": paper_id,
                    "target": name,
                    "relation": relation,
                }
            )

    return {
        "concept_nodes": concept_nodes,
        "paper_nodes": paper_nodes,
        "edges": edges,
        "concept_papers": dict(concept_papers),
    }


def build_semantic_graph(data: dict[str, Any]) -> dict[str, Any]:
    """Build semantic graph with UMAP positions."""
    concept_nodes = data["concept_nodes"]
    paper_nodes = data["paper_nodes"]
    edges = data["edges"]

    if not concept_nodes:
        return {"nodes": [], "edges": [], "metadata": {}}

    # Get semantic UMAP positions for concepts
    concept_names = list(concept_nodes.keys())
    positions = generate_semantic_umap_layout(concept_names)

    # Build concept-concept edge data for PageRank
    concept_cooccurrence: dict[tuple[str, str], int] = defaultdict(int)
    for paper in paper_nodes.values():
        concepts = paper.get("concepts", [])
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i + 1 :]:
                key = tuple(sorted([c1, c2]))
                concept_cooccurrence[key] += 1

    edge_data = {k: {"count": v} for k, v in concept_cooccurrence.items()}

    # Calculate PageRank for concept sizing
    pagerank = calculate_pagerank(concept_names, edge_data)
    colors = pagerank_to_colors(pagerank, concept_names)

    # Build concept nodes with positions
    nodes = []

    # Compute citation percentiles for concepts
    all_concept_citations = [c["total_citations"] for c in concept_nodes.values()]
    sorted_concept_citations = sorted(all_concept_citations)
    total_concepts = len(sorted_concept_citations)

    for i, (name, concept) in enumerate(concept_nodes.items()):
        x, y = positions.get(name, (0, 0))
        relations = concept.get("relations", {})

        # Color based on PageRank (green=low, red=high centrality)
        color = colors[i]

        # Size based on citation percentile (using bisect for O(log n))
        citations = concept["total_citations"]
        below = bisect_left(sorted_concept_citations, citations)
        citation_percentile = round(below / total_concepts * 100, 1) if total_concepts > 0 else 50

        # Map percentile to size (6-18)
        size = 6 + (citation_percentile / 100) * 12

        nodes.append(
            {
                "id": name,
                "label": name,
                "type": "concept",
                "shape": "circle",
                "x": x,
                "y": y,
                "size": round(size, 1),
                "color": color,
                "count": concept["count"],
                "total_citations": concept["total_citations"],
                "citation_percentile": citation_percentile,
                "pagerank": round(pagerank.get(name, 0) * 100, 3),
                "entity_type": concept.get("entity_type"),
                "wikipedia_url": concept.get("wikipedia_url"),
                "relations": relations,
            }
        )

    # Compute citation percentiles for papers
    all_paper_citations = [p.get("citations", 0) for p in paper_nodes.values()]
    sorted_paper_citations = sorted(all_paper_citations)
    total_papers = len(sorted_paper_citations)

    # Position papers at centroid of their concepts
    for paper_id, paper in paper_nodes.items():
        concepts = paper.get("concepts", [])
        if concepts:
            # Calculate centroid of concept positions
            concept_positions = [positions.get(c, (500, 500)) for c in concepts]
            x = sum(p[0] for p in concept_positions) / len(concept_positions)
            y = sum(p[1] for p in concept_positions) / len(concept_positions)

            # Add small jitter to prevent overlap
            x += np.random.uniform(-20, 20)
            y += np.random.uniform(-20, 20)
        else:
            x, y = 500, 500  # Center if no concepts

        relations = paper.get("relations", {})

        # All papers same color (blue)
        color = "#3498db"

        # Size based on citation percentile (using bisect for O(log n))
        citations = paper.get("citations", 0)
        below = bisect_left(sorted_paper_citations, citations)
        citation_percentile = round(below / total_papers * 100, 1) if total_papers > 0 else 50

        # Map percentile to size (6-18, same as concepts)
        size = 6 + (citation_percentile / 100) * 12

        nodes.append(
            {
                "id": paper_id,
                "label": paper["label"],
                "full_title": paper["full_title"],
                "type": "paper",
                "shape": "square",
                "x": float(x),
                "y": float(y),
                "size": round(size, 1),
                "color": color,
                "year": paper.get("year"),
                "citations": paper.get("citations", 0),
                "citation_percentile": citation_percentile,
                "topic": paper.get("topic"),
                "doi": paper.get("doi"),
                "concept_count": len(concepts),
                "relations": relations,
            }
        )

    # Add concept-concept edges
    concept_edges = []
    for (c1, c2), count in concept_cooccurrence.items():
        concept_edges.append(
            {
                "source": c1,
                "target": c2,
                "type": "concept_cooccurrence",
                "count": count,
            }
        )

    # Relation counts
    relation_counts = defaultdict(int)
    for edge in edges:
        relation_counts[edge["relation"]] += 1

    return {
        "nodes": nodes,
        "edges": edges + concept_edges,
        "metadata": {
            "layout": "semantic_umap",
            "paper_count": len(paper_nodes),
            "concept_count": len(concept_nodes),
            "edge_count": len(edges),
            "concept_edge_count": len(concept_edges),
            "relation_counts": dict(relation_counts),
        },
    }


def generate_semantic_graph(papers: list[dict], output_dir: Path) -> bool:
    """Generate and save semantic knowledge graph."""
    emit.status_public_info(f"Generating semantic KG ({len(papers)} papers)")

    data = extract_semantic_data(papers)

    if not data["concept_nodes"]:
        emit.status_public_warning("No concepts found")
        return False

    graph_json = build_semantic_graph(data)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "full.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(graph_json, f, indent=2, ensure_ascii=False)
        emit.status_public_success(
            f"Saved semantic KG: {output_file.name} "
            f"({graph_json['metadata']['paper_count']} papers, "
            f"{graph_json['metadata']['concept_count']} concepts)"
        )
        return True
    except Exception as e:
        emit.status_public_error(f"Failed to save graph: {e}")
        return False
