#!/usr/bin/env python3
"""
Ontology graph generator.

Creates a graph from Wikidata ontological relationships:
- instance_of (P31): concept is an instance of class
- subclass_of (P279): concept is a subclass of another
- part_of (P361): concept is part of another
- has_parts (P527): concept has parts

Nodes: Concepts + their Wikidata parents/parts
Edges: Ontological relations (directed)
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from aii_lib.run import emit

# Ontological property names (human-readable keys from wikidata enrichment)
# Maps claim key name -> (relation name for graph, original property ID)
ONTOLOGY_PROPERTIES = {
    "instance_of": ("instance_of", "P31"),
    "subclass_of": ("subclass_of", "P279"),
    "part_of": ("part_of", "P361"),
    "has_parts": ("has_parts", "P527"),
}


def extract_ontology_data(
    papers: list[dict],
) -> tuple[dict[str, dict], list[dict], dict[str, set[str]]]:
    """
    Extract ontological relationships from Wikidata enrichment.

    Returns:
        Tuple of (nodes_dict, edges_list, node_topics)
    """
    # Track all nodes (concepts from triples + their wikidata parents)
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()

    # Track topics per node
    node_topics: dict[str, set[str]] = defaultdict(set)

    for paper in papers:
        triples_section = paper.get("triples", {}) or {}
        triples = triples_section.get("triples", [])
        paper_data = paper.get("paper", {}) or {}
        topic_name = paper_data.get("topic_name")

        for triple in triples:
            name = triple.get("name")
            if not name:
                continue

            # Track topic for this concept
            if topic_name:
                node_topics[name].add(topic_name)

            # Add concept node if not exists
            if name not in nodes:
                nodes[name] = {
                    "id": name,
                    "label": name,
                    "type": "concept",
                    "wikidata_id": triple.get("wikidata_id"),
                    "wikipedia_url": triple.get("wikipedia_url"),
                    "entity_type": triple.get("entity_type"),
                    "from_triple": True,
                }

            # Extract Wikidata ontological relationships
            wikidata = triple.get("wikidata", {})
            if not wikidata:
                continue

            claims = wikidata.get("claims", {})

            for claim_key, (relation_name, prop_id) in ONTOLOGY_PROPERTIES.items():
                prop_values = claims.get(claim_key)
                if not prop_values:
                    continue

                # Normalize to list (can be dict, list, or string)
                if isinstance(prop_values, (dict, str)):
                    prop_values = [prop_values]

                for value in prop_values:
                    # Value can be a string (label) or dict with id/label
                    if isinstance(value, dict):
                        target_id = value.get("id")
                        target_label = value.get("label", target_id)
                    else:
                        target_id = None
                        target_label = str(value) if value else None

                    if not target_label:
                        continue

                    # Add target node if not exists
                    if target_label not in nodes:
                        nodes[target_label] = {
                            "id": target_label,
                            "label": target_label,
                            "type": "wikidata_class",
                            "wikidata_id": target_id,
                            "from_triple": False,
                        }

                    # Create edge (avoid duplicates)
                    # Direction depends on relation type
                    if relation_name in ("instance_of", "subclass_of", "part_of"):
                        # concept -> parent/whole
                        edge_key = (name, target_label, relation_name)
                        source, target = name, target_label
                    else:  # has_parts
                        # concept -> part (reverse direction)
                        edge_key = (name, target_label, relation_name)
                        source, target = name, target_label

                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append(
                            {
                                "source": source,
                                "target": target,
                                "relation": relation_name,
                                "property_id": prop_id,
                            }
                        )

    emit.status_public_info(f"Found {len(nodes)} ontology nodes")
    emit.status_public_info(f"Found {len(edges)} ontology edges")

    return nodes, edges, node_topics


def build_ontology_graph(
    nodes: dict[str, dict], edges: list[dict], node_topics: dict[str, set[str]] | None = None
) -> dict[str, Any]:
    """Build the ontology graph JSON."""
    if node_topics is None:
        node_topics = {}

    # Count edges per node for sizing
    edge_counts = defaultdict(int)
    for edge in edges:
        edge_counts[edge["source"]] += 1
        edge_counts[edge["target"]] += 1

    # Build nodes list
    nodes_list = []
    for name, data in nodes.items():
        topics = node_topics.get(name, set())
        topic = next(iter(topics)) if topics else None

        nodes_list.append(
            {
                **data,
                "edge_count": edge_counts.get(name, 0),
                "size": 8 + min(edge_counts.get(name, 0) * 2, 30),
                "topic": topic,
                "topics": list(topics) if len(topics) > 1 else None,
            }
        )

    # Sort by edge count
    nodes_list.sort(key=lambda x: x["edge_count"], reverse=True)

    # Count relations
    relation_counts = defaultdict(int)
    for edge in edges:
        relation_counts[edge["relation"]] += 1

    return {
        "nodes": nodes_list,
        "edges": edges,
        "metadata": {
            "total_nodes": len(nodes_list),
            "concept_nodes": sum(1 for n in nodes_list if n.get("from_triple")),
            "wikidata_nodes": sum(1 for n in nodes_list if not n.get("from_triple")),
            "total_edges": len(edges),
            "relation_counts": dict(relation_counts),
        },
    }


def generate_ontology_graph(papers: list[dict], output_file: Path) -> bool:
    """Generate and save ontology graph."""
    emit.status_public_info("Generating ontology graph from Wikidata")

    nodes, edges, node_topics = extract_ontology_data(papers)

    if not nodes:
        emit.status_public_warning("No ontology data found")
        return False

    graph_json = build_ontology_graph(nodes, edges, node_topics)

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(graph_json, f, indent=2, ensure_ascii=False)
        emit.status_public_success(
            f"Saved ontology graph: {output_file.name} "
            f"({graph_json['metadata']['total_nodes']} nodes, "
            f"{graph_json['metadata']['total_edges']} edges)"
        )
        return True
    except Exception as e:
        emit.status_public_error(f"Failed to save graph: {e}")
        return False
