#!/usr/bin/env python3
"""
Common utilities for graph generation.

Shared functions for layout, PageRank, colors, etc.
"""

import math
import os
from typing import Any

import networkx as nx
import numpy as np
import numpy.typing as npt
from aii_lib.run import emit
from scipy.stats import rankdata

try:
    import umap
except ImportError:
    umap = None

try:
    from sentence_transformers import SentenceTransformer

    _semantic_model = None  # Lazy load
except ImportError:
    SentenceTransformer = None
    _semantic_model = None


def get_temporal_window(year: int, windows: list[list[int]]) -> str | None:
    """Get temporal window label for a year."""
    for start, end in windows:
        if start <= year <= end:
            return f"{start}-{end}"
    return None


def generate_umap_layout(
    node_names: list[str],
    edge_data: dict[tuple[str, str], dict[str, Any]],
    node_data: dict[str, dict[str, Any]] | None = None,
) -> dict[str, tuple[float, float]]:
    """
    Generate graph layout using UMAP embeddings.

    Args:
        node_names: List of node names
        edge_data: Edge data with counts
        node_data: Optional node data for self-loop weights

    Returns:
        Dictionary mapping node names to (x, y) positions
    """
    if umap is None:
        emit.status_public_error("UMAP not installed. Install with: uv pip install umap-learn")
        return {name: (i * 10, i * 10) for i, name in enumerate(node_names)}

    n_nodes = len(node_names)
    if n_nodes < 2:
        return dict.fromkeys(node_names, (0, 0))

    name_to_idx = {name: i for i, name in enumerate(node_names)}

    # Create co-occurrence matrix
    co_occurrence_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    for (source, target), edge_info in edge_data.items():
        count = edge_info["count"] if isinstance(edge_info, dict) else edge_info
        if source in name_to_idx and target in name_to_idx:
            i, j = name_to_idx[source], name_to_idx[target]
            co_occurrence_matrix[i, j] = count
            co_occurrence_matrix[j, i] = count

    # Add node usage count as self-loop
    if node_data:
        for name, data in node_data.items():
            if name in name_to_idx:
                i = name_to_idx[name]
                co_occurrence_matrix[i, i] = data.get("count", 1)

    # Configure UMAP
    max_threads = min(os.cpu_count() or 1, 4)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, n_nodes - 1),
        min_dist=0.1,
        metric="cosine",
        n_jobs=max_threads,
    )

    embedding = reducer.fit_transform(co_occurrence_matrix)

    # Normalize with rank-based scaling
    embedding_array: npt.NDArray[np.floating[Any]] = np.asarray(embedding)
    x_ranks = rankdata(embedding_array[:, 0], method="average") / len(embedding_array)
    y_ranks = rankdata(embedding_array[:, 1], method="average") / len(embedding_array)

    power = 0.7
    x_scaled = np.power(x_ranks, power) * 1000
    y_scaled = np.power(y_ranks, power) * 1000

    return {name: (float(x_scaled[i]), float(y_scaled[i])) for i, name in enumerate(node_names)}


def calculate_pagerank(
    node_names: list[str], edge_data: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, float]:
    """Calculate PageRank centrality."""
    G = nx.Graph()
    G.add_nodes_from(node_names)

    for (source, target), edge_info in edge_data.items():
        weight = edge_info["count"] if isinstance(edge_info, dict) else edge_info
        G.add_edge(source, target, weight=weight)

    return nx.pagerank(G, weight="weight")


def pagerank_to_colors(pagerank: dict[str, float], node_names: list[str]) -> list[str]:
    """Generate colors based on PageRank scores.

    Low = Green, High = Red.
    """
    pr_values = [pagerank.get(name, 0) for name in node_names]
    pr_ranks = rankdata(pr_values, method="average") / len(pr_values)

    colors = []
    for percentile in pr_ranks:
        hue = (120 - percentile * 120) / 360

        def hsl_to_rgb(h, s=0.9, lightness=0.5):
            c = (1 - abs(2 * lightness - 1)) * s
            x = c * (1 - abs((h * 6) % 2 - 1))
            m = lightness - c / 2
            if h < 1 / 6:
                r, g, b = c, x, 0
            elif h < 2 / 6:
                r, g, b = x, c, 0
            elif h < 3 / 6:
                r, g, b = 0, c, x
            elif h < 4 / 6:
                r, g, b = 0, x, c
            elif h < 5 / 6:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            return int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)

        r, g, b = hsl_to_rgb(hue)
        colors.append(f"#{r:02x}{g:02x}{b:02x}")

    return colors


def log_scale_size(
    value: float, all_values: list[float], min_size: float = 6, max_size: float = 18
) -> float:
    """Apply log scaling to a value for node sizing."""
    log_values = [math.log(v + 1) for v in all_values]
    min_log = min(log_values) if log_values else 0
    max_log = max(log_values) if log_values else 1
    log_range = max_log - min_log if max_log > min_log else 1

    log_val = math.log(value + 1)
    normalized = (log_val - min_log) / log_range
    return min_size + (max_size - min_size) * normalized


def getPercentileColor(percentile: float) -> str:
    """
    Get color based on percentile (0-100).

    Uses the same color scheme as graph.js:
    - 0-20:  #3b82f6 (blue) - lowest
    - 20-40: #22c55e (green)
    - 40-60: #eab308 (yellow)
    - 60-80: #f59e0b (orange)
    - 80-100: #ef4444 (red) - highest

    Args:
        percentile: Value from 0-100

    Returns:
        Hex color string
    """
    if percentile >= 80:
        return "#ef4444"  # red
    if percentile >= 60:
        return "#f59e0b"  # orange
    if percentile >= 40:
        return "#eab308"  # yellow
    if percentile >= 20:
        return "#22c55e"  # green
    return "#3b82f6"  # blue


def generate_semantic_umap_layout(
    concept_names: list[str], model_name: str = "all-MiniLM-L6-v2"
) -> dict[str, tuple[float, float]]:
    """
    Generate graph layout using semantic embeddings + UMAP.

    Concepts are positioned by their semantic similarity (meaning),
    not by co-occurrence in papers.

    Args:
        concept_names: List of concept names to embed
        model_name: sentence-transformers model name

    Returns:
        Dictionary mapping concept names to (x, y) positions
    """
    global _semantic_model

    if SentenceTransformer is None:
        emit.status_public_error(
            "sentence-transformers not installed. Install with: uv pip install sentence-transformers"
        )
        return {name: (i * 10, i * 10) for i, name in enumerate(concept_names)}

    if umap is None:
        emit.status_public_error("UMAP not installed. Install with: uv pip install umap-learn")
        return {name: (i * 10, i * 10) for i, name in enumerate(concept_names)}

    n_concepts = len(concept_names)
    if n_concepts < 2:
        return dict.fromkeys(concept_names, (0, 0))

    # Lazy load model (CPU-friendly)
    if _semantic_model is None:
        emit.status_private_info(f"Loading semantic model: {model_name}")
        _semantic_model = SentenceTransformer(model_name)

    # Get embeddings with progress bar and batch size 256
    emit.status_public_info(f"Generating embeddings for {n_concepts} concepts...")
    embeddings = _semantic_model.encode(concept_names, show_progress_bar=True, batch_size=256)

    # Apply UMAP
    emit.status_public_info("Running UMAP...")
    max_threads = min(os.cpu_count() or 1, 4)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, n_concepts - 1),
        min_dist=0.1,
        metric="cosine",
        n_jobs=max_threads,
    )

    embedding_2d = reducer.fit_transform(embeddings)

    # Normalize with rank-based scaling
    embedding_array: npt.NDArray[np.floating[Any]] = np.asarray(embedding_2d)
    x_ranks = rankdata(embedding_array[:, 0], method="average") / len(embedding_array)
    y_ranks = rankdata(embedding_array[:, 1], method="average") / len(embedding_array)

    power = 0.7
    x_scaled = np.power(x_ranks, power) * 1000
    y_scaled = np.power(y_ranks, power) * 1000

    emit.status_public_success("Semantic UMAP complete")

    return {name: (float(x_scaled[i]), float(y_scaled[i])) for i, name in enumerate(concept_names)}
