"""Scoring and analysis functions for topic blind spot detection."""

import math
from collections import defaultdict
from typing import Any

import numpy as np
from aii_lib.run import emit

# Optional: sentence-transformers for semantic distance
try:
    from sentence_transformers import SentenceTransformer

    _semantic_model = None
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    _semantic_model = None
    HAS_SENTENCE_TRANSFORMERS = False


def extract_topic_data(papers: list[dict]) -> dict[str, dict[str, Any]]:
    """
    Extract comprehensive topic data including concept details.

    Returns:
        {topic_name: {
            "field": str,
            "subfield": str,
            "paper_count": int,
            "concepts": {concept_name: {
                "count": int,
                "uses": int,
                "proposes": int,
                "entity_type": str,
                "papers": [{paper_id, citations, year}, ...]
            }},
            "concept_cooccurrence": {concept: {other_concept: count}}
        }}
    """
    topic_data: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "field": None,
            "subfield": None,
            "paper_count": 0,
            "concepts": defaultdict(
                lambda: {
                    "count": 0,
                    "uses": 0,
                    "proposes": 0,
                    "entity_type": None,
                    "papers": [],
                }
            ),
            "concept_cooccurrence": defaultdict(lambda: defaultdict(int)),
        }
    )

    for paper in papers:
        paper_data = paper.get("paper", {}) or {}
        topic_name = paper_data.get("topic_name")
        if not topic_name:
            continue

        paper_id = paper_data.get("id", "")
        citations = paper_data.get("cited_by_count", 0)
        year = paper_data.get("publication_year", 0)

        topic_data[topic_name]["paper_count"] += 1
        if topic_data[topic_name]["field"] is None:
            topic_data[topic_name]["field"] = paper_data.get("field")
            topic_data[topic_name]["subfield"] = paper_data.get("subfield")

        # Extract concepts from triples
        triples_section = paper.get("triples", {}) or {}
        triples = triples_section.get("triples", [])

        # Track concepts in this paper for co-occurrence
        paper_concepts = set()

        for triple in triples:
            name = triple.get("name")
            if not name:
                continue

            paper_concepts.add(name)
            concept_data = topic_data[topic_name]["concepts"][name]
            concept_data["count"] += 1

            relation = triple.get("relation", "uses")
            if relation == "uses":
                concept_data["uses"] += 1
            elif relation == "proposes":
                concept_data["proposes"] += 1

            # Keep first entity_type found (they should be consistent)
            if concept_data["entity_type"] is None:
                concept_data["entity_type"] = triple.get("entity_type")

            # Track paper info for this concept
            concept_data["papers"].append(
                {"paper_id": paper_id, "citations": citations, "year": year}
            )

        # Build co-occurrence (concepts appearing in same paper)
        for c1 in paper_concepts:
            for c2 in paper_concepts:
                if c1 != c2:
                    topic_data[topic_name]["concept_cooccurrence"][c1][c2] += 1

    # Convert defaultdicts to regular dicts
    result = {}
    for topic, data in topic_data.items():
        result[topic] = {
            "field": data["field"],
            "subfield": data["subfield"],
            "paper_count": data["paper_count"],
            "concepts": {
                c: {
                    "count": d["count"],
                    "uses": d["uses"],
                    "proposes": d["proposes"],
                    "entity_type": d["entity_type"],
                    "papers": d["papers"],
                }
                for c, d in data["concepts"].items()
            },
            "concept_cooccurrence": {
                c: dict(others) for c, others in data["concept_cooccurrence"].items()
            },
        }

    return result


def calculate_topic_similarity(
    topic_a_concepts: set[str], topic_b_concepts: set[str]
) -> tuple[float, set[str]]:
    """
    Calculate Jaccard similarity and shared concepts between two topics.

    Returns:
        (similarity_score, shared_concepts)
    """
    shared = topic_a_concepts & topic_b_concepts
    union = topic_a_concepts | topic_b_concepts

    if not union:
        return 0.0, set()

    similarity = len(shared) / len(union)
    return similarity, shared


def compute_topic_centroid_distance(
    topic_a_concepts: set[str],
    topic_b_concepts: set[str],
    embeddings_cache: dict[str, np.ndarray],
    model,
) -> float:
    """
    Compute semantic distance between two topics using concept centroids.

    Instead of comparing topic names, we compare the centroid (average embedding)
    of all concepts in each topic. This better reflects actual topic content.

    Returns:
        Distance score (0-1), higher = more semantically different
    """
    if model is None or not topic_a_concepts or not topic_b_concepts:
        return 0.5  # Fallback if no model or empty topics

    # Get embeddings for all concepts in topic A
    topic_a_embeddings = []
    for concept in topic_a_concepts:
        if concept in embeddings_cache:
            topic_a_embeddings.append(embeddings_cache[concept])

    # Get embeddings for all concepts in topic B
    topic_b_embeddings = []
    for concept in topic_b_concepts:
        if concept in embeddings_cache:
            topic_b_embeddings.append(embeddings_cache[concept])

    if not topic_a_embeddings or not topic_b_embeddings:
        return 0.5  # Fallback if no embeddings

    # Compute centroids
    centroid_a = np.mean(topic_a_embeddings, axis=0)
    centroid_b = np.mean(topic_b_embeddings, axis=0)

    # Cosine similarity → distance
    similarity = np.dot(centroid_a, centroid_b) / (
        np.linalg.norm(centroid_a) * np.linalg.norm(centroid_b) + 1e-8
    )

    # Convert to distance (0 = identical, 1 = orthogonal)
    distance = 1 - max(0, min(1, float(similarity)))
    return round(float(distance), 4)


def compute_idf(concept: str, all_topic_concepts: dict[str, set[str]]) -> float:
    """
    Compute standard IDF (inverse document frequency) for a concept.

    Returns:
        log(total_topics / topics_with_concept)
        Higher = more rare/specialized
    """
    total_topics = len(all_topic_concepts)
    if total_topics == 0:
        return 0.0

    topics_with_concept = sum(1 for concepts in all_topic_concepts.values() if concept in concepts)

    if topics_with_concept == 0:
        return 0.0

    return math.log(total_topics / topics_with_concept)


def compute_tf_idf(concept_count: int, total_concepts_in_topic: int, idf: float) -> float:
    """
    Compute TF-IDF score.

    TF = concept_count / total_concepts_in_topic
    TF-IDF = TF * IDF

    Returns:
        TF-IDF score (raw, will be z-score + sigmoid normalized later)
    """
    if total_concepts_in_topic == 0:
        return 0.0

    tf = concept_count / total_concepts_in_topic
    return tf * idf


def compute_citation_weight(papers: list[dict]) -> float:
    """Compute average citations for papers using this concept."""
    if not papers:
        return 0.0
    total_citations = sum(p.get("citations", 0) for p in papers)
    return total_citations / len(papers)


def compute_recency_score(
    papers: list[dict], global_min_year: int, global_max_year: int
) -> tuple[float, float]:
    """
    Compute recency score and average publication year.

    Returns:
        (avg_year, recency_score)
        recency_score: 0-1, higher = more recent
    """
    if not papers:
        return 0.0, 0.0

    years = [p.get("year", 0) for p in papers if p.get("year", 0) > 0]
    if not years:
        return 0.0, 0.0

    avg_year = sum(years) / len(years)

    year_range = global_max_year - global_min_year
    if year_range <= 0:
        return avg_year, 0.5

    recency = (avg_year - global_min_year) / year_range
    return round(avg_year, 1), round(min(max(recency, 0), 1), 3)


def compute_bridge_potential(
    concept: str, shared_concepts: set[str], cooccurrence: dict[str, dict[str, int]]
) -> int:
    """Count cooccurrences of a blind spot concept with shared concepts.

    Count how many shared concepts co-occur with this blind spot concept.
    Higher = easier transfer path.
    """
    if concept not in cooccurrence:
        return 0

    concept_cooccurs = set(cooccurrence[concept].keys())
    return len(concept_cooccurs & shared_concepts)


def zscore_sigmoid_normalize(value: float, all_values: list[float]) -> float:
    """
    Z-score standardization followed by sigmoid squashing to 0-1.

    This preserves magnitude differences better than percentile:
    - Z-score: "how many standard deviations from mean"
    - Sigmoid: smoothly compresses to (0, 1) range

    Properties:
    - Mean value → 0.5
    - 1 std above mean → ~0.73
    - 2 std above mean → ~0.88
    - Outliers compressed but not collapsed (3 std still > 2 std)
    """
    if not all_values or len(all_values) < 2:
        return 0.5

    mean = np.mean(all_values)
    std = np.std(all_values)

    if std < 1e-8:  # All values identical
        return 0.5

    # Z-score: how many standard deviations from mean
    z = (value - mean) / std

    # Sigmoid: squash to (0, 1) range
    # 1 / (1 + e^(-z))
    sigmoid = 1 / (1 + np.exp(-z))

    return float(sigmoid)


def compute_topic_pair_score(topic_sem_dist: float, topic_shared_ratio: float) -> float:
    """
    Compute how good this topic pairing is for knowledge transfer.

    We want topics that are:
    - Semantically different (high distance = novel transfer)
    - But share some concepts (bridge exists for feasibility)

    Inputs are z-score + sigmoid normalized.
    """
    score = (
        topic_sem_dist * 0.6  # want different topics
        + topic_shared_ratio * 0.4  # but with bridge
    )
    return round(score, 4)


def compute_concept_ref_importance_score(
    concept_ref_tfidf: float, concept_citation: float, paper_recency: float
) -> float:
    """
    Compute how important/validated this concept is in ref_topic.

    High score = high TF-IDF (frequent locally + rare globally), highly-cited, and recent.

    All inputs are z-score + sigmoid normalized.

    TF-IDF replaces count_percentile to capture both:
    - TF: How important is concept in this ref_topic
    - IDF: How specialized/rare is concept globally
    """
    score = concept_ref_tfidf * 0.5 + concept_citation * 0.3 + paper_recency * 0.2
    return round(score, 4)


def compute_concept_transferability_score(
    blind_shared_concept_cooccur: float,
) -> float:
    """
    Compute how transferable this concept is to another domain.

    High score = bridges exist (co-occurring shared concepts make transfer easier).

    Input is z-score + sigmoid normalized.
    """
    return round(blind_shared_concept_cooccur, 4)


def compute_concept_novelty_score(
    min_blind_concept_dist: float,
) -> float:
    """
    Compute how novel this concept would be to the blind_topic.

    High score = semantically distant from blind_topic's existing concepts.
    Uses min pairwise distance - if close to ANY existing concept, not novel.

    Input is z-score + sigmoid normalized.
    """
    return round(min_blind_concept_dist, 4)


def compute_seed_score(
    topic_pair_score: float,
    concept_ref_importance_score: float,
    concept_transferability_score: float,
    concept_novelty_score: float,
) -> float:
    """
    Final seed score combining all hierarchical scores.

    Weights: Equal 25% each
    - topic_pair: 25% - good topic pairing is foundation
    - importance: 25% - concept should be validated/important
    - transferability: 25% - co-occurrence with shared concepts
    - novelty: 25% - should be new to blind_topic
    """
    score = (
        topic_pair_score * 0.25
        + concept_ref_importance_score * 0.25
        + concept_transferability_score * 0.25
        + concept_novelty_score * 0.25
    )
    return round(score, 4)


def get_semantic_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy load semantic model with HF token authentication."""
    global _semantic_model
    if _semantic_model is None and HAS_SENTENCE_TRANSFORMERS:
        import os

        # Set HF token from config or environment
        if "HF_TOKEN" not in os.environ:
            try:
                from pathlib import Path as _Path

                from aii_pipeline.utils import PipelineConfig

                _default_cfg_dir = _Path(__file__).resolve().parents[6] / "aii_config" / "pipeline"
                cfg = PipelineConfig.from_yaml(_default_cfg_dir)
                hf_token = cfg.raw.get("api_keys", {}).get("huggingface", "")
                if hf_token:
                    os.environ["HF_TOKEN"] = hf_token
                    emit.status_private_debug("Set HF_TOKEN from config")
            except Exception:
                pass  # Config not available, continue without token

        emit.status_private_info(f"Loading semantic model: {model_name}")
        try:
            # Try local cache first (fast, no network)
            _semantic_model = SentenceTransformer(model_name, local_files_only=True)
            emit.status_private_info("Loaded model from local cache")
        except Exception:
            # Not in cache, download it
            emit.status_private_info("Model not cached, downloading...")
            _semantic_model = SentenceTransformer(model_name)
            emit.status_private_info("Model downloaded and cached")
    return _semantic_model


def compute_semantic_distance_to_topic(
    concept: str,
    topic_concepts: set[str],
    embeddings_cache: dict[str, np.ndarray],
    model,
) -> float | None:
    """
    Compute semantic distance between a concept and the closest concept in a topic.

    Uses pairwise MIN distance instead of centroid - if the concept is close to
    ANY existing concept in the topic, it's not novel (centroid would miss outlier matches).

    Returns:
        Distance score (0-1), higher = more foreign/distant from topic
        None if embeddings unavailable
    """
    if model is None or not topic_concepts:
        return None

    # Get concept embedding
    if concept not in embeddings_cache:
        embeddings_cache[concept] = model.encode([concept], show_progress_bar=False)[0]

    concept_emb = embeddings_cache[concept]
    concept_norm = np.linalg.norm(concept_emb)

    # Compute distance to each topic concept, keep minimum
    min_distance = 1.0
    for tc in topic_concepts:
        if tc not in embeddings_cache:
            embeddings_cache[tc] = model.encode([tc], show_progress_bar=False)[0]
        tc_emb = embeddings_cache[tc]
        tc_norm = np.linalg.norm(tc_emb)

        # Cosine similarity → distance
        similarity = np.dot(concept_emb, tc_emb) / (concept_norm * tc_norm + 1e-8)
        distance = 1 - max(0, min(1, float(similarity)))

        if distance < min_distance:
            min_distance = distance

    return round(float(min_distance), 3)


def batch_encode_concepts(
    concepts: list[str],
    embeddings_cache: dict[str, np.ndarray],
    model,
) -> None:
    """
    Encode concepts that aren't in cache.

    Args:
        concepts: List of concept strings to encode
        embeddings_cache: Cache dict to store embeddings
        model: SentenceTransformer model
    """
    if model is None:
        raise ValueError("Semantic model is required but not available")

    # Find concepts not yet encoded
    to_encode = [c for c in concepts if c not in embeddings_cache]
    if not to_encode:
        return

    # Encode with progress bar and batch size 256
    embeddings = model.encode(to_encode, show_progress_bar=True, batch_size=256)

    for concept, emb in zip(to_encode, embeddings, strict=False):
        embeddings_cache[concept] = emb
