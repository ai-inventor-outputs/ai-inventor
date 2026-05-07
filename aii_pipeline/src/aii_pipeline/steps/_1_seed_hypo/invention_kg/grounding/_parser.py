"""Wikidata entity parsing — property mapping, claim parsing, QID resolution."""

import re
from typing import Any
from urllib.parse import unquote

# Property ID to human-readable name mapping
KNOWN_PROPERTIES = {
    # Ontological
    "P31": "instance_of",
    "P279": "subclass_of",
    "P361": "part_of",
    "P527": "has_parts",
    "P1889": "different_from",
    "P460": "same_as",
    "P461": "opposite_of",
    "P2283": "uses",
    "P366": "use",
    "P155": "follows",
    "P156": "followed_by",
    "P1382": "coincident_with",
    # People/orgs
    "P178": "developer",
    "P170": "creator",
    "P287": "designed_by",
    "P112": "founded_by",
    "P3095": "practiced_by",
    # Topics
    "P921": "main_subject",
    "P910": "topic_main_category",
    "P1424": "topic_maintained_by",
    "P2578": "study_of",
    # External IDs
    "P10283": "openalex_id",
    "P6366": "mag_id",
    "P646": "freebase_id",
    "P227": "gnd_id",
    "P244": "loc_id",
    "P2581": "babelnet_id",
    "P10565": "openalex_topic_id",
    "P373": "commons_category",
    "P18": "image",
    "P5555": "schematic",
    "P1813": "short_name",
    "P2671": "google_kg_id",
    "P3417": "quora_topic_id",
    "P1417": "britannica_id",
    "P486": "mesh_id",
    "P672": "mesh_tree_code",
    # Other useful
    "P1343": "described_by_source",
    "P1482": "stack_exchange_tag",
}


def extract_title_from_wikipedia_url(url: str) -> str | None:
    """Extract Wikipedia article title from URL."""
    if not url:
        return None
    match = re.match(r"https?://en\.wikipedia\.org/wiki/(.+)", url)
    if not match:
        return None
    title = unquote(match.group(1)).replace("_", " ")
    return title


def parse_claim_value(datavalue: dict[str, Any]) -> Any:
    """Parse a Wikidata claim datavalue into a simple value."""
    if not datavalue:
        return None

    dtype = datavalue.get("type")
    value = datavalue.get("value")

    if dtype == "wikibase-entityid":
        return value.get("id")
    if dtype == "string":
        return value
    if dtype == "monolingualtext":
        return value.get("text")
    if dtype == "time":
        time_str = value.get("time", "")
        if time_str.startswith("+"):
            time_str = time_str[1:]
        return time_str[:10]
    if dtype == "quantity":
        return value.get("amount")
    if dtype == "globecoordinate":
        return {"lat": value.get("latitude"), "lon": value.get("longitude")}
    return str(value) if value else None


def parse_wikidata_entity_full(entity: dict[str, Any]) -> dict[str, Any]:
    """Parse ALL data from a Wikidata entity."""
    result = {
        "id": entity.get("id"),
        "label": entity.get("labels", {}).get("en", {}).get("value"),
        "description": entity.get("descriptions", {}).get("en", {}).get("value"),
        "aliases": [a["value"] for a in entity.get("aliases", {}).get("en", [])],
        "claims": {},
    }

    claims = entity.get("claims", {})
    for prop_id, values in claims.items():
        prop_name = KNOWN_PROPERTIES.get(prop_id, prop_id)
        parsed_values = []
        for v in values:
            mainsnak = v.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue")
            if datavalue:
                parsed = parse_claim_value(datavalue)
                if parsed is not None:
                    parsed_values.append(parsed)
        if parsed_values:
            if len(parsed_values) == 1:
                result["claims"][prop_name] = parsed_values[0]
            else:
                result["claims"][prop_name] = parsed_values

    return result


def resolve_qids_in_claims(claims: dict[str, Any], qid_labels: dict[str, str]) -> dict[str, Any]:
    """Replace Q-IDs with {id, label} objects in claims."""
    resolved = {}
    for prop_name, value in claims.items():
        if isinstance(value, str) and value.startswith("Q"):
            label = qid_labels.get(value)
            resolved[prop_name] = {"id": value, "label": label} if label else value
        elif isinstance(value, list):
            resolved_list = []
            for v in value:
                if isinstance(v, str) and v.startswith("Q"):
                    label = qid_labels.get(v)
                    resolved_list.append({"id": v, "label": label} if label else v)
                else:
                    resolved_list.append(v)
            resolved[prop_name] = resolved_list
        else:
            resolved[prop_name] = value
    return resolved
