"""OpenRouter response extractors."""

import re
from typing import Any


def extract_output(response: Any) -> str:
    """Extract text output from response."""
    if not response:
        return ""

    # Try choices[0].message.content
    if hasattr(response, "choices") and response.choices:
        first_choice = response.choices[0]
        if hasattr(first_choice, "message") and first_choice.message:
            msg = first_choice.message

            # Check for refusal first
            refusal = getattr(msg, "refusal", None)
            if refusal:
                return ""  # Model refused, return empty

            # Try content field (standard)
            content = getattr(msg, "content", None)
            if content:
                return content

            # Try parsed field (some structured output implementations)
            parsed = getattr(msg, "parsed", None)
            if parsed:
                import json

                return json.dumps(parsed) if not isinstance(parsed, str) else parsed

    return ""


def extract_json_from_text(text: str) -> str:
    """Extract JSON object from text that may contain non-JSON content.

    Handles cases where models return reasoning/thinking before JSON,
    wrap JSON in markdown code blocks, or output multiple JSON objects
    (returns the best one - longest with content).

    Args:
        text: Raw text that may contain JSON

    Returns:
        Extracted JSON string, or empty string if no JSON object found
    """
    if not text:
        return ""

    # Try to parse as-is first (fast path)
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        # Check if it's a single valid JSON (no concatenated objects)
        try:
            import json

            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass  # Multiple objects concatenated, extract below

    # Remove markdown code blocks: ```json ... ``` or ``` ... ```
    json_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_block_match:
        return json_block_match.group(1).strip()

    # Find all complete JSON objects using brace matching
    json_objects = []
    pos = 0
    while pos < len(text):
        start = text.find("{", pos)
        if start == -1:
            break

        depth = 0
        in_string = False
        escape = False
        end = -1
        for i, char in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end > start:
            json_objects.append(text[start:end])
            pos = end
        else:
            break  # Incomplete JSON, stop

    if not json_objects:
        return ""

    # Return the best JSON object - prefer longer ones with actual content
    # (handles cases where model outputs empty objects before the real one)
    best = json_objects[0]
    best_score = 0
    for obj in json_objects:
        # Score based on length, penalize if looks empty (short answer field)
        score = len(obj)
        if '"answer":""' in obj.replace(" ", "") or '"answer": ""' in obj:
            score = 0  # Empty answer, deprioritize
        if score > best_score:
            best_score = score
            best = obj

    return best


def extract_usage(response: Any) -> dict:
    """Extract usage data from response.

    When usage accounting is enabled (usage: {include: true}), OpenRouter returns
    the actual cost charged in the response. This is more accurate than calculating
    from hardcoded prices.

    See: https://openrouter.ai/docs/guides/usage-accounting
    """
    if not response or not hasattr(response, "usage") or not response.usage:
        return {}

    usage = response.usage
    result = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
    }

    # Extract actual cost from OpenRouter (requires usage: {include: true} in request)
    cost = getattr(usage, "cost", None)
    if cost is not None:
        result["cost"] = round(float(cost), 10)

    # Extract completion_tokens_details (reasoning_tokens, etc.)
    completion_details = getattr(usage, "completion_tokens_details", None)
    if completion_details:
        reasoning_tokens = getattr(completion_details, "reasoning_tokens", None)
        if reasoning_tokens:
            result["reasoning_tokens"] = int(reasoning_tokens)

    # Extract prompt_tokens_details (cached_tokens, etc.)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    if prompt_details:
        cached_tokens = getattr(prompt_details, "cached_tokens", None)
        if cached_tokens:
            result["cached_tokens"] = int(cached_tokens)

    return result
