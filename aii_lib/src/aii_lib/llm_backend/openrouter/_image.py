"""OpenRouter image generation — separate from main chat completion client."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import aiohttp
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from aii_lib.utils.retry import make_retry_log

from .client import OpenRouterError

_log_retry_error = make_retry_log(max_retries=4, label="OpenRouter image")


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=2, max=60, jitter=5),
    before_sleep=_log_retry_error,
)
async def _send_image_with_retry(api_key: str, payload: dict, timeout_seconds: int = 180) -> dict:
    """Send image generation request with retry logic."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with (
        aiohttp.ClientSession(timeout=timeout) as session,
        session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
        ) as response,
    ):
        if response.status != 200:
            error_text = await response.text()
            raise OpenRouterError(f"OpenRouter API error {response.status}: {error_text[:500]}")
        return await response.json()


async def generate_image(
    api_key: str,
    model: str,
    prompt: str,
    system: str | None = None,
    aspect_ratio: str | None = None,
    image_size: str | None = None,
    timeout_per_call: int = 180,
    *,
    task_id: str = "",
    task_name: str = "",
) -> bytes | None:
    """Generate an image using OpenRouter with image-capable models.

    Each emit lands on the active Run bus via ``current_run()._on(typed_msg)``.

    Args:
        api_key: OpenRouter API key
        model: Image-capable model name
        prompt: The image generation prompt
        system: System prompt
        aspect_ratio: Aspect ratio (e.g., "1:1", "16:9")
        image_size: Image size (e.g., "1K", "2K", "4K")
        timeout_per_call: Timeout in seconds per attempt
        task_id: Task identity stamped onto every emitted Run-bus message.
        task_name: Display name stamped alongside task_id.

    Returns:
        Image bytes (PNG) or None if generation failed
    """
    from aii_lib.run import get_current_run

    from ..openrouter.openrouter_llm_tel_adapter import adapt as _adapt

    def _emit(raw: dict) -> None:
        run = get_current_run()
        if run is None:
            return
        run._on(_adapt(raw, task_id, task_name))

    _emit(
        {
            "type": "prompt",
            "text": f"[Image Generation] {prompt}",
            "backend": "openrouter",
        }
    )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "modalities": ["image"],
    }

    if aspect_ratio or image_size:
        payload["image_config"] = {}
        if aspect_ratio:
            payload["image_config"]["aspect_ratio"] = aspect_ratio
        if image_size:
            payload["image_config"]["image_size"] = image_size

    start_time = datetime.now(UTC)

    try:
        result = await _send_image_with_retry(api_key, payload, timeout_per_call)

        runtime_seconds = (datetime.now(UTC) - start_time).total_seconds()
        usage = result.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_cost = usage.get("cost", 0.0) or result.get("cost", 0.0)
        actual_model = result.get("model", model)

        def emit_summary(status: str = "success"):
            _emit(
                {
                    "type": "summary",
                    "total_cost": total_cost,
                    "token_cost": total_cost,
                    "tool_cost": 0.0,
                    "model": actual_model,
                    "finish_reason": status,
                    "num_calls": 1,
                    "runtime_seconds": runtime_seconds,
                    "llm_time_seconds": runtime_seconds,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "reasoning_tokens": 0,
                    "cache_write_tokens": 0,
                    "cache_read_tokens": 0,
                    "tool_calls": {},
                    "tool_costs": {},
                    "backend": "openrouter",
                }
            )

        import base64 as b64_module

        if result.get("choices"):
            for choice in result["choices"]:
                message = choice.get("message", {})

                # Check images field
                images = message.get("images", [])
                if images:
                    for img_block in images:
                        if isinstance(img_block, dict):
                            if img_block.get("type") == "image_url":
                                url = img_block.get("image_url", {}).get("url", "")
                                if url.startswith("data:image"):
                                    mime_part = url.split(",")[0]
                                    img_format = mime_part.split("/")[1].split(";")[0]
                                    _, b64_data = url.split(",", 1)
                                    _emit(
                                        {
                                            "type": "or_img",
                                            "text": f"Image generated: {img_format.upper()} ({len(b64_data)} base64 chars)",
                                            "backend": "openrouter",
                                            "extras": {"format": img_format},
                                        }
                                    )
                                    emit_summary("success")
                                    return b64_module.b64decode(b64_data)
                        elif isinstance(img_block, str) and img_block.startswith("data:image"):
                            mime_part = img_block.split(",")[0]
                            img_format = mime_part.split("/")[1].split(";")[0]
                            _, b64_data = img_block.split(",", 1)
                            _emit(
                                {
                                    "type": "or_img",
                                    "text": f"Image generated: {img_format.upper()} ({len(b64_data)} base64 chars)",
                                    "backend": "openrouter",
                                    "extras": {"format": img_format},
                                }
                            )
                            emit_summary("success")
                            return b64_module.b64decode(b64_data)

                # Fallback: check content for data URL
                content = message.get("content", "")
                if isinstance(content, str) and "data:image" in content:
                    import re

                    match = re.search(r"data:image/[^;]+;base64,([A-Za-z0-9+/=]+)", content)
                    if match:
                        _emit(
                            {
                                "type": "or_img",
                                "text": "Image extracted from content",
                                "backend": "openrouter",
                            }
                        )
                        emit_summary("success")
                        return b64_module.b64decode(match.group(1))

                # Content array with image blocks
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "image_url":
                                url = block.get("image_url", {}).get("url", "")
                                if url.startswith("data:image"):
                                    _, b64_data = url.split(",", 1)
                                    emit_summary("success")
                                    return b64_module.b64decode(b64_data)
                            elif block.get("type") == "image":
                                data = block.get("data", "")
                                if data:
                                    emit_summary("success")
                                    return b64_module.b64decode(data)

        _emit(
            {
                "type": "warning",
                "text": f"No image in response: {json.dumps(result)[:200]}",
            }
        )
        emit_summary("no_image")
        return None

    except Exception as e:
        _emit(
            {
                "type": "error",
                "text": f"Image generation failed: {e}",
            }
        )
        logger.exception(f"Image generation failed: {e}")
        raise
