"""LLM-summarizer engine — fallback chain walker for plain-text completions.

Single low-level entry point used by every summarizer in the project
(per-message buffer, periodic interim summary, run-title generator):
:func:`summarize` — synchronous, ``requests.Session``-pooled. Per-tier
kernel-level timeout (``SO_TIMEOUT``) means a stalled tier truly gives
up at the requested deadline.

Each call walks ``chain`` tier-by-tier with ``allow_fallbacks: False``
at the OpenRouter API so the provider can't silently spill over —
the next tier in our chain IS the explicit fallback. Single attempt
per tier (``max_retries=1``); on any failure (429 / timeout / empty
result / upstream error) we fall straight through to the next tier.
First tier that produces non-empty output wins.

Prompts live next to their callers (the per-message prompt is in
``aii_lib.run.llm_summary``, the interim prompt is in
``aii_lib.run.sinks.interim_summary``, the title prompt is in
``aii_lib.run.sinks.title``); this module owns only the
chain-walking infrastructure.
"""

import json
import os
import re
import threading

from loguru import logger

# ---------------------------------------------------------------------------
# Default fallback chain
# ---------------------------------------------------------------------------

# Tiered fallback. Each tier passes ``allow_fallbacks: False`` at the
# OpenRouter call site so the provider can't silently spill over to a
# non-listed alternative — the next tier in this chain IS the explicit
# fallback. Combined with ``max_retries=1`` on the OpenRouterClient,
# each tier is a single attempt.
#
#   1. openai/gpt-oss-20b on Groq — priority tier (smallest
#      reasoning-capable model healthy on Groq).
#   2. openai/gpt-oss-120b on Cerebras — DIFFERENT provider, so when
#      Groq throttles the earlier tier we don't keep hitting the same
#      per-IP backpressure.
#   3. openai/gpt-oss-120b on Groq — same provider, bigger model.
#   4. openai/gpt-oss-120b on SambaNova — fourth provider for fault
#      isolation when Groq+Cerebras both throttle. Same model family as
#      tiers 1-3, ~5s p50 / 16s p90 at 20-concurrent on the real
#      ~12k-token interim-summary prompt — comparable to Groq, well
#      under the 20 s per-tier timeout. Replaced
#      ``google/gemma-4-26b-a4b-it`` on google-vertex which was timing
#      out at this tier and reducing it to a no-op survivability slot.
#   5-7. openai/gpt-oss-120b on BaseTen / Google / Nebius —
#      survivability backstops; three more independent providers added
#      after benchmarking the full 17-provider gpt-oss-120b matrix.
#      p50s 5-7s on the 12k-token interim-summary prompt; Nebius's p90
#      (~15s) brushes the sum_msg/title 10 s timeout, so those workloads
#      will occasionally timeout at this tier — acceptable since these
#      tiers fire only after the upstream four have already failed.
# Per-tier ``max_context_tokens`` is the model's OpenRouter-published
# context length (verified via the ``aii-openrouter-llms`` skill). The
# walker truncates the user prompt to fit *before* the request fires,
# leaving :data:`PROMPT_OVERHEAD_TOKENS` headroom for the system message
# + the model's own output tokens — caller passes whatever it has, the
# walker per-tier clamps. A bigger-window tier downstream can attempt a
# longer prompt than an earlier small-window tier was given.
#
# Removed: meta-llama/llama-4-scout on Groq (was tier 2). The slug
# returned 404 from OpenRouter on every call (confirmed in
# run_anDiSbGaeE4M). Drop saves the wasted retry attempt; if a
# llama-family fallback is needed re-add with a verified slug.
DEFAULT_FALLBACK_CHAIN = [
    {
        "model": "openai/gpt-oss-20b",
        "provider_order": ["Groq"],
        "max_context_tokens": 131_072,
    },
    {
        "model": "openai/gpt-oss-120b",
        "provider_order": ["Cerebras"],
        "max_context_tokens": 131_072,
    },
    {
        "model": "openai/gpt-oss-120b",
        "provider_order": ["Groq"],
        "max_context_tokens": 131_072,
    },
    {
        "model": "openai/gpt-oss-120b",
        "provider_order": ["SambaNova"],
        "max_context_tokens": 131_072,
    },
    {
        "model": "openai/gpt-oss-120b",
        "provider_order": ["BaseTen"],
        "max_context_tokens": 128_072,
    },
    {
        "model": "openai/gpt-oss-120b",
        "provider_order": ["Google"],
        "max_context_tokens": 131_072,
    },
    {
        "model": "openai/gpt-oss-120b",
        "provider_order": ["Nebius"],
        "max_context_tokens": 131_072,
    },
]

# Approximate token-to-char ratio for the per-tier truncation cap.
# Empirically the actual ratio is closer to 3 (English mixed with code
# / JSON / source paths), so 4 OVERESTIMATES capacity and the truncated
# prompt still overflowed the small-tier 131K limit on ~98% of calls
# (1205/1225 failures observed in run_anDiSbGaeE4M). 3 errs toward
# "fits" rather than "still overflows".
CHARS_PER_TOKEN = 3

# Headroom subtracted from each tier's context budget when truncating
# the user prompt — covers the system message + the model's own output
# tokens + tokenizer slop. 5k tokens ≈ 20k chars.
PROMPT_OVERHEAD_TOKENS = 5_000


# ---------------------------------------------------------------------------
# Output scrubbing (small models occasionally wrap in JSON or fences)
# ---------------------------------------------------------------------------


def _scrub_plain_text(text: str) -> str:
    """Strip JSON wrappers, code fences, and stray quotes from plain-text.

    Small models occasionally emit
    ``{"summary": "actual text"}`` or wrap the answer in ```fences```
    despite instructions; pull the inner string out so the dashboard
    never shows the wrapper.
    """
    s = (text or "").strip()
    if not s:
        return s
    # ``` fences (with optional language tag) on first/last lines.
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip()
    # JSON object → pull "summary"/"text"/"current"/"output" string out.
    if s.startswith("{"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                for key in ("summary", "text", "current", "output", "title"):
                    val = parsed.get(key)
                    if isinstance(val, str) and val.strip():
                        s = val.strip()
                        break
        except Exception:
            # Malformed JSON — peel braces/keys/quotes off the edges.
            s = re.sub(
                r"^\{+\s*\"?(?:summary|text|current|output|title)\"?\s*:\s*\"?",
                "",
                s,
                flags=re.IGNORECASE,
            )
            s = re.sub(r"\"?\s*\}+\s*$", "", s)
            s = s.strip().strip('"')
    # Whole-string quotes wrap.
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    return s


def _summarize_provider_error(e: Exception) -> str:
    """Compress provider exceptions into a one-liner for hot-path logging.

    The full chain-of-fallbacks pattern fires often; common errors
    (429s from a single provider) shouldn't dump a 30-line traceback
    every time. Specific failures get short tags; anything unfamiliar
    falls back to the first line of the exception text.
    """
    s = str(e).strip()
    cls = type(e).__name__
    if cls in (
        "TimeoutError",
        "asyncio.TimeoutError",
        "ReadTimeout",
        "ConnectTimeout",
        "WriteTimeout",
        "PoolTimeout",
    ):
        return f"timed out ({cls})"
    if cls == "CancelledError":
        return f"cancelled ({cls})"
    if "429" in s:
        return "rate-limited (429)"
    if "401" in s or "Unauthorized" in s:
        return "unauthorized (401)"
    if "404" in s:
        return "not found (404)"
    if "503" in s or "Service Unavailable" in s:
        return "service unavailable (503)"
    if "timeout" in s.lower() or "timed out" in s.lower():
        return "timed out"
    first_line = s.split("\n", 1)[0][:140]
    return first_line if first_line else cls


# ---------------------------------------------------------------------------
# Sync chain walker — production hot path
# ---------------------------------------------------------------------------

# The async/aiohttp version below suffers a 20s hang on the FIRST call
# from a fresh worker-thread asyncio loop in production:
#   - aiohttp's _resolve_host wraps DNS in asyncio.shield(), so the
#     per-request timeout doesn't fire (cancellation can't cut DNS short)
#   - new_event_loop + close per call pays cold-start every time
#   - aiodns Channel created per loop never gets warm
# v25 (commit 452ee51f9) replaced asyncio.run in the buffer worker with
# sync requests.Session — single shared session with connection pooling.
# Per-tier ``timeout`` is enforced by the kernel SO_TIMEOUT, so a stuck
# tier truly gives up at the requested deadline.

_session_lock = threading.Lock()
_shared_session = None


def _get_shared_session(api_key: str) -> object:
    """Lazy-init a single ``requests.Session`` shared across all summary calls.

    Connection pooling means only the very first call pays TCP/TLS
    handshake; subsequent calls reuse the warm connection. Headers are
    refreshed each call in case the API key rotates (cheap).
    """
    global _shared_session
    with _session_lock:
        if _shared_session is None:
            import requests
            from requests.adapters import HTTPAdapter

            s = requests.Session()
            adapter = HTTPAdapter(pool_maxsize=20, pool_connections=20)
            s.mount("https://", adapter)
            _shared_session = s
        _shared_session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )
    return _shared_session


def summarize(
    *,
    prompt: str,
    system: str,
    chain: list[dict] | None = None,
    api_key: str | None = None,
    timeout: float = 3.0,
    reasoning_effort: str = "low",
) -> dict:
    """Synchronous OpenRouter chain walker.

    Args:
        prompt:           User-message body.
        system:           System-message body (caller-owned prompt).
        chain:            Tier list — each ``{"model": str,
                          "provider_order": [str]}``. Defaults to
                          :data:`DEFAULT_FALLBACK_CHAIN`.
        api_key:          OpenRouter API key. Defaults to env
                          ``OPENROUTER_API_KEY``.
        timeout:          Per-tier wall-clock budget in seconds (kernel
                          SO_TIMEOUT). Default 3 s — fail fast and walk
                          the chain.
        reasoning_effort: Forwarded to OpenRouter.

    Returns:
        ``{"text": str, "raw_text": str, "cost_usd": float, "model": str,
        "error": str}`` — ``text`` is post-:func:`_scrub_plain_text`;
        ``model`` is the tier that produced the output (empty if every
        tier failed); ``error`` is empty on success or carries the
        joined per-tier error chain when every tier failed (lets callers
        distinguish total failure from a tier returning empty content).
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return {
            "text": "",
            "raw_text": "",
            "cost_usd": 0.0,
            "model": "",
            "error": "missing OPENROUTER_API_KEY",
        }

    chain = chain or DEFAULT_FALLBACK_CHAIN

    log = logger.bind(module="summarize")
    session = _get_shared_session(api_key)

    # Accumulate per-tier failures and only emit ONE warning if every
    # tier in the chain fails. Per-tier noise (e.g. tier1 Groq 429 →
    # tier2 catches it) was a 1:1 spam source on every dashboard event.
    tier_errors: list[str] = []

    for tier_idx, spec in enumerate(chain):
        model = spec["model"]
        order = list(spec["provider_order"])
        tier_label = f"tier{tier_idx + 1} {model}/{order[0]}"
        # Per-tier truncation: clamp the user prompt to this tier's
        # context budget minus the system+completion headroom. Keeps
        # the most recent (tail) content — the same drop-from-top
        # policy the interim-summary formatter uses, applied uniformly
        # at the chain layer.
        max_ctx = int(spec.get("max_context_tokens") or 0)
        if max_ctx > 0:
            budget_chars = max(
                0,
                (max_ctx - PROMPT_OVERHEAD_TOKENS) * CHARS_PER_TOKEN,
            )
            tier_prompt = prompt[-budget_chars:] if len(prompt) > budget_chars else prompt
        else:
            tier_prompt = prompt
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": tier_prompt},
            ],
            "provider": {"order": order, "allow_fallbacks": False},
            "reasoning_effort": reasoning_effort,
            "usage": {"include": True},
        }
        try:
            r = session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                timeout=timeout,
            )
        except Exception as e:
            tier_errors.append(f"{tier_label}: {_summarize_provider_error(e)}")
            continue

        if r.status_code != 200:
            tier_errors.append(
                f"{tier_label}: "
                f"{_summarize_provider_error(Exception(f'OpenRouter API error {r.status_code}: {r.text[:200]}'))}"
            )
            continue

        try:
            data = r.json()
        except Exception:
            data = {}

        if "error" in data:
            err_info = data.get("error", {})
            tier_errors.append(f"{tier_label}: upstream error: {err_info.get('code', '?')}")
            continue

        try:
            raw_text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raw_text = ""

        usage_obj = data.get("usage") or {}
        cost_usd = float(usage_obj.get("total_cost", 0) or 0)

        if raw_text:
            return {
                "text": _scrub_plain_text(raw_text),
                "raw_text": raw_text,
                "cost_usd": cost_usd,
                "model": model,
                "error": "",
            }
        tier_errors.append(f"{tier_label}: empty result")

    error_chain = " | ".join(tier_errors)
    if tier_errors:
        log.warning(f"summarize: all {len(tier_errors)} tier(s) failed — {error_chain}")

    # Empty text + populated ``error`` lets callers distinguish total
    # chain failure from a model that legitimately had nothing to say.
    return {
        "text": "",
        "raw_text": "",
        "cost_usd": 0.0,
        "model": "",
        "error": error_chain,
    }


__all__ = [
    "CHARS_PER_TOKEN",
    "DEFAULT_FALLBACK_CHAIN",
    "PROMPT_OVERHEAD_TOKENS",
    "summarize",
]
