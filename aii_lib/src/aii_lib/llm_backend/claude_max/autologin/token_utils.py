"""OAuth token inspection and validation utilities."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback

from loguru import logger


def get_oauth_token_remaining_seconds() -> float:
    """Get remaining seconds until the Claude Code OAuth token expires.

    Reads expiresAt from ~/.claude/.credentials.json (Layer 2).
    Returns 0.0 if token is missing/expired/unreadable.
    """
    from aii_lib.llm_backend.claude_max import aii_claude_dir

    creds_path = aii_claude_dir() / ".credentials.json"
    if not creds_path.exists():
        return 0.0
    try:
        creds = json.loads(creds_path.read_text())
        oauth = creds.get("claudeAiOauth", {})
        expires_at_ms = oauth.get("expiresAt", 0)
        if not expires_at_ms or not oauth.get("accessToken"):
            return 0.0
        now_ms = int(time.time() * 1000)
        remaining_ms = expires_at_ms - now_ms
        return max(0.0, remaining_ms / 1000.0)
    except Exception:
        return 0.0


def check_oauth_token_expired() -> bool:
    """Check if the Claude Code OAuth token (Layer 2) is expired.

    Returns True if token is expired/missing/unreadable (= needs OAuth flow).
    """
    return get_oauth_token_remaining_seconds() <= 0


# ---------------------------------------------------------------------------
# OAuth token (Layer 2) validity check
# ---------------------------------------------------------------------------


def check_oauth_token_valid() -> bool:
    """Check if the Claude Code OAuth token (Layer 2) is valid and working.

    Three checks:
    1. Token expiry (expiresAt in .credentials.json)
    2. `claude auth status --json` (reports loggedIn)
    3. Claude Agent SDK call (2+2 test) to verify token works end-to-end
    """
    if check_oauth_token_expired():
        return False
    try:
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        result = subprocess.run(
            ["claude", "auth", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if data.get("loggedIn"):
                if _verify_oauth_token_works():
                    logger.success(f"OAuth token valid for {data.get('email', 'unknown')}")
                    return True
                logger.warning("OAuth token exists but is invalid/revoked")
                return False
    except Exception:
        pass
    return False


def _verify_oauth_token_works() -> bool:
    """Verify the OAuth token (Layer 2) works via Claude Agent SDK call.

    Spawns a subprocess (same approach as preflight) to avoid event loop
    conflicts. Asks Claude "What is 2+2?" and checks for a non-error response.
    Returns True if token is valid or on any non-auth error (network, etc.).
    """
    import textwrap

    script = textwrap.dedent("""\
        import asyncio, json, os, sys
        os.chdir("/tmp")
        os.environ.pop("CLAUDECODE", None)

        from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, ResultMessage

        async def test():
            opts = ClaudeAgentOptions(
                model="claude-haiku-4-5",
                max_turns=1,
                system_prompt="You are a calculator. Reply with ONLY the number.",
                cwd="/tmp",
            )
            async with ClaudeSDKClient(options=opts) as client:
                async def prompt():
                    yield {"type": "user", "message": {"role": "user", "content": "What is 2+2?"}}
                await client.query(prompt())
                async for msg in client.receive_response():
                    if isinstance(msg, ResultMessage):
                        return {"response": (msg.result or "")[:100], "cost": msg.total_cost_usd or 0.0}
            return {"error": "no ResultMessage"}

        try:
            r = asyncio.run(test())
            print(json.dumps(r))
        except Exception as e:
            print(json.dumps({"error": str(e)[:200]}))
    """)

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        stdout = result.stdout.strip()
        if not stdout:
            logger.warning("Token verify: empty output from SDK test")
            return True  # Don't block on subprocess issues

        last_line = stdout.strip().split("\n")[-1]
        data = json.loads(last_line)

        if "error" in data:
            error_msg = data["error"]
            if "401" in error_msg or "authentication" in error_msg.lower():
                logger.warning(f"Token verify: auth error — {error_msg[:100]}")
                return False
            # Non-auth error (rate limit, network, etc.) — assume token is fine
            logger.warning(f"Token verify: non-auth error — {error_msg[:100]}")
            return True

        response = data.get("response", "").strip()
        if response:
            # SDK may return auth errors as the response content
            resp_lower = response.lower()
            if (
                "not logged in" in resp_lower
                or "401" in resp_lower
                or "authentication" in resp_lower
            ):
                logger.warning(f"Token verify: auth failure in response — {response[:100]}")
                return False
            logger.info(f"Token verify: SDK response={response!r}")
            return True

        return True  # Got a response, token works
    except subprocess.TimeoutExpired:
        logger.warning("Token verify: SDK test timed out")
        return True  # Don't block on timeout
    except Exception as e:
        logger.warning(f"Token verify: {e}\n{traceback.format_exc()}")
        return True  # Don't block on unexpected errors


# ---------------------------------------------------------------------------
# Browser automation
# ---------------------------------------------------------------------------
