"""OAuth authentication flow — get auth code and exchange for token."""

from __future__ import annotations

import asyncio
import json
import re
import time
import traceback
import uuid
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

from loguru import logger

from aii_lib.utils.tmux import (
    capture_pane,
    kill_session,
    launch_in_tmux,
    send_keys,
    send_text,
    wait_for_text,
)

from .token_utils import check_oauth_token_expired

if TYPE_CHECKING:
    from pathlib import Path


def _extract_oauth_url(pane_output: str) -> str | None:
    """Extract Claude's OAuth URL from a captured tmux pane.

    The CLI prints the URL split across lines (terminal wrap), so collapse
    newlines first. Match URL-safe chars only — bracketed-paste artifacts
    like "Paste" can append themselves on copy and need stripping.
    """
    joined = re.sub(r"\n\s*", "", pane_output)
    urls = re.findall(
        r"https://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+/oauth/authorize"
        r"[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+",
        joined,
    )
    if not urls:
        return None
    return re.sub(r"(Paste|\\e\[\d+~)$", "", urls[0])


def get_oauth_auth_code(
    oauth_url: str,
    web_session_path: Path,
) -> str | None:
    """Complete OAuth consent using web session cookies (Layer 1) → return auth code.

    Opens the OAuth URL in a browser loaded with claude.ai web session cookies,
    clicks Authorize, and extracts the code#state from the callback redirect.
    Uses Xvfb virtual display + headless=False to bypass bot detection.
    """
    from ._browser_login import ensure_display, restore_display

    logger.info("Launching browser for OAuth consent (using web session cookies)...")
    xvfb_proc, saved_env = ensure_display()
    try:
        import nodriver

        return nodriver.loop().run_until_complete(
            _get_oauth_auth_code_inner(oauth_url, web_session_path)
        )
    finally:
        if xvfb_proc:
            xvfb_proc.terminate()
            xvfb_proc.wait()
            logger.info("Stopped Xvfb (OAuth)")
        if saved_env:
            restore_display(saved_env)


async def _get_oauth_auth_code_inner(
    oauth_url: str,
    web_session_path: Path,
) -> str | None:
    """Open OAuth URL in browser with web session cookies → extract auth code.

    Uses nodriver (async CDP) to bypass Cloudflare challenge_redirect.
    """
    from ._browser_login import load_cookies, start_browser, try_cf_verify

    browser = None
    tab = None
    try:
        browser = await start_browser()
        await asyncio.sleep(10)  # Wait for CDP port to be fully ready

        # nodriver 0.48 race: browser.get() does next(filter(type_=='page', targets))
        # which raises StopIteration → PEP 479 → "RuntimeError: coroutine raised
        # StopIteration" if chromium hasn't registered a page target yet. Wait for
        # at least one page target, and retry the first navigation.
        for _ in range(60):
            if any(getattr(t, "type_", None) == "page" for t in browser.targets):
                break
            await asyncio.sleep(0.5)
        else:
            raise RuntimeError("chromium never registered a page target")

        logger.info("Loading web session cookies into browser...")
        for nav_attempt in range(3):
            try:
                tab = await browser.get("https://claude.ai")
                break
            except RuntimeError as nav_e:
                msg = str(nav_e)
                if "StopIteration" in msg or "coroutine raised" in msg:
                    logger.warning(f"nodriver navigation flake — retry {nav_attempt + 1}/3")
                    await asyncio.sleep(2)
                    continue
                raise
        else:
            raise RuntimeError("browser.get retries exhausted (nodriver StopIteration race)")
        await asyncio.sleep(2)

        n_loaded = await load_cookies(tab, web_session_path)
        if n_loaded == 0:
            logger.error("No cookies loaded — web session file may be empty")
            return None

        # Navigate to OAuth URL
        logger.info("Navigating to OAuth URL (nodriver)...")
        tab = await browser.get(oauth_url)
        await asyncio.sleep(5)

        # Wait for Cloudflare to resolve (up to 60s)
        for i in range(30):
            url = tab.target.url
            page_text = await tab.get_content()
            if "challenge_redirect" in url or "Verify you are human" in page_text:
                if i % 5 == 0:
                    logger.info(f"Cloudflare challenge (OAuth) — waiting... ({i * 2}s)")
                await try_cf_verify(tab)
                await asyncio.sleep(2)
            else:
                logger.info(f"Cloudflare passed (OAuth, {i * 2}s)")
                break
        else:
            logger.error("Cloudflare challenge did not resolve (OAuth)")
            await tab.save_screenshot("/tmp/oauth_cloudflare_timeout.png")
            return None

        await asyncio.sleep(2)
        url = tab.target.url
        logger.info(f"OAuth page: {url[:80]}...")

        # Check for immediate redirect to callback
        auth_code = _extract_code_from_url(url)
        if auth_code:
            logger.success("Got code from redirect (OAuth)")
            return auth_code

        # Login page = web session expired
        # Check URL path (not full URL — query params may contain "oauth")
        url_path = urlparse(url).path.lower()
        if "/login" in url_path:
            logger.warning("Web session cookies expired — redirected to login")
            await tab.save_screenshot("/tmp/oauth_login_redirect.png")
            raise RuntimeError("Web session cookies expired")

        # Look for Authorize button (search buttons, links, role=button)
        auth_btn = None
        for i in range(20):
            url = tab.target.url
            auth_code = _extract_code_from_url(url)
            if auth_code:
                logger.success("Got code from auto-redirect (OAuth)")
                return auth_code

            try:
                # Search <button> elements
                buttons = await tab.select_all("button")
                for btn in buttons:
                    btn_text = (btn.text or "").strip().lower()
                    if btn_text and any(kw in btn_text for kw in ["authorize", "allow", "grant"]):
                        auth_btn = btn
                        break

                # Fallback: search <a> and [role=button] elements
                if not auth_btn:
                    clickables = await tab.select_all("a, [role='button']")
                    for el in clickables:
                        el_text = (el.text or "").strip().lower()
                        if el_text and any(kw in el_text for kw in ["authorize", "allow", "grant"]):
                            auth_btn = el
                            break

                if i == 5 and not auth_btn:
                    # Log page state for debugging (only once)
                    btn_texts = [
                        (btn.text or "").strip()[:30] for btn in buttons if (btn.text or "").strip()
                    ]
                    logger.warning(f"Authorize button not found after 5s — buttons: {btn_texts}")
                    try:
                        await tab.save_screenshot("/tmp/oauth_no_authorize.png")
                    except Exception:
                        pass
            except Exception as e:
                if i == 0:
                    logger.warning(f"Button search error: {e}")

            if auth_btn:
                break
            await asyncio.sleep(1)

        if auth_btn:
            # Anthropic's OAuth page renders Authorize disabled until the
            # page receives a real mouse interaction. Use CDP mouse events
            # at the button's coordinates — works on both Xvfb and real displays.
            # First click activates the page, second click authorizes.
            coords = await tab.evaluate("""
                (() => {
                    const btns = document.querySelectorAll('button');
                    for (const b of btns) {
                        if (/authorize/i.test(b.textContent.trim())) {
                            const r = b.getBoundingClientRect();
                            return JSON.stringify({x: Math.round(r.x + r.width/2), y: Math.round(r.y + r.height/2)});
                        }
                    }
                    return null;
                })()
            """)
            if not coords:
                logger.error("Could not get Authorize button coordinates")
                await tab.save_screenshot("/tmp/oauth_no_coords.png")
                raise RuntimeError("Authorize button coordinates not found")

            import json as _json
            import random

            pos = _json.loads(coords)
            cx, cy = pos["x"], pos["y"]
            logger.info(f"Found Authorize button at ({cx}, {cy}), clicking to activate...")

            GET_BTN_COORDS_JS = """
                (() => {
                    const btns = document.querySelectorAll('button');
                    for (const b of btns) {
                        if (/authorize/i.test(b.textContent.trim())) {
                            const r = b.getBoundingClientRect();
                            return JSON.stringify({x: Math.round(r.x + r.width/2), y: Math.round(r.y + r.height/2)});
                        }
                    }
                    return null;
                })()
            """

            from nodriver.cdp import input_ as cdp_input

            for attempt in range(3):
                # Re-query coords each time (button may shift after activation)
                fresh = await tab.evaluate(GET_BTN_COORDS_JS)
                if fresh:
                    pos = _json.loads(fresh)
                    cx, cy = pos["x"], pos["y"]

                x = cx + random.randint(-20, 20)
                y = cy + random.randint(-5, 5)
                await tab.send(
                    cdp_input.dispatch_mouse_event(
                        type_="mousePressed",
                        x=x,
                        y=y,
                        button=cdp_input.MouseButton("left"),
                        click_count=1,
                    )
                )
                await tab.send(
                    cdp_input.dispatch_mouse_event(
                        type_="mouseReleased",
                        x=x,
                        y=y,
                        button=cdp_input.MouseButton("left"),
                        click_count=1,
                    )
                )
                delay = random.uniform(0.5, 1.5)
                if attempt < 2:
                    logger.info(f"Click {attempt + 1} done, waiting {delay:.1f}s...")
                    await asyncio.sleep(delay)

                    # Check if redirect happened already
                    url = tab.target.url
                    auth_code = _extract_code_from_url(url)
                    if auth_code:
                        logger.success("Got auth code (OAuth)")
                        return auth_code

            await asyncio.sleep(5)

            url = tab.target.url
            auth_code = _extract_code_from_url(url)
            if auth_code:
                logger.success("Got auth code (OAuth)")
                return auth_code

            # Wait longer for redirect
            for _ in range(15):
                url = tab.target.url
                auth_code = _extract_code_from_url(url)
                if auth_code:
                    logger.success("Got auth code (OAuth, delayed)")
                    return auth_code
                await asyncio.sleep(1)

        await tab.save_screenshot("/tmp/oauth_debug.png")
        logger.error(f"OAuth failed — on: {tab.target.url}")
        raise RuntimeError(f"Unexpected page: {tab.target.url}")

    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"OAuth error: {e}\n{traceback.format_exc()}")
        if browser and tab is not None:
            try:
                await tab.save_screenshot("/tmp/oauth_error.png")
            except Exception:
                pass
        return None
    finally:
        if browser:
            try:
                browser.stop()
            except Exception:
                pass


def _extract_code_from_url(url: str) -> str | None:
    """Extract code#state from callback URL."""
    if "oauth/code/callback" not in url:
        return None
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    code = params.get("code", [None])[0]
    state = params.get("state", [None])[0]
    if code and state:
        return f"{code}#{state}"
    return None


# ---------------------------------------------------------------------------
# OAuth flow (Layer 2): obtain Claude Code token via TUI + browser
# ---------------------------------------------------------------------------


def run_oauth_flow(web_session_path: Path) -> bool:
    """
    Full OAuth flow: obtain a new Claude Code token (Layer 2).

    Requires valid web session cookies (Layer 1) in web_session_path.
    Starts the `claude` CLI in a tmux session, navigates onboarding,
    extracts the OAuth URL, completes consent in browser using web session
    cookies, and feeds the auth code back into the TUI.
    """
    session = f"claude_login_{uuid.uuid4().hex[:8]}"

    try:
        # 1. Start claude in tmux.
        # IS_SANDBOX=1 bypasses Claude CLI's root-user refusal of
        # bypassPermissions (see binary: process.env.IS_SANDBOX !== "1").
        logger.info(f"Starting claude in tmux session: {session}")
        launch_in_tmux(
            session=session,
            cmd="claude",
            extra_env={"IS_SANDBOX": "1"},
        )

        # 2. Wait for initial UI to appear
        #    Two paths:
        #    a) First-ever run: theme selector → login method → OAuth URL
        #    b) Already onboarded but logged out: main TUI with "Not logged in"
        logger.info("Waiting for UI to load...")
        time.sleep(5)
        output = capture_pane(session)

        # Path A: First-run onboarding — advance through known screens.
        # Most use Enter; the new-MCP-server prompt is dismissed with Esc
        # because Enter would enable an MCP we didn't ask for during login.
        onboarding_screens: list[tuple[str, str]] = [
            ("Dark mode", "Enter"),
            ("Light mode", "Enter"),  # color theme
            ("Syntax theme", "Enter"),
            ("Monokai", "Enter"),  # syntax theme
            ("trust this folder", "Enter"),  # workspace trust
            ("security notice", "Enter"),
            ("run code", "Enter"),  # security warning
            ("New MCP server found", "Escape"),  # MCP prompt — skip
        ]
        for _step in range(6):
            matched = next(((kw, key) for kw, key in onboarding_screens if kw in output), None)
            if matched is None:
                break
            kw, key = matched
            logger.info(f"Navigating onboarding: '{kw}' → {key}")
            send_keys(session, key)
            time.sleep(3)
            output = capture_pane(session)

        # Path C: CLI landed on main TUI — might mean token was auto-refreshed.
        # IMPORTANT: The CLI can land on the TUI even with an expired token if
        # web session cookies are valid. We must verify the token is actually fresh.
        # Markers cover multiple CLI versions: "for shortcuts" (v2.1.x footer),
        # "bypass permissions" / "Try " (older versions).
        sent_login = False
        main_tui_markers = ("for shortcuts", "bypass permissions", "Try ")
        if (
            any(m in output for m in main_tui_markers)
            and "Not logged in" not in output
            and "Select login method" not in output
            and "Dark mode" not in output
        ):
            if not check_oauth_token_expired():
                logger.success("CLI landed on main TUI — token confirmed valid")
                return True
            # Info level (was warning): the /login retry below is the
            # designed recovery path — this branch is the entry condition
            # for it, not a failure. Web session cookies routinely outlast
            # the bearer token.
            logger.info(
                "CLI landed on main TUI but token is still expired — "
                "web session allowed TUI load without refreshing token"
            )
            # Send /login to trigger re-auth from the TUI
            send_text(session, "/login")
            time.sleep(0.5)
            send_keys(session, "Enter")
            time.sleep(3)
            output = capture_pane(session)
            sent_login = True

        if not sent_login and ("Not logged in" in output or "/login" in output):
            # Path B: Already onboarded, but logged out — trigger /login.
            # Skipped if Path C just sent /login: after Path C, the pane shows the
            # literal "/login" text we just typed, which would re-trigger this
            # branch and send /login a second time (lands on the wrong screen).
            logger.info("Detected 'Not logged in' — sending /login command...")
            send_text(session, "/login")
            time.sleep(0.5)
            send_keys(session, "Enter")
            time.sleep(3)
            output = capture_pane(session)

        # 3. Wait for login method selector
        if "Select login method" not in output:
            logger.info("Waiting for login selector...")
            output = wait_for_text(session, "Select login method", timeout=10)
        if output and "Select login method" in output:
            logger.info("Selecting Claude subscription...")
            send_keys(session, "Enter")
        else:
            logger.warning("No login selector — checking for OAuth URL...")

        # 4. Wait for "Paste code here" prompt (Ink TUI input)
        logger.info("Waiting for paste prompt...")
        output = wait_for_text(session, "Paste code here", timeout=15)
        if not output:
            # Fallback: check if URL is visible without paste prompt
            output = capture_pane(session)
            if "oauth/authorize" not in output:
                logger.error("No paste prompt or OAuth URL found")
                logger.error(f"Pane:\n{output[-400:]}")
                return False

        # 5. Extract OAuth URL
        oauth_url = _extract_oauth_url(output)
        if not oauth_url:
            logger.error("Could not extract OAuth URL from pane")
            return False
        logger.info(f"OAuth URL: {oauth_url[:80]}...")

        # 6. Complete OAuth consent in browser (uses web session cookies)
        auth_code = get_oauth_auth_code(oauth_url, web_session_path)
        if not auth_code:
            logger.error("Could not get auth code from browser")
            return False
        logger.info(f"Auth code: {auth_code[:30]}...")

        # 7. Type code into TUI via tmux send-keys -l (literal)
        logger.info("Typing auth code into TUI...")
        send_text(session, auth_code)
        time.sleep(0.5)
        send_keys(session, "Enter")

        # 8. Wait for OAuth token creation
        logger.info("Waiting for OAuth to complete...")
        output = wait_for_text(session, "Login successful", timeout=20)
        if output and "Login successful" in output:
            logger.success("OAuth token created")
            # Navigate through ALL remaining onboarding prompts until main TUI.
            # After login there can be multiple screens:
            #   - "Login successful. Press Enter to continue"
            #   - Security notice about running code
            #   - Workspace trust dialog
            #   - Effort level selector
            #   - Possibly others added in future versions
            # Press Enter through all of them so .claude.json state is persisted
            # and subsequent launches skip onboarding.
            onboarding_keywords = [
                "Press Enter",
                "trust this folder",
                "Enter to confirm",
                "Select",
                "effort",
                "Use high",
                "Use medium",
            ]
            for _i in range(15):
                send_keys(session, "Enter")
                time.sleep(3)
                output = capture_pane(session)
                # Check if any onboarding keywords remain
                still_onboarding = any(kw in output for kw in onboarding_keywords)
                if not still_onboarding:
                    logger.info("Reached main TUI prompt")
                    break
            # Give the CLI time to write .claude.json state
            time.sleep(5)
            return True

        # Check if credentials were created even without success message.
        # Must verify the token is actually FRESH (expiresAt > now) — a stale
        # accessToken from a prior session is not success. Without this check,
        # a TUI timing race (CLI didn't print "Login successful" in 20s)
        # promotes to false-success, then the post-OAuth verify fails, looping
        # forever.
        from aii_lib.llm_backend.claude_max import aii_claude_dir

        creds_path = aii_claude_dir() / ".credentials.json"
        if creds_path.exists():
            creds = json.loads(creds_path.read_text())
            oa = creds.get("claudeAiOauth", {})
            expires_at_ms = oa.get("expiresAt", 0)
            now_ms = int(time.time() * 1000)
            if oa.get("accessToken") and expires_at_ms > now_ms:
                logger.success("Credentials file created")
                return True
            if oa.get("accessToken"):
                logger.warning(
                    f"Credentials file present but token expired "
                    f"({(now_ms - expires_at_ms) / 1000:.0f}s ago) — TUI did not "
                    f"complete OAuth (real failure, will fall through to error path)"
                )

        logger.error("Login did not complete successfully")
        output = capture_pane(session)
        logger.error(f"Final pane:\n{output[-400:]}")
        return False

    except RuntimeError as e:
        if "Web session cookies expired" in str(e):
            raise  # Propagate so ensure_oauth_token can refresh web session
        logger.error(f"OAuth flow failed: {e}")
        return False
    except Exception as e:
        logger.error(f"OAuth flow failed: {e}\n{traceback.format_exc()}")
        return False
    finally:
        kill_session(session)


# ---------------------------------------------------------------------------
# Dependency management
# ---------------------------------------------------------------------------
