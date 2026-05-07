"""Shared helpers for browser login — nodriver startup, Xvfb display, cookies."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# nodriver browser startup
# ---------------------------------------------------------------------------


async def start_browser() -> object:
    """Start a nodriver browser with standard anti-detection args.

    Returns the Browser instance. Caller must call browser.stop() when done.
    """
    import nodriver

    browser = await nodriver.start(
        headless=False,
        browser_args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ],
    )
    logger.info("nodriver browser started")
    return browser


# ---------------------------------------------------------------------------
# Cloudflare challenge helper
# ---------------------------------------------------------------------------


async def try_cf_verify(tab: object) -> bool:
    """Attempt nodriver's built-in Cloudflare checkbox solver.

    Uses OpenCV template matching to find and click the CF checkbox.
    Returns True if verification was attempted, False if no challenge or error.
    Silently returns False if no checkbox is present — caller continues normally.
    """
    try:
        await tab.verify_cf()
        logger.info("Cloudflare checkbox clicked via verify_cf()")
        return True
    except Exception:
        # No checkbox found, or opencv not available — not an error
        return False


# ---------------------------------------------------------------------------
# Xvfb virtual display management
# ---------------------------------------------------------------------------


def _kill_stale_display_processes() -> None:
    """Kill stale Xvfb and Chrome/Chromium processes from previous runs.

    Called before starting a new Xvfb to avoid 'display :99 in use' errors
    and zombie Chrome processes holding dead CDP sockets.
    """
    for proc_name in ("Xvfb", "chrome", "chromium"):
        try:
            subprocess.run(
                ["pkill", "-f", proc_name],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass
    # Remove stale X lock file
    lock_file = Path("/tmp/.X99-lock")
    if lock_file.exists():
        try:
            lock_file.unlink()
        except Exception:
            pass
    time.sleep(0.5)


def ensure_display() -> tuple[subprocess.Popen | None, dict[str, str | None]]:
    """Always use Xvfb virtual display for browser automation.

    Google blocks headless Chromium login. Running with headless=False
    on a virtual display (Xvfb) bypasses detection without popping up
    a visible browser window (even on machines with a real display).

    Kills stale Xvfb/Chrome processes before starting to prevent
    'display :99 in use' and ConnectionRefusedError on zombie CDP sockets.

    Returns (xvfb_process, saved_env) where saved_env holds original
    DISPLAY/WAYLAND_DISPLAY values for restoration after Xvfb is stopped.
    Returns (None, {}) if Xvfb is unavailable.
    """
    import shutil

    if not shutil.which("Xvfb"):
        logger.warning("Xvfb not found — install with: apt-get install xvfb")
        return None, {}

    # Save original display env vars for restoration
    saved_env: dict[str, str | None] = {
        "DISPLAY": os.environ.get("DISPLAY"),
        "WAYLAND_DISPLAY": os.environ.get("WAYLAND_DISPLAY"),
    }

    # Kill stale processes from previous failed runs
    _kill_stale_display_processes()

    # Start Xvfb on display :99
    xvfb = subprocess.Popen(
        ["Xvfb", ":99", "-screen", "0", "1920x1080x24", "-nolisten", "tcp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1)

    if xvfb.poll() is not None:
        logger.warning(
            f"Xvfb failed to start (exit code {xvfb.returncode}) — display :99 may be in use"
        )
        return None, {}

    os.environ["DISPLAY"] = ":99"
    os.environ.pop("WAYLAND_DISPLAY", None)
    logger.info("Started Xvfb virtual display on :99")
    return xvfb, saved_env


def restore_display(saved_env: dict[str, str | None]) -> None:
    """Restore original DISPLAY/WAYLAND_DISPLAY env vars after Xvfb is stopped."""
    for key, value in saved_env.items():
        if value is not None:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# Cookie helpers
# ---------------------------------------------------------------------------


async def save_cookies(
    browser: object, output_path: Path, domain_filter: str = "claude.ai"
) -> bool:
    """Save nodriver cookies in Playwright storage_state format.

    Playwright format: {"cookies": [...], "origins": []}
    CDP returns ALL cookies from ALL domains; we filter to ``domain_filter``
    to match the old Selenium get_cookies() behavior (domain-scoped).

    Returns True on success, False on error.
    """
    import json

    try:
        cookies = await browser.cookies.get_all()
    except Exception as e:
        logger.error(f"Failed to get cookies: {e}")
        return False

    if not cookies:
        logger.error("No cookies obtained from browser")
        return False

    pw_cookies = []
    for c in cookies:
        # CDP returns Cookie objects with attributes, not dicts
        domain = getattr(c, "domain", "")
        if domain_filter and domain_filter not in domain:
            continue
        same_site = getattr(c, "same_site", None)
        if same_site is not None:
            # CDP enum .value gives "Lax"/"Strict"/"None"; str() gives "CookieSameSite.LAX"
            same_site = getattr(same_site, "value", None) or str(same_site)
        pw_cookie = {
            "name": getattr(c, "name", ""),
            "value": getattr(c, "value", ""),
            "domain": domain,
            "path": getattr(c, "path", "/"),
            "expires": getattr(c, "expires", -1),
            "httpOnly": getattr(c, "http_only", False),
            "secure": getattr(c, "secure", False),
            "sameSite": same_site or "Lax",
        }
        pw_cookies.append(pw_cookie)

    if not pw_cookies:
        logger.error(f"No cookies matched domain filter '{domain_filter}' (total: {len(cookies)})")
        return False

    state = {"cookies": pw_cookies, "origins": []}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(state))
    logger.info(
        f"Browser state saved to {output_path} ({len(pw_cookies)} cookies, filtered from {len(cookies)})"
    )
    return True


async def load_cookies(tab: object, cookie_path: Path) -> int:
    """Load cookies from a Playwright storage_state JSON file via CDP on a tab.

    Returns the number of cookies successfully set; 0 if the file is missing
    or empty (caller treats that as a clean "skip this account").
    """
    import json

    from nodriver.cdp import network

    if not cookie_path.exists():
        logger.info(f"No web session at {cookie_path}; skipping cookie load")
        return 0

    state = json.loads(cookie_path.read_text())
    all_cookies = state.get("cookies", [])
    loaded = 0
    failed = 0

    for c in all_cookies:
        domain = c.get("domain", "")
        scheme = "https" if c.get("secure") else "http"
        url = f"{scheme}://{domain.lstrip('.')}"

        # CDP expects TimeSinceEpoch for expires, not raw float
        raw_expires = c.get("expires")
        expires_val = None
        if isinstance(raw_expires, (int, float)) and raw_expires > 0:
            expires_val = network.TimeSinceEpoch(raw_expires)

        # Map sameSite string to CDP enum
        same_site_str = c.get("sameSite", "Lax")
        same_site_val = None
        if same_site_str:
            _ss_map = {"none": "None", "lax": "Lax", "strict": "Strict"}
            mapped = _ss_map.get(same_site_str.lower())
            if mapped:
                try:
                    same_site_val = network.CookieSameSite(mapped)
                except Exception:
                    pass

        try:
            await tab.send(
                network.set_cookie(
                    name=c["name"],
                    value=c["value"],
                    url=url,
                    domain=domain or None,
                    path=c.get("path") or None,
                    secure=bool(c.get("secure")) or None,
                    http_only=bool(c.get("httpOnly")) or None,
                    same_site=same_site_val,
                    expires=expires_val,
                )
            )
            loaded += 1
        except Exception as e:
            failed += 1
            if failed <= 3:
                logger.warning(f"Cookie set failed ({c.get('name', '?')}@{domain}): {e}")
    logger.info(f"Loaded {loaded}/{len(all_cookies)} cookies into browser")
    return loaded
