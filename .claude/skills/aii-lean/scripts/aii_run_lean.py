#!/usr/bin/env python
"""
Lean 4 Code Runner

Compiles and verifies Lean 4 code using lean-interact library.
Mathlib is always enabled. Each request gets a fresh LeanServer (no memory accumulation).
When code contains sorry, returns goal states at each sorry position.

Usage:
    python aii_run_lean.py proof.lean
    echo "theorem test : 1 + 1 = 2 := rfl" | python aii_run_lean.py -
"""

import argparse
import sys
from pathlib import Path

from aii_lib.abilities.aii_ability import aii_ability

SERVER_NAME = "aii_lean__run"
MATHLIB_LEAN_VERSION = "v4.14.0"
DEFAULT_TIMEOUT = 120.0

# Cached config (reused across requests, lightweight)
_config = None


# =============================================================================
# Core Logic (used by server handler)
# =============================================================================


def init_run_lean():
    """Initialize Lean environment - setup PATH, warm up disk cache."""
    import fcntl
    import os

    global _config

    # Add elan/lake to PATH
    elan_bin = Path.home() / ".elan" / "bin"
    if elan_bin.exists() and str(elan_bin) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{elan_bin}:{os.environ.get('PATH', '')}"

    # Create config (downloads/builds Mathlib on first run, then cached on disk)
    # File lock prevents parallel workers from racing on the same REPL build cache
    from lean_interact import Command, LeanREPLConfig, LeanServer, TempRequireProject

    lock_path = Path("/tmp/lean_repl_build.lock")
    lock_path.touch(exist_ok=True)
    with open(lock_path) as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        project = TempRequireProject(lean_version=MATHLIB_LEAN_VERSION, require="mathlib")
        _config = LeanREPLConfig(project=project, verbose=False)

    # Warmup: populate disk cache with a temp server
    warmup_server = LeanServer(_config)
    warmup_server.run(Command(cmd="#check Nat"))
    warmup_server.run(Command(cmd="import Mathlib.Tactic\nexample : 1 + 1 = 2 := by ring"))
    warmup_server.kill()


@aii_ability(
    name="aii_lean__run",
    description="Compile and verify Lean 4 code with Mathlib support.",
    venv="../../.ability_client_venv",
    requirements="server_requirements.txt",
    worker_init="init_run_lean",
    max_workers=4,
    timeout=120.0,
    check_env="check_env.sh",
)
def core_run_lean(code: str = "") -> dict:
    """
    Run Lean 4 code and return compilation results.

    Creates a fresh LeanServer for each request - no memory accumulation.
    When code has sorry placeholders, returns goal states at each sorry position.

    Args:
        code: Lean 4 code to compile

    Returns:
        Dict with:
            - success: bool - tool ran without exceptions
            - verified: bool - proof compiled without errors/sorries
            - errors: list[str] - error messages
            - warnings: list[str] - warnings
            - infos: list[str] - info messages
            - has_sorries: bool - code contains sorry
            - sorry_goals: list[dict] - goal state at each sorry position
              Each dict has: proof_state (int), goal (str if available)
    """
    global _config

    if not code.strip():
        return {"success": False, "verified": False, "error": "No code provided"}

    try:
        from lean_interact import Command, LeanServer

        # Fresh server for each request - no memory accumulation
        server = LeanServer(_config)
        try:
            response = server.run(Command(cmd=code))

            errors, warnings, infos = [], [], []
            for msg in response.messages:
                severity = getattr(msg, "severity", "info")
                data = getattr(msg, "data", str(msg))
                if severity == "error":
                    errors.append(data)
                elif severity == "warning":
                    warnings.append(data)
                else:
                    infos.append(data)

            has_sorries = bool(response.sorries) if hasattr(response, "sorries") else False

            # Extract goal states from sorry positions
            sorry_goals = []
            if has_sorries:
                for i, sorry in enumerate(response.sorries):
                    goal_info = {"sorry_index": i}
                    if hasattr(sorry, "proof_state"):
                        goal_info["proof_state"] = sorry.proof_state
                    if hasattr(sorry, "goal"):
                        goal_info["goal"] = sorry.goal
                    elif hasattr(sorry, "goals"):
                        goal_info["goal"] = sorry.goals
                    sorry_goals.append(goal_info)

            verified = len(errors) == 0 and not has_sorries

            return {
                "success": True,
                "verified": verified,
                "has_sorries": has_sorries,
                "sorry_goals": sorry_goals,
                "errors": errors,
                "warnings": warnings,
                "infos": infos,
            }
        finally:
            server.kill()

    except ImportError as e:
        return {"success": False, "verified": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "verified": False, "error": str(e)}


# =============================================================================
# CLI
# =============================================================================


def main():
    import json

    parser = argparse.ArgumentParser(description="Compile and verify Lean 4 code (with Mathlib)")
    parser.add_argument("file", help="Lean file to verify, or '-' for stdin")
    args = parser.parse_args()

    if args.file == "-":
        code = sys.stdin.read()
    else:
        file_path = Path(args.file)
        if not file_path.exists():
            print(
                json.dumps(
                    {"success": False, "error": f"File not found: {args.file}"},
                    indent=2,
                )
            )
            sys.exit(1)
        code = file_path.read_text()

    from aii_lib.abilities.ability_server import call_server

    result = call_server(
        SERVER_NAME,
        {
            "code": code,
        },
        timeout=DEFAULT_TIMEOUT,
    )

    if result is None:
        print(json.dumps({"success": False, "error": "Ability service not available."}, indent=2))
        sys.exit(1)

    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("verified", False) else 1)


if __name__ == "__main__":
    main()
