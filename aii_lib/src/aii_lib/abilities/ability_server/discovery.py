"""Skill discovery, venv setup, and environment checks."""

import importlib
import importlib.util
import inspect
import subprocess
import sys
import time
import traceback
from pathlib import Path

from loguru import logger

from aii_lib.abilities.aii_ability import get_registry

# Server config — deep-merge ``abilities.yaml`` with optional ``.private`` sibling
from aii_lib.utils.config_overrides import load_config_with_overrides as _load_config

# Project root is 5 levels up from this file
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
SKILLS_DIR = _PROJECT_ROOT / ".claude" / "skills"

_CONFIG_FILE = _PROJECT_ROOT / "aii_config" / "server" / "abilities.yaml"
_server_config: dict = _load_config(_CONFIG_FILE) if _CONFIG_FILE.exists() else {}


def _discover_skill_dirs() -> list[Path]:
    """Find all .claude/skills/*/scripts/ directories."""
    dirs = []
    if not SKILLS_DIR.is_dir():
        logger.bind(source="server").warning(f"Skills directory not found: {SKILLS_DIR}")
        return dirs

    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.name.startswith("aii-"):
            continue
        scripts_dir = skill_dir / "scripts"
        if scripts_dir.is_dir():
            dirs.append(scripts_dir)

    return dirs


def _add_to_sys_path(dirs: list[Path]) -> None:
    """Add all script directories to sys.path (needed for sibling imports)."""
    for d in dirs:
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)


def _import_scripts(dirs: list[Path]) -> int:
    """Import all .py scripts from discovered directories.

    Returns count of imported modules.
    """
    log = logger.bind(source="server")
    count = 0

    for scripts_dir in dirs:
        for py_file in sorted(scripts_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue

            module_name = py_file.stem

            # Skip if already imported (sibling imports may have triggered it)
            if module_name in sys.modules:
                count += 1
                continue

            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                count += 1
            except Exception as e:
                log.exception(f"Failed to import {py_file}: {e}\n{traceback.format_exc()}")

    return count


# =============================================================================
# Venv-ensure phase
# =============================================================================


def _ensure_venvs() -> None:
    """Create shared venvs and install aii_lib[ability-client] + requirements.

    Scans the registry for (venv, requirements) metadata, groups by venv path,
    creates the venv if needed, installs aii_lib[ability-client] (thin HTTP
    client deps), then installs any per-skill server_requirements.txt files.
    """
    log = logger.bind(source="venv")

    registry = get_registry()
    if not registry:
        return

    # Collect (venv_abs_path → set of requirements_abs_paths) from registry
    venv_reqs: dict[Path, set[Path]] = {}

    for meta in registry.values():
        venv_rel = meta.get("venv")
        if not venv_rel:
            continue

        # Resolve paths relative to the script that registered the tool
        func = meta.get("func")
        if func is None:
            continue
        try:
            script_path = Path(inspect.getfile(func)).resolve().parent
        except (TypeError, OSError):
            continue

        venv_path = (script_path / venv_rel).resolve()

        if venv_path not in venv_reqs:
            venv_reqs[venv_path] = set()

        # requirements are optional — ability-client covers the basics
        req_rel = meta.get("requirements")
        if req_rel:
            if isinstance(req_rel, str):
                req_paths = [(script_path / req_rel).resolve()]
            elif isinstance(req_rel, list):
                req_paths = [(script_path / r).resolve() for r in req_rel]
            else:
                req_paths = []
            for rp in req_paths:
                if rp.exists():
                    venv_reqs[venv_path].add(rp)

    if not venv_reqs:
        return

    # Find aii_lib root (contains pyproject.toml with [ability-client] extra)
    aii_lib_root = Path(__file__).resolve().parents[4]  # .../aii_lib/
    aii_lib_spec = f"{aii_lib_root}[ability-client]"

    for venv_path, req_files in venv_reqs.items():
        python_bin = venv_path / "bin" / "python"

        # Create venv if it doesn't exist
        if not python_bin.exists():
            log.info(f"Creating venv: {venv_path}")
            try:
                subprocess.run(
                    # ``--seed`` installs pip + setuptools + wheel inside the
                    # venv so ``<venv>/bin/pip install X`` works directly.
                    # Agents (and most skill docs) reflexively reach for
                    # ``pip``; without seeding, that's a 127 + retry loop.
                    ["uv", "venv", str(venv_path), "--python=3.12", "--seed"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                log.info(f"Created venv: {venv_path}")
            except FileNotFoundError:
                log.warning(
                    "uv not found — skipping venv creation (install with: curl -LsSf https://astral.sh/uv/install.sh | sh)"
                )
                return
            except subprocess.CalledProcessError as e:
                log.exception(f"Failed to create venv {venv_path}: {e.stderr}")
                return
            except subprocess.TimeoutExpired:
                log.exception(f"Timeout creating venv {venv_path}")
                return

        # Install aii_lib[ability-client] + any per-skill requirements
        cmd = ["uv", "pip", "install", f"--python={python_bin}", "-e", aii_lib_spec]
        for rf in sorted(req_files):
            cmd.extend(["-r", str(rf)])

        n_req = len(req_files)
        log.info(f"Installing aii_lib[ability-client] + {n_req} requirements into {venv_path.name}")
        start = time.perf_counter()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            elapsed = time.perf_counter() - start

            if result.returncode == 0:
                log.info(f"Deps installed into {venv_path.name} ({elapsed:.1f}s)")
            else:
                log.error(f"Failed to install deps into {venv_path.name}: {result.stderr[:500]}")
        except subprocess.TimeoutExpired:
            log.exception(f"Timeout installing deps into {venv_path.name} (300s)")
        except Exception as e:
            log.exception(
                f"Error installing deps into {venv_path.name}: {e}\n{traceback.format_exc()}"
            )


# =============================================================================
# Environment checks
# =============================================================================


def _run_env_checks() -> dict[str, str]:
    """Run check_env scripts for tools that declare them.

    Each check_env is a bash script (path relative to the tool's script dir)
    that exits 0 if all prerequisites are available, non-zero otherwise.
    Stderr output describes what's missing.

    Returns dict of {tool_name: error_message} for failed checks.
    """
    log = logger.bind(source="env-check")
    registry = get_registry()
    failures: dict[str, str] = {}

    for name, meta in sorted(registry.items()):
        check_env_rel = meta.get("check_env")
        if not check_env_rel:
            continue

        func = meta.get("func")
        if func is None:
            continue

        try:
            script_dir = Path(inspect.getfile(func)).resolve().parent
        except (TypeError, OSError):
            continue

        check_script = (script_dir / check_env_rel).resolve()

        if not check_script.exists():
            log.warning(f"{name}: check_env script not found: {check_script}")
            failures[name] = f"check_env script not found: {check_env_rel}"
            continue

        try:
            result = subprocess.run(
                ["bash", str(check_script)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(script_dir),
            )
            if result.returncode == 0:
                log.debug(f"{name}: env check passed")
            else:
                msg = (
                    result.stderr.strip()
                    or result.stdout.strip()
                    or f"exit code {result.returncode}"
                )
                log.warning(f"{name}: env check FAILED — {msg}")
                failures[name] = msg
        except subprocess.TimeoutExpired:
            log.warning(f"{name}: env check timed out (30s)")
            failures[name] = "check_env timed out (30s)"
        except Exception as e:
            log.warning(f"{name}: env check error — {e}")
            failures[name] = str(e)

    if failures:
        log.warning(f"{len(failures)} env check(s) failed: {', '.join(failures.keys())}")
    else:
        checked = sum(1 for m in registry.values() if m.get("check_env"))
        if checked:
            log.info(f"All {checked} env checks passed")

    return failures


# =============================================================================
# Schema endpoint
# =============================================================================
