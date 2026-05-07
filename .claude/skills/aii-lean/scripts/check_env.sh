#!/bin/bash
# Check Lean 4 prerequisites: elan, lake, lean
set -euo pipefail

ERRORS=0

# elan (Lean toolchain manager)
if ! command -v elan &>/dev/null; then
    if [ -x "$HOME/.elan/bin/elan" ]; then
        export PATH="$HOME/.elan/bin:$PATH"
    fi
fi

if ! command -v elan &>/dev/null; then
    echo "elan not found (install: curl https://elan.lean-lang.org/install.sh -sSf | sh)" >&2
    ERRORS=$((ERRORS + 1))
fi

# lake (Lean build tool, comes with elan)
if ! command -v lake &>/dev/null; then
    if [ -x "$HOME/.elan/bin/lake" ]; then
        export PATH="$HOME/.elan/bin:$PATH"
    fi
fi

if ! command -v lake &>/dev/null; then
    echo "lake not found (should come with elan)" >&2
    ERRORS=$((ERRORS + 1))
fi

# lean binary
if ! command -v lean &>/dev/null; then
    if [ -x "$HOME/.elan/bin/lean" ]; then
        export PATH="$HOME/.elan/bin:$PATH"
    fi
fi

if ! command -v lean &>/dev/null; then
    echo "lean not found (should come with elan)" >&2
    ERRORS=$((ERRORS + 1))
fi

# LEANEXPLORE_API_KEY (optional but needed for semantic search)
if [ -z "${LEANEXPLORE_API_KEY:-}" ]; then
    # Try loading from .env
    PROJECT_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
    if [ -f "$PROJECT_ROOT/.env" ]; then
        LEANEXPLORE_API_KEY=$(grep -E '^LEANEXPLORE_API_KEY=' "$PROJECT_ROOT/.env" 2>/dev/null | cut -d= -f2- | tr -d '"'"'" || true)
    fi
    if [ -z "${LEANEXPLORE_API_KEY:-}" ]; then
        echo "LEANEXPLORE_API_KEY not set (semantic search will fail)" >&2
        ERRORS=$((ERRORS + 1))
    fi
fi

exit $ERRORS
