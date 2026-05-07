#!/bin/bash
# Check OpenRouter prerequisites: OPENROUTER_API_KEY
set -euo pipefail

ERRORS=0
PROJECT_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"

OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
if [ -z "$OPENROUTER_API_KEY" ] && [ -f "$PROJECT_ROOT/.env" ]; then
    OPENROUTER_API_KEY=$(grep -E '^OPENROUTER_API_KEY=' "$PROJECT_ROOT/.env" 2>/dev/null | cut -d= -f2- | tr -d '"'"'" || true)
fi

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "OPENROUTER_API_KEY not set" >&2
    ERRORS=$((ERRORS + 1))
fi

exit $ERRORS
