#!/bin/bash
# Check web tools prerequisites: SERPER_API_KEY
set -euo pipefail

ERRORS=0
PROJECT_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"

# SERPER_API_KEY (required for web search)
SERPER_API_KEY="${SERPER_API_KEY:-}"
if [ -z "$SERPER_API_KEY" ] && [ -f "$PROJECT_ROOT/.env" ]; then
    SERPER_API_KEY=$(grep -E '^SERPER_API_KEY=' "$PROJECT_ROOT/.env" 2>/dev/null | cut -d= -f2- | tr -d '"'"'" || true)
fi

if [ -z "$SERPER_API_KEY" ]; then
    echo "SERPER_API_KEY not set (web search will fail)" >&2
    ERRORS=$((ERRORS + 1))
fi

exit $ERRORS
