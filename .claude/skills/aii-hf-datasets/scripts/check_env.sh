#!/bin/bash
# Check HuggingFace prerequisites: HF_TOKEN (optional but recommended)
set -euo pipefail

ERRORS=0
PROJECT_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"

HF_TOKEN="${HF_TOKEN:-}"
if [ -z "$HF_TOKEN" ] && [ -f "$PROJECT_ROOT/.env" ]; then
    HF_TOKEN=$(grep -E '^HF_TOKEN=' "$PROJECT_ROOT/.env" 2>/dev/null | cut -d= -f2- | tr -d '"'"'" || true)
fi

if [ -z "$HF_TOKEN" ]; then
    echo "HF_TOKEN not set (some datasets may be inaccessible)" >&2
    ERRORS=$((ERRORS + 1))
fi

exit $ERRORS
