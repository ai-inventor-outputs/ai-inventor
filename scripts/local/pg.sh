#!/usr/bin/env bash
# =============================================================================
# AI Inventor — Project-local PostgreSQL
# =============================================================================
# Runs PostgreSQL from aii_data/db/pgdata (same path as RunPod).
# Usage: bash scripts/local/pg.sh {start|stop|status|init|log}
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PG_DATA="$PROJECT_ROOT/aii_data/db/pgdata"
PG_LOG="$PROJECT_ROOT/aii_data/db/pg.log"
PG_SOCK="$PROJECT_ROOT/aii_data/db/sock"
# ``PG_PORT`` env var lets a contributor run multiple project-local
# clusters side-by-side (e.g., private repo on 5432 + a public clone on
# 5443). When unset, defaults to 5432 — matching ``aii_config/dbos.yaml``.
PG_PORT="${PG_PORT:-5432}"
DB_NAME="aii_inventor"
DB_YAML="$PROJECT_ROOT/aii_config/server/db.yaml"

# Postgres tuning constants live in db.yaml — read max_connections at startup.
# Use the project venv if present so ``aii_lib`` is importable for deep-merge
# loading of ``db.private.yaml`` overrides; fall back to system python3.
_PG_PY="$PROJECT_ROOT/.venv/bin/python"
[ -x "$_PG_PY" ] || _PG_PY=python3
PG_MAX_CONN=$("$_PG_PY" -c "from aii_lib.utils.config_overrides import load_config_with_overrides; print(load_config_with_overrides('$DB_YAML').get('postgres',{}).get('max_connections',200))" 2>/dev/null || echo 200)

# Auto-detect PG version (16 on local Ubuntu, 15 on RunPod Debian)
PG_BIN=""
for v in 17 16 15 14; do
    if [[ -x "/usr/lib/postgresql/$v/bin/pg_ctl" ]]; then
        PG_BIN="/usr/lib/postgresql/$v/bin"
        break
    fi
done
if [[ -z "$PG_BIN" ]]; then
    echo "ERROR: PostgreSQL not found in /usr/lib/postgresql/*/bin/"
    exit 1
fi

pg_init() {
    if [[ -f "$PG_DATA/PG_VERSION" ]]; then
        echo "Cluster already exists at $PG_DATA"
        return 0
    fi
    mkdir -p "$PG_DATA"
    echo "Initializing PG cluster in $PG_DATA ..."
    "$PG_BIN/initdb" -D "$PG_DATA" --auth=trust --username="$USER"
    # Allow local TCP connections (Django connects via localhost)
    cat > "$PG_DATA/pg_hba.conf" <<'HBA'
local all all trust
host  all all 127.0.0.1/32 trust
host  all all ::1/128 trust
HBA
    echo "Cluster initialized (PG $("$PG_BIN/pg_ctl" --version | grep -oP '\d+\.\d+'))"
}

pg_start() {
    if "$PG_BIN/pg_isready" -h localhost -p "$PG_PORT" -q 2>/dev/null; then
        echo "PostgreSQL already running on port $PG_PORT"
        return 0
    fi
    if [[ ! -f "$PG_DATA/PG_VERSION" ]]; then
        pg_init
    fi
    mkdir -p "$PG_SOCK"
    echo "Starting PostgreSQL from $PG_DATA ..."
    "$PG_BIN/pg_ctl" start -D "$PG_DATA" -l "$PG_LOG" -o "-p $PG_PORT -k $PG_SOCK -c max_connections=$PG_MAX_CONN" -w
    # Create database if missing
    if ! "$PG_BIN/psql" -h "$PG_SOCK" -p "$PG_PORT" -lqt 2>/dev/null | cut -d\| -f1 | grep -qw "$DB_NAME"; then
        "$PG_BIN/createdb" -h "$PG_SOCK" -p "$PG_PORT" "$DB_NAME"
        echo "Created database '$DB_NAME'"
    fi
    echo "PostgreSQL running (port $PG_PORT, data: $PG_DATA)"
}

pg_stop() {
    if [[ ! -f "$PG_DATA/postmaster.pid" ]]; then
        echo "PostgreSQL not running"
        return 0
    fi
    "$PG_BIN/pg_ctl" stop -D "$PG_DATA" -m fast
    echo "PostgreSQL stopped"
}

pg_status() {
    "$PG_BIN/pg_ctl" status -D "$PG_DATA" 2>&1 || true
    if "$PG_BIN/pg_isready" -h "$PG_SOCK" -p "$PG_PORT" -q 2>/dev/null; then
        echo "Port $PG_PORT: accepting connections"
        "$PG_BIN/psql" -h "$PG_SOCK" -p "$PG_PORT" -c "SELECT datname, pg_size_pretty(pg_database_size(datname)) FROM pg_database WHERE datistemplate = false;" 2>/dev/null || true
    fi
}

pg_log() {
    if [[ -f "$PG_LOG" ]]; then
        tail -50 "$PG_LOG"
    else
        echo "No log file at $PG_LOG"
    fi
}

case "${1:-}" in
    start)  pg_start ;;
    stop)   pg_stop ;;
    status) pg_status ;;
    init)   pg_init ;;
    log)    pg_log ;;
    *)
        echo "Usage: $0 {start|stop|status|init|log}"
        exit 1
        ;;
esac
