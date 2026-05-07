# AI Inventor

An autonomous AI research pipeline that generates novel scientific hypotheses, designs experiments, runs them, and produces academic papers — all powered by Claude.

## What it does

AI Inventor takes a AII prompt (e.g. "computational linguistics") and autonomously:

1. **Generates hypotheses** — searches literature, identifies gaps, proposes novel ideas
2. **Designs experiments** — creates concrete experimental plans with code
3. **Runs experiments** — executes code, collects results, iterates on failures
4. **Writes papers** — produces LaTeX papers with figures, references, and results

## Quick start

### Prerequisites

- Python 3.12+
- [Claude Code CLI](https://claude.ai/download) signed in with an active
  [Claude Max plan](https://www.anthropic.com/max) — **recommended**: the
  default ``claude_max`` llm_backend drives the agent through your
  subscription quota with zero extra config. An Anthropic API key works
  too (set ``ANTHROPIC_API_KEY`` and switch ``llm_backend`` in
  ``aii_config/pipeline/harness/llm_backend.yaml``), but expect more
  setup tweaks and per-token billing.
- [uv](https://docs.astral.sh/uv/) package manager
- PostgreSQL 14+ binaries (the launcher runs Postgres project-locally via
  `scripts/local/pg.sh` — no system service or root role required)
- `tmux` (the launcher runs the server + pipeline as detached sessions)

On Ubuntu/Debian:

```bash
sudo apt install -y postgresql tmux
# No need to start the system service — the project-local cluster spawned
# below uses its own data dir + socket and binds the same binaries.
```

On macOS (Homebrew):

```bash
brew install postgresql@16 tmux
# Don't ``brew services start`` — pg.sh boots its own cluster.
```

### Install

```bash
git clone https://github.com/ai-inventor-outputs/ai-inventor.git
cd ai-inventor

# uv creates the venv and installs all four sub-packages (declared as a
# uv workspace in the root pyproject.toml) in editable mode in one shot.
uv sync
source .venv/bin/activate  # or: source .venv/bin/activate.fish
```

### Configure

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
# Edit .env — at minimum, set OPENROUTER_API_KEY (LLM calls) and
# SERPER_API_KEY (web search). See the comments in .env.example for
# the rest.
```

Authenticate the Claude Code CLI (used to drive the agent):

```bash
claude login
```

Pipeline configuration lives in `aii_config/pipeline/pipeline.yaml`.

### Start the project-local Postgres

Workflow state is persisted via [DBOS](https://docs.dbos.dev/), which needs a
Postgres instance. The repo ships a tiny helper that boots one in
`aii_data/db/pgdata` (no `sudo`, no system service):

```bash
bash scripts/local/pg.sh init    # one-time: initdb + create the aii_inventor DB
bash scripts/local/pg.sh start   # boot the cluster (idempotent)
# bash scripts/local/pg.sh status / stop / log when needed
```

DB connection settings live in `aii_config/dbos.yaml` (host = relative
socket dir `aii_data/db/sock`, port 5432, db name `aii_inventor`). To
override any of these per machine — different port, different socket,
or an external/managed Postgres — drop a sibling `dbos.private.yaml`
next to it. The file is gitignored (matched by `*.private.yaml` in
`.gitignore`) and deep-merges on top of the public defaults at load
time.

```yaml
# aii_config/dbos.private.yaml — pointing DBOS at a remote Postgres
dbos:
  postgres:
    host: db.example.com    # bare hostname → TCP; leading '/' → Unix socket dir
    port: 5432
    user: aii
    password: ${DB_PASSWORD} # or set via .env, never commit secrets
    app_db_name: aii_inventor
    sys_db_name: aii_inventor_dbos_sys
```

### Run

```bash
# Pass your research prompt with --prompt. The launcher starts the
# ability server + pipeline in tmux and streams pipeline output.
# Ctrl+C detaches; `aii_launcher --stop-local` halts everything.
aii_launcher --prompt "Multi-LLM Agent Systems"
```

The pipeline will run autonomously. Output lands in `aii_data/runs/<run-id>/`.

## Architecture

```text
aii_lib/       — Core library: agent orchestration, LLM backends, telemetry
aii_pipeline/  — Pipeline runner: hypothesis → experiment → paper
aii_server/    — Ability server: web search, Semantic Scholar, HuggingFace, etc.
aii_launcher/       — CLI orchestrator: starts server + pipeline, manages tmux
aii_config/    — Configuration (YAML)
.claude/skills/— Ability implementations (Python scripts)
```

## Configuration

| File | Purpose |
|------|---------|
| `aii_config/pipeline/pipeline.yaml` | Pipeline steps, iterations, research scope |
| `aii_config/pipeline/harness/agent_backend.yaml` | Active agent_backend + per-backend defaults (timeouts, retries, telemetry) |
| `aii_config/pipeline/harness/llm_backend.yaml` | Active llm_backend (claude_max / openrouter) + per-backend defaults (model, effort) |
| `aii_config/pipeline/harness/execute_env.yaml` | Exec env mode (local / runpod) |
| `aii_config/pipeline/io/{sources,sinks}.yaml` | Run-bus channels (send_message, console, otel, …) |
| `.env` | API keys (not committed) |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `claude: command not found` | Install [Claude Code CLI](https://claude.ai/download) and run `claude login`. |
| `OPENROUTER_API_KEY not set` (or similar) | Copy `.env.example` to `.env` and fill in the keys you actually use. |
| Pipeline starts but stalls inside `gen_hypo` | The agent is running through web-search + reasoning. First iterations can take a few minutes. Tail `aii_data/logs/runs/<session>.log` to watch progress. |
| `--runpod` says "not included in this build" | Expected — RunPod orchestration is private. Use `aii_launcher --local` (the default). |
| Want to re-attach to a running pipeline | `tmux attach -t aii-<run-id>` (run id is printed at startup). |
| Halt everything | `aii_launcher --stop-local`. |
| Re-run a fresh pipeline | `aii_launcher --stop-local && aii_launcher --prompt "..."`. |
| `connection ... failed: No such file or directory` (Postgres socket) | Project-local Postgres isn't running. Run `bash scripts/local/pg.sh start` (or `init` first if you haven't). |

## License

Apache-2.0 — see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. We use the Developer Certificate of Origin (DCO).
