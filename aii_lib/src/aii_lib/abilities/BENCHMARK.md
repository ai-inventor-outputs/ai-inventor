# Skill Scripts Benchmark Results

Benchmark: 60 unique requests per script, 20 parallel per run, 3 runs total.
Using GNU parallel with `-j20`.

Date: 2026-01-13

## Results (seconds per 20 parallel requests)

| Script | Run 1 | Run 2 | Run 3 | Notes |
|--------|-------|-------|-------|-------|
| **fast_web_search** | 22s* | 3.9s | 4.0s | Serper API |
| **fast_web_fetch** | 4.1s | 3.3s | 3.1s | httpbin.org |
| **verify_quotes** | 21s* | 3.7s | 3.4s | example.com |
| **hf_search_datasets** | 3.7s | 3.4s | 3.5s | HuggingFace Hub API |
| **hf_preview_datasets** | 42s | 39s | 63s | Slow - datasets.load_dataset() |
| **hf_download_datasets** | 12s | 12s | 13s | 50 rows per dataset |
| **json_format_mini** | 3.0s | 3.3s | 3.4s | Local file ops |
| **json_validate** | 2.9s | 3.7s | 2.8s | Local validation |
| **lean_runner** | 2.9s | 3.1s | 2.8s | lean-interact |
| **mathlib_pattern** | 4.0s | 4.3s | 4.6s | Loogle API |
| **mathlib_semantic** | 4.6s | 5.5s | 6.9s | Moogle API |
| **or_search_llms** | 22s* | 4.3s | 3.2s | OpenRouter models API |
| **or_get_llm_params** | 3.5s | 3.3s | 3.1s | OpenRouter models API |
| **or_call_llms** | 20s* | 5.1s | 4.6s | gpt-4o-mini, 50 tokens |
| **owid_search** | 64s | 50s | 47s | Local BM25 index |

*Cold start = connection pool warming up after idle period

## Key Findings

### Fast Scripts (~3-5s for 20 parallel)
- `fast_web_search`, `fast_web_fetch`, `verify_quotes` (after warmup)
- `hf_search_datasets` - session pooling via HfApi
- `json_format_mini`, `json_validate` - local operations
- `lean_runner` - lean-interact server
- `mathlib_pattern`, `mathlib_semantic` - external APIs with session pooling
- `or_search_llms`, `or_get_llm_params`, `or_call_llms` (after warmup)

### Slow Scripts
- `hf_preview_datasets` (40-60s): `datasets.load_dataset()` creates new HTTP connections internally, can't be pooled
- `hf_download_datasets` (12s): Actually downloads data
- `owid_search` (50-60s): Loads BM25 index from disk for each request

### Cold Start Issue
Some scripts show ~20s on first run after idle:
- TCP connections timeout after inactivity (~30-60s)
- Warmup creates 1 connection, but 20 parallel requests need more
- Pool scales up on-demand, causing first-batch delay

## Session Pooling Implementation

Scripts with session pooling use this pattern:

```python
POOL_CONNECTIONS = 50
POOL_MAXSIZE = 50

_session = None

def init_xxx():
    global _session
    from requests.adapters import HTTPAdapter

    _session = requests.Session()
    adapter = HTTPAdapter(pool_maxsize=POOL_MAXSIZE, pool_connections=POOL_CONNECTIONS)
    _session.mount("https://", adapter)
    _session.mount("http://", adapter)

    # Warmup
    _session.get("https://api.example.com", timeout=10)

def core_xxx(**kwargs):
    global _session
    # Use _session for all requests
    response = _session.get(...)
```

## Test Command

```bash
# Example: Test fast_web_search with 60 unique queries
for run in 1 2 3; do
  base=$((($run - 1) * 20))
  seq $((base + 1)) $((base + 20)) | parallel -j20 \
    'python .claude/skills/aii_fast_web_research/scripts/aii_fast_web_search.py --query "unique query {}" --max-results 3'
done
```
