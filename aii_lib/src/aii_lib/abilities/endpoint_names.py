"""Canonical ability endpoint names — single source of truth.

Every @aii_ability(name=...) decorator and every consumer (prompts, tool loops,
preflight tests, convenience wrappers) should reference these constants instead
of duplicating string literals.

Naming convention: aii_{group}__{action}
  - aii_ prefix for all endpoints
  - double underscore (__) separates group from action

Constant naming: AII_ prefix on all constants to avoid collision with
Claude's built-in tools (WebSearch, WebFetch, etc.).
"""

# ── Web Research ──────────────────────────────────────────────────────────────
AII_WEB_SEARCH = "aii_web_tools__search"
AII_WEB_FETCH = "aii_web_tools__fetch"
AII_WEB_FETCH_GREP = "aii_web_tools__fetch_grep"
AII_WEB_VERIFY_QUOTES = "aii_web_tools__verify_quotes"

# ── Literature ────────────────────────────────────────────────────────────────
AII_SEMSCHOLAR_BIB_FETCH = "aii_semscholar_bib__fetch"

# ── LLM Access ────────────────────────────────────────────────────────────────
AII_OPENROUTER_SEARCH = "aii_openrouter_llms__search"
AII_OPENROUTER_CALL = "aii_openrouter_llms__call"
AII_OPENROUTER_GET_PARAMS = "aii_openrouter_llms__get_params"

# ── Datasets ──────────────────────────────────────────────────────────────────
AII_HF_SEARCH = "aii_hf_datasets__search_datasets"
AII_HF_PREVIEW = "aii_hf_datasets__preview_datasets"
AII_HF_DOWNLOAD = "aii_hf_datasets__download_datasets"
AII_OWID_SEARCH = "aii_owid_datasets__search_datasets"
AII_OWID_DOWNLOAD = "aii_owid_datasets__download_datasets"

# ── Formal Verification ──────────────────────────────────────────────────────
AII_LEAN_RUN = "aii_lean__run"
AII_LEAN_SUGGEST = "aii_lean__suggest"
AII_MATHLIB_SEARCH = "aii_lean__mathlib_pattern_search"

# ── JSON Utilities ────────────────────────────────────────────────────────────
AII_JSON_VALIDATE = "aii_json__validate"
AII_JSON_FORMAT = "aii_json__format"

