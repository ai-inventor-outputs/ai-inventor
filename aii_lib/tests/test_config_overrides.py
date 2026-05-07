"""Tests for ``load_config_with_overrides`` — the deep-merge loader that
underlies the ``<thing>.yaml`` + ``<thing>.private.yaml`` convention.

The merge contract:
  - dicts merge recursively, overlay keys win
  - lists are replaced wholesale (no append)
  - scalars are replaced
  - missing private file → identity
  - missing public file → empty + private (or empty + empty if both missing)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from aii_lib.utils.config_overrides import deep_merge, load_config_with_overrides

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# deep_merge — pure function, no FS
# ---------------------------------------------------------------------------


def test_deep_merge_disjoint_keys():
    assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


def test_deep_merge_overlay_wins_on_scalar():
    assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}


def test_deep_merge_recurses_into_dicts():
    base = {"server": {"port": 10010, "host": "0.0.0.0"}}
    over = {"server": {"port": 8020}}
    assert deep_merge(base, over) == {"server": {"port": 8020, "host": "0.0.0.0"}}


def test_deep_merge_lists_replace_not_append():
    """Config lists are ordered alternatives — overlay must not extend."""
    base = {"fallbacks": ["A", "B", "C"]}
    over = {"fallbacks": ["X"]}
    assert deep_merge(base, over) == {"fallbacks": ["X"]}


def test_deep_merge_scalar_overrides_dict():
    """Type changes are explicit overrides, not deep merges."""
    assert deep_merge({"a": {"x": 1}}, {"a": "replaced"}) == {"a": "replaced"}


def test_deep_merge_dict_overrides_scalar():
    assert deep_merge({"a": "old"}, {"a": {"x": 1}}) == {"a": {"x": 1}}


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1}}
    over = {"a": {"c": 2}}
    deep_merge(base, over)
    assert base == {"a": {"b": 1}}
    assert over == {"a": {"c": 2}}


def test_deep_merge_empty_overlay_is_identity():
    assert deep_merge({"a": 1, "b": [1, 2]}, {}) == {"a": 1, "b": [1, 2]}


# ---------------------------------------------------------------------------
# load_config_with_overrides — touches the FS
# ---------------------------------------------------------------------------


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_load_returns_empty_when_public_missing(tmp_path: Path):
    missing = tmp_path / "nonexistent.yaml"
    assert load_config_with_overrides(missing) == {}


def test_load_returns_public_when_no_private(tmp_path: Path):
    pub = tmp_path / "config.yaml"
    _write(pub, "server:\n  port: 10010\n")
    assert load_config_with_overrides(pub) == {"server": {"port": 10010}}


def test_load_merges_private_on_top(tmp_path: Path):
    pub = tmp_path / "config.yaml"
    priv = tmp_path / "config.private.yaml"
    _write(pub, "server:\n  port: 10010\n  host: '0.0.0.0'\n")
    _write(priv, "server:\n  port: 8020\n")
    out = load_config_with_overrides(pub)
    assert out == {"server": {"port": 8020, "host": "0.0.0.0"}}


def test_load_private_only_when_public_empty(tmp_path: Path):
    pub = tmp_path / "config.yaml"
    priv = tmp_path / "config.private.yaml"
    _write(pub, "")
    _write(priv, "secret_key: only-in-private\n")
    assert load_config_with_overrides(pub) == {"secret_key": "only-in-private"}


def test_load_lists_replace(tmp_path: Path):
    """Concrete check that list-replace semantics hold through the FS path."""
    pub = tmp_path / "config.yaml"
    priv = tmp_path / "config.private.yaml"
    _write(pub, "models:\n  - alpha\n  - beta\n")
    _write(priv, "models:\n  - solo\n")
    assert load_config_with_overrides(pub) == {"models": ["solo"]}


def test_load_pure_yaml_safe_load(tmp_path: Path):
    """No Python-tag / loader gadgets — yaml.safe_load only."""
    pub = tmp_path / "config.yaml"
    # ``!!python/object`` would be honoured by ``yaml.load`` but not
    # by ``yaml.safe_load``. Verify the loader uses the safe variant.
    _write(pub, "value: !!python/object/apply:os.system ['echo pwned']\n")
    with pytest.raises(Exception):
        load_config_with_overrides(pub)
