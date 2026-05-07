#!/usr/bin/env python
"""
HuggingFace Dataset Download Tool

Download datasets from HuggingFace Hub.

Usage:
    python aii_hf_download_datasets.py openai/gsm8k --config main
    python aii_hf_download_datasets.py openai/gsm8k --config main --split train
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from aii_lib.abilities.aii_ability import aii_ability

SERVER_NAME = "aii_hf_datasets__download_datasets"
DATASETS_DIR = str(Path(__file__).parent.parent / "temp" / "datasets")
CONNECTION_TIMEOUT = 180  # seconds

# =============================================================================
# Core Logic (used by server handler)
# =============================================================================

HF_TOKEN = os.environ.get("HF_TOKEN", "")


def init_download_dataset():
    """Initialize HuggingFace environment for download."""
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_VERBOSITY"] = "error"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(CONNECTION_TIMEOUT)

    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()

    import logging

    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)

    # Pre-import to cache
    from datasets import load_dataset

    # Warmup with tiny dataset slice
    try:
        ds = load_dataset("dair-ai/emotion", split="train[:3]")
        ds.to_pandas()
    except Exception:
        pass


def _truncate_value(value, max_array=3, max_str=200):
    """Recursively truncate arrays/strings/dicts for preview output."""
    if isinstance(value, list):
        return [_truncate_value(v) for v in value[:max_array]]
    if isinstance(value, str):
        return value[:max_str] + "..." if len(value) > max_str else value
    if isinstance(value, dict):
        return {k: _truncate_value(v) for k, v in value.items()}
    return value


def _datasets_server_parquet_files(
    dataset_id: str, config: str | None, split: str | None
) -> list[dict] | None:
    """Query HF Datasets Server for pre-converted parquet files.

    Returns:
      * list of ``{"config","split","url","filename","size"}`` entries when
        the dataset has been auto-converted (the typical case for popular
        datasets, including legacy script-based ones — HF runs the script
        once server-side and freezes the output as Parquet shards).
      * ``None`` if the API can't serve this dataset (uncovered, gated
        without auth, validation failure on HF side, etc.). Caller should
        fall back to ``load_dataset``.

    This is the modern replacement for script-based loading: ``datasets>=3``
    refuses to execute ``<repo>.py`` loader scripts, but the same data
    remains reachable through the Datasets Server's frozen parquet output.
    """
    import httpx

    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    try:
        resp = httpx.get(
            "https://datasets-server.huggingface.co/parquet",
            params={"dataset": dataset_id},
            headers=headers,
            timeout=30.0,
        )
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
        return None
    if resp.status_code != 200:
        return None
    try:
        files = resp.json().get("parquet_files") or []
    except Exception:
        return None
    if not files:
        return None
    if config:
        files = [f for f in files if f.get("config") == config]
    if split:
        files = [f for f in files if f.get("split") == split]
    return files or None


def _save_split_from_rows(rows, output_dir: str, base_name: str) -> dict:
    """Write full/mini/preview JSON for a split given an iterable of rows."""
    import gc
    import json

    rows = list(rows)
    mini_data = rows[: min(3, len(rows))]
    mini_file = Path(output_dir) / f"mini_{base_name}.json"
    with open(mini_file, "w") as f:
        json.dump(mini_data, f, indent=2, ensure_ascii=False, default=str)

    preview_data = [_truncate_value(row) for row in mini_data]
    preview_file = Path(output_dir) / f"preview_{base_name}.json"
    with open(preview_file, "w") as f:
        json.dump(preview_data, f, indent=2, ensure_ascii=False, default=str)

    full_file = Path(output_dir) / f"full_{base_name}.json"
    chunk_size = 1000
    with open(full_file, "w") as f:
        f.write("[\n")
        first = True
        for i in range(0, len(rows), chunk_size):
            for row in rows[i : i + chunk_size]:
                if not first:
                    f.write(",\n")
                first = False
                json.dump(row, f, ensure_ascii=False, default=str)
            gc.collect()
        f.write("\n]")
    return {
        "num_rows": len(rows),
        "preview_file": str(preview_file),
        "mini_file": str(mini_file),
        "full_file": str(full_file),
    }


def _download_via_parquet_api(
    dataset_id: str,
    config: str | None,
    output_dir: str,
    parquet_files: list[dict],
) -> dict:
    """Download + materialize a dataset using HF Datasets Server parquet shards.

    No ``load_dataset`` call → no script execution → works for legacy
    script-based datasets that ``datasets>=3`` would reject.
    Caller applied any ``split`` filter upstream in
    :func:`_datasets_server_parquet_files`; ``parquet_files`` is
    already pre-narrowed.
    """
    import httpx
    import pyarrow.parquet as pq

    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    safe_name = dataset_id.replace("/", "_")
    actual_config = config or (parquet_files[0].get("config") if parquet_files else None)

    # Group shards by split so we materialise one JSON triple per split.
    by_split: dict[str, list[dict]] = {}
    for f in parquet_files:
        by_split.setdefault(f.get("split", "train"), []).append(f)

    result = {
        "success": True,
        "dataset_id": dataset_id,
        "config": actual_config,
        "splits": {},
        "output_files": [],
        "source": "datasets-server-parquet",
    }
    os.makedirs(output_dir, exist_ok=True)

    # follow_redirects=True is required: parquet shard URLs return 302 to a
    # signed S3 URL on cas-bridge.xethub.hf.co. Without it, the redirect
    # body is empty and pyarrow.read_table chokes on a 0-byte file.
    with httpx.Client(headers=headers, timeout=CONNECTION_TIMEOUT, follow_redirects=True) as client:
        for split_name, shards in by_split.items():
            try:
                rows: list[dict] = []
                for shard in shards:
                    r = client.get(shard["url"])
                    r.raise_for_status()
                    tmp_path = Path(output_dir) / f".shard_{shard['filename']}"
                    tmp_path.write_bytes(r.content)
                    table = pq.read_table(tmp_path)
                    rows.extend(table.to_pylist())
                    tmp_path.unlink(missing_ok=True)
                base_name = (
                    f"{safe_name}_{actual_config}_{split_name}"
                    if actual_config
                    else f"{safe_name}_{split_name}"
                )
                info = _save_split_from_rows(rows, output_dir, base_name)
                result["splits"][split_name] = info
                result["output_files"].append(info["full_file"])
            except Exception as e:
                result["splits"][split_name] = {"error": f"{type(e).__name__}: {e}"}
    return result


@aii_ability(
    name="aii_hf_datasets__download_datasets",
    description="Download datasets from HuggingFace Hub as Parquet/JSON files.",
    venv="../../.ability_client_venv",
    requirements="server_requirements.txt",
    worker_init="init_download_dataset",
    check_env="check_env.sh",
)
def core_download_dataset(
    dataset_id: str = "",
    config: str | None = None,
    split: str | None = None,
    output_dir: str | None = None,
) -> dict:
    """
    Download a HuggingFace dataset.

    Strategy: try HF Datasets Server's pre-converted Parquet shards first
    (works for most datasets including legacy script-based ones, since the
    server runs the loader script once on their side and freezes the
    output). Fall back to ``load_dataset`` only when the parquet API
    can't serve this dataset — that path now only works for natively
    Parquet-backed datasets, since ``datasets>=3`` refuses to run
    ``<repo>.py`` loader scripts.

    Args:
        dataset_id: HuggingFace dataset ID (e.g., "openai/gsm8k")
        config: Dataset configuration/subset name
        split: Specific split to load (optional, loads all if empty)
        output_dir: Directory to save files

    Returns:
        Dict with success status and file paths.
    """
    import gc
    import json

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(CONNECTION_TIMEOUT)

    if not dataset_id or not dataset_id.strip():
        return {"success": False, "error": "dataset_id is required"}

    if ".." in dataset_id or dataset_id.startswith("/"):
        return {
            "success": False,
            "error": "Invalid dataset_id: path traversal characters are not allowed",
        }

    if not output_dir:
        return {"success": False, "error": "output_dir is required"}

    # Path 1: Datasets Server parquet API (no script execution).
    parquet_files = _datasets_server_parquet_files(dataset_id, config, split)
    if parquet_files:
        return _download_via_parquet_api(dataset_id, config, output_dir, parquet_files)

    # Path 2: native ``load_dataset`` fallback for datasets the server
    # hasn't auto-converted. Will fail for script-based datasets — by
    # design (datasets>=3 dropped script support and the server-side
    # parquet conversion is the supported migration path).
    from datasets import load_dataset

    try:
        os.makedirs(output_dir, exist_ok=True)
        ds = load_dataset(dataset_id, config, split=split)
        safe_name = dataset_id.replace("/", "_")

        result = {
            "success": True,
            "dataset_id": dataset_id,
            "config": config,
            "splits": {},
            "output_files": [],
        }

        def save_split(split_ds, split_name):
            base_name = (
                f"{safe_name}_{config}_{split_name}" if config else f"{safe_name}_{split_name}"
            )

            # Mini (3 full rows) - extract first before streaming
            mini_data = [dict(split_ds[i]) for i in range(min(3, len(split_ds)))]
            mini_file = Path(output_dir) / f"mini_{base_name}.json"
            with open(mini_file, "w") as f:
                json.dump(mini_data, f, indent=2, ensure_ascii=False, default=str)

            # Preview (3 truncated rows)
            preview_data = [_truncate_value(row) for row in mini_data]
            preview_file = Path(output_dir) / f"preview_{base_name}.json"
            with open(preview_file, "w") as f:
                json.dump(preview_data, f, indent=2, ensure_ascii=False, default=str)

            # Full dataset - stream to disk in chunks to avoid RAM explosion
            full_file = Path(output_dir) / f"full_{base_name}.json"
            chunk_size = 1000
            with open(full_file, "w") as f:
                f.write("[\n")
                first = True
                for i in range(0, len(split_ds), chunk_size):
                    chunk = split_ds.select(range(i, min(i + chunk_size, len(split_ds))))
                    for row in chunk:
                        if not first:
                            f.write(",\n")
                        first = False
                        json.dump(dict(row), f, ensure_ascii=False, default=str)
                    del chunk
                    gc.collect()
                f.write("\n]")

            return {
                "num_rows": len(split_ds),
                "preview_file": str(preview_file),
                "mini_file": str(mini_file),
                "full_file": str(full_file),
            }

        if hasattr(ds, "keys"):
            for split_name, split_ds in ds.items():
                try:
                    result["splits"][split_name] = save_split(split_ds, split_name)
                    result["output_files"].append(result["splits"][split_name]["full_file"])
                except Exception as e:
                    result["splits"][split_name] = {"error": str(e)}
                finally:
                    gc.collect()
        else:
            split_name = split or "train"
            try:
                result["splits"][split_name] = save_split(ds, split_name)
                result["output_files"].append(result["splits"][split_name]["full_file"])
            except Exception as e:
                result["splits"][split_name] = {"error": str(e)}
            finally:
                gc.collect()

        # Drop the HF hub cache for this dataset now that we've materialised
        # full/mini/preview JSON. ``load_dataset`` mirrors the entire dataset
        # (parquet shards + arrow tables) into ``~/.cache/huggingface/hub/``
        # — for big audio/protein datasets that's tens of GB per download
        # and it never gets cleaned up, eating the server pod's container
        # disk until pg+aii_server can't write. We already wrote the JSON
        # the pipeline actually consumes; the cache is dead weight.
        _purge_hf_cache(dataset_id)

        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to load dataset '{dataset_id}': {type(e).__name__}: {e}",
        }


def _purge_hf_cache(dataset_id: str) -> None:
    """Delete the HuggingFace hub cache dir for a dataset.

    HF stores per-dataset caches at
    ``~/.cache/huggingface/hub/datasets--<owner>--<name>/`` (slashes in the
    id become double-dashes). Best-effort deletion: silent on missing dir
    or permission error so a cleanup failure never breaks the download
    response.
    """
    import shutil

    cache_root = Path(os.environ.get("HF_HOME", "/root/.cache/huggingface")) / "hub"
    cache_dir_name = "datasets--" + dataset_id.replace("/", "--")
    target = cache_root / cache_dir_name
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Download a HuggingFace dataset")
    parser.add_argument("dataset_id", help="HuggingFace dataset ID")
    parser.add_argument("--config", default="", help="Dataset configuration")
    parser.add_argument("--split", default="", help="Specific split to load")
    parser.add_argument("--output-dir", default=DATASETS_DIR, help="Output directory")
    args = parser.parse_args()

    from aii_lib.abilities.ability_server import call_server

    result = call_server(
        SERVER_NAME,
        {
            "dataset_id": args.dataset_id,
            "config": args.config,
            "split": args.split,
            "output_dir": args.output_dir,
        },
        timeout=180.0,
    )

    if result is None:
        print(
            "Error: Ability service not available. Start with: aii_server",
            file=sys.stderr,
        )
        sys.exit(1)

    if result.get("success"):
        print(f"\n✓ Downloaded: {result['dataset_id']}")
        for split_name, info in result.get("splits", {}).items():
            print(f"\n  {split_name}:")
            if info.get("error"):
                print(f"    Error: {info['error']}")
            else:
                print(f"    Rows: {info.get('num_rows', '?')}")
                print(f"    Preview: {info.get('preview_file', '')}")
                print(f"    Mini: {info.get('mini_file', '')}")
                print(f"    Full: {info.get('full_file', '')}")
    else:
        print(f"Error: {result.get('error')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
