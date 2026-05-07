#!/usr/bin/env python
"""
OWID Table Download Tool

Download a table from Our World in Data catalog by path.

Usage:
    python aii_owid_download_datasets.py "grapher/energy/2023-12-12/energy_mix"
"""

import argparse
import json
import sys
from pathlib import Path

from aii_lib.abilities.aii_ability import aii_ability

SERVER_NAME = "aii_owid_datasets__download_datasets"
TABLES_DIR = str(Path(__file__).parent.parent / "temp" / "tables")

OWID_BASE_URI = "https://catalog.ourworldindata.org/"
OWID_FORMAT = "feather"


# =============================================================================
# Core Logic (used by server handler)
# =============================================================================


def init_owid_download():
    """Initialize OWID download environment and warmup imports."""
    import os

    os.environ["TQDM_DISABLE"] = "1"

    # Pre-import heavy dependencies

    # Warmup - just ensure imports are loaded
    try:
        pass
    except Exception:
        pass


@aii_ability(
    name="aii_owid_datasets__download_datasets",
    description="Download a table from Our World in Data catalog by path.",
    venv="../../.ability_client_venv",
    requirements="server_requirements.txt",
    worker_init="init_owid_download",
)
def core_owid_download(path: str = "", output_dir: str | None = None) -> dict:
    """
    Download a table from OWID catalog by path.

    Args:
        path: Table path from search results (e.g., "grapher/energy/2023-12-12/energy_mix")
        output_dir: Directory to save files

    Returns:
        Dict with success status and result string
    """
    import os

    from owid.catalog import Table
    from tenacity import retry, stop_after_attempt, wait_exponential

    if not path:
        return {
            "success": False,
            "error": "path is required. Use aii_owid_datasets__search_datasets to find table paths.",
        }

    output_dir = output_dir or TABLES_DIR

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def load_table(path: str):
        url = f"{OWID_BASE_URI}{path}.{OWID_FORMAT}"
        return Table.read(url)

    try:
        import gc

        os.makedirs(output_dir, exist_ok=True)

        df = load_table(path)
        safe_name = path.replace("/", "_").replace("\\", "_")[:80]

        df_reset = df.reset_index()
        num_rows = len(df_reset)

        # Mini (3 full rows) - small enough to hold in memory
        mini_data = df_reset.head(3).to_dict(orient="records")
        mini_file = Path(output_dir) / f"mini_{safe_name}.json"
        with open(mini_file, "w", encoding="utf-8") as f:
            json.dump(mini_data, f, indent=2, ensure_ascii=False, default=str)

        # Preview (3 truncated rows)
        preview_data = []
        for row in mini_data:
            preview_row = {
                k: (str(v)[:200] + "..." if isinstance(v, str) and len(str(v)) > 200 else v)
                for k, v in row.items()
            }
            preview_data.append(preview_row)
        preview_file = Path(output_dir) / f"preview_{safe_name}.json"
        with open(preview_file, "w", encoding="utf-8") as f:
            json.dump(preview_data, f, indent=2, ensure_ascii=False, default=str)

        # Full dataset - stream to disk in chunks to avoid RAM explosion
        full_file = Path(output_dir) / f"full_{safe_name}.json"
        chunk_size = 1000
        with open(full_file, "w", encoding="utf-8") as f:
            f.write("[\n")
            first = True
            for start in range(0, num_rows, chunk_size):
                chunk = df_reset.iloc[start : start + chunk_size].to_dict(orient="records")
                for row in chunk:
                    if not first:
                        f.write(",\n")
                    first = False
                    json.dump(row, f, ensure_ascii=False, default=str)
                del chunk
                gc.collect()
            f.write("\n]")

        # Build human-readable output
        lines = [
            f"Downloaded OWID table: {path}",
            "",
            f"Dimensions: {df.shape[0]:,} rows x {df.shape[1]} columns",
            f"Columns: {', '.join(df_reset.columns[:20])}{'...' if len(df_reset.columns) > 20 else ''}",
            "",
            "Files saved:",
            f"  Mini (READ THIS for development/testing): {mini_file}",
            f"  Preview (DO NOT READ - for logging only): {preview_file}",
            f"  Full (DO NOT READ - for scripts only):    {full_file}",
            "",
            "Sample data (first 3 rows):",
        ]

        # Add preview of data
        for i, row in enumerate(mini_data[:3]):
            lines.append(f"  Row {i + 1}:")
            for k, v in list(row.items())[:10]:
                v_str = str(v)
                if len(v_str) > 80:
                    v_str = v_str[:80] + "..."
                lines.append(f"    {k}: {v_str}")
            if len(row) > 10:
                lines.append(f"    ... ({len(row) - 10} more columns)")

        return {"success": True, "result": "\n".join(lines)}

    except Exception as e:
        return {"success": False, "error": f"downloading {path}: {e!s}"}


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Download an OWID table by path")
    parser.add_argument("path", help="Table path from search results")
    parser.add_argument("--output-dir", default=TABLES_DIR, help="Output directory")
    args = parser.parse_args()

    from aii_lib.abilities.ability_server import call_server

    result = call_server(
        SERVER_NAME,
        {
            "path": args.path,
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

    if isinstance(result, dict):
        if result.get("success"):
            print(result.get("result", ""))
        else:
            print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
            sys.exit(1)
    else:
        print(result)


if __name__ == "__main__":
    main()
