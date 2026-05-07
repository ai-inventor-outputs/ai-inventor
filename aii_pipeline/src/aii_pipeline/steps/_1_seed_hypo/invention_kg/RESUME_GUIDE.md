# Resume Functionality Guide (Timestamped Runs)

## Overview

The pipeline now uses **timestamped directories** for all outputs. This allows you to:
- ✅ Keep multiple runs without data loss
- ✅ Resume from specific runs explicitly
- ✅ Track experiment history
- ✅ Never worry about accidentally overwriting data

## How It Works

### Directory Structure

All steps create timestamped output directories:

```
invention_kg/
├── data/
│   ├── _1_papers/
│   │   ├── 350_20250110_143022/   # 350 papers, Jan 10, 14:30:22
│   │   │   ├── papers_2020.json
│   │   │   ├── papers_2021.json
│   │   │   └── ...
│   │   ├── 500_20250110_160145/   # 500 papers (different run)
│   │   └── ...
│   ├── _2_papers_clean/
│   │   ├── 350_20250110_143530/   # 350 papers cleaned
│   │   │   ├── papers_clean_2020.json
│   │   │   └── ...
│   │   ├── 500_20250110_160200/   # 500 papers (different run)
│   │   └── ...
│   └── _3_bblock_runs/
│       ├── 350_20250110_144200/   # 350 papers processed
│       │   ├── paper_idx0/
│       │   │   └── agent_cwd/
│       │   │       └── triples_output.json
│       │   ├── paper_idx1/
│       │   └── ...
│       ├── 200_20250110_161000/   # 200 papers (different run)
│       └── ...
└── config.yaml
```

### Directory Name Format

`{total_papers}_{YYYYMMDD_HHMMSS}` (e.g., `350_20250110_143022` = 350 papers, January 10, 2025 at 14:30:22)

This makes it easy to identify runs by paper count at a glance!

## Configuration

Edit `config.yaml`:

```yaml
# Resume setting for step 3 (get_bblocks)
resume_get_bblocks: true  # true = resume from existing runs, false = create fresh runs

# Specify which runs to use (only when resume_get_bblocks: true)
# Leave empty ("") to auto-detect most recent run
resume_step1_dir: ""  # e.g., "data/_1_papers/350_20250110_143022"
resume_step2_dir: ""  # e.g., "data/_2_papers_clean/350_20250110_143530"
resume_step3_dir: ""  # e.g., "data/_3_bblock_runs/350_20250110_144200"
```

## Usage Modes

### Mode 1: Auto-Resume (Default)

**Config:**
```yaml
resume_get_bblocks: true
resume_step1_dir: ""
resume_step2_dir: ""
resume_step3_dir: ""
```

**What happens:**
- Step 2: Uses most recent step 1 output automatically
- Step 3: Uses most recent step 2 output automatically
- Step 3: Resumes from most recent step 3 output (skips completed papers)

**When to use:**
- Normal iterative development
- Resuming after a crash
- You trust the most recent runs

### Mode 2: Explicit Resume

**Config:**
```yaml
resume_get_bblocks: true
resume_step1_dir: "data/_1_papers/350_20250110_143022"
resume_step2_dir: "data/_2_papers_clean/350_20250110_143530"
resume_step3_dir: "data/_3_bblock_runs/350_20250110_144200"
```

**What happens:**
- Uses exactly the runs you specified
- Step 3 resumes from the specified directory (skips completed papers)

**When to use:**
- You want to use a specific historical run
- Multiple team members working on different branches
- You want reproducibility with exact run versions

### Mode 3: Fresh Start

**Config:**
```yaml
resume_get_bblocks: false
```

**What happens:**
- Step 1: Creates new timestamped directory
- Step 2: Creates new timestamped directory (uses most recent step 1)
- Step 3: Creates new timestamped directory (uses most recent step 2)

**When to use:**
- Starting a new experiment
- Step 1 parameters changed (different papers fetched)
- You want a clean slate

## Resume Logic in Step 3

When `resume_get_bblocks: true` and resuming from a step 3 directory:

### 1. Validation Check
```python
# For each paper_*/agent_cwd/triples_output.json:
is_valid, errors = validator.validate_analysis(output_file)

# Validation checks:
# - Valid JSON format
# - Required fields: paper_type, decision, building_blocks
# - Valid enum values
# - If decision="KEEP", building_blocks must be non-empty (≥1 block)
# - Wikipedia URL format correct
```

### 2. Resume Behavior
- ✅ **Valid output**: Skip this paper
- ❌ **Invalid/missing output**: Process this paper
- ❌ **KEEP with 0 building blocks**: Re-process (validation fails)

### 3. Progress Display
```
Checking for already completed papers...
Found 127 already completed papers
Skipping 127 already completed papers
Will process 223 remaining papers
```

## Examples

### Example 1: Continue After Crash

**Scenario**: Step 3 crashed after processing 50/200 papers

**Solution:**
```yaml
# config.yaml (no changes needed if already set)
resume_get_bblocks: true
resume_step2_dir: ""  # Auto-detect
resume_step3_dir: ""  # Auto-detect
```

**Run step 3 again:**
```bash
python steps/_3_get_bblocks.py
```

**Result:**
- Automatically finds most recent step 3 run
- Validates 50 completed papers
- Processes remaining 150 papers

### Example 2: Use Specific Historical Run

**Scenario**: You want to reprocess papers from a specific experiment

**Solution:**
```yaml
resume_get_bblocks: true
resume_step2_dir: "data/_2_papers_clean/350_20250105_100000"  # Specific run
resume_step3_dir: ""  # Create new (or specify existing to resume)
```

**Run step 3:**
```bash
python steps/_3_get_bblocks.py
```

**Result:**
- Uses papers from Jan 5 run
- Creates new timestamped step 3 output
- Processes all papers fresh

### Example 3: Completely Fresh Run

**Scenario**: Changed year_range or papers_per_year in step 1

**Solution:**
```yaml
resume_get_bblocks: false
```

**Run all steps:**
```bash
python steps/_1_get_papers.py   # New timestamp: 350_20250111_090000
python steps/_2_clean_papers.py # New timestamp: 350_20250111_092000
python steps/_3_get_bblocks.py  # New timestamp: 350_20250111_093000
```

**Result:**
- All new timestamped directories
- No data overwritten
- Old runs preserved

## Helper Messages

Each step prints the directory path to use for resuming:

**Step 1:**
```
To use this run in step 2, set in config.yaml:
  resume_step1_dir: "data/_1_papers/350_20250110_143022"
```

**Step 2:**
```
To use this run in step 3, set in config.yaml:
  resume_step2_dir: "data/_2_papers_clean/350_20250110_143530"
```

**Step 3:**
```
To use this run in step 4, set in config.yaml:
  resume_step3_dir: "data/_3_bblock_runs/350_20250110_144200"
```

## Best Practices

### 1. Default to Auto-Resume
```yaml
resume_get_bblocks: true
resume_step1_dir: ""
resume_step2_dir: ""
resume_step3_dir: ""
```
Most efficient for normal operation.

### 2. Use Explicit Paths for Reproducibility
When you need exact version control:
```yaml
resume_get_bblocks: true
resume_step1_dir: "data/_1_papers/350_20250110_143022"
resume_step2_dir: "data/_2_papers_clean/350_20250110_143530"
resume_step3_dir: "data/_3_bblock_runs/350_20250110_144200"
```

### 3. Fresh Start When Parameters Change
If you modified step 1 config:
```yaml
resume_get_bblocks: false
```
Then run all 3 steps to get consistent data.

### 4. Clean Up Old Runs
Periodically delete old timestamped directories you don't need:
```bash
# Keep only last 3 runs
cd data/_3_bblock_runs
ls -t | tail -n +4 | xargs rm -rf
```

## Validation Rules

The validator (`agent_cwd/validate_bblock_json.py`) checks:

| Check | Description | Error if Fails |
|-------|-------------|----------------|
| **JSON format** | Valid JSON syntax | Invalid JSON → Reprocess |
| **Required fields** | `paper_type`, `decision`, `building_blocks` present | Missing field → Reprocess |
| **Enum values** | `paper_type` in ["new_approach", "literature_review", "dataset", "other"] | Invalid enum → Reprocess |
| **Enum values** | `decision` in ["KEEP", "DISCARD"] | Invalid enum → Reprocess |
| **KEEP consistency** | If `decision="KEEP"`, `building_blocks` must have ≥1 element | Empty blocks → Reprocess |
| **Building block fields** | Each block has `name`, `wikipedia_url`, `relevance` | Missing field → Reprocess |
| **Wikipedia URL format** | Must start with `https://en.wikipedia.org/wiki/` | Invalid format → Reprocess |
| **Wikipedia URL exists** | Page exists on Wikipedia (optional, slow) | Page not found → Reprocess |

## Troubleshooting

### "No step 2 runs found"

**Problem**: Step 3 can't find step 2 output

**Solution**: Run step 2 first, or set explicit path:
```yaml
resume_step2_dir: "data/_2_papers_clean/350_20250110_143530"
```

### "Paper index mismatch"

**Problem**: Resume uses wrong step 2 run with different paper indices

**Solution**: Use explicit paths to ensure consistency:
```yaml
resume_step2_dir: "data/_2_papers_clean/350_20250110_143530"
resume_step3_dir: "data/_3_bblock_runs/350_20250110_144200"
```

### "All papers being reprocessed"

**Problem**: Resume mode not working

**Possible causes:**
1. `resume_get_bblocks: false` in config (set to `true`)
2. Validation failing for all papers (check logs)
3. Wrong step 3 directory (check path in config)

## Summary

| Mode | Config | Use Case |
|------|--------|----------|
| **Auto-Resume** | `resume_get_bblocks: true` + empty paths | Normal operation, resume after crash |
| **Explicit Resume** | `resume_get_bblocks: true` + specific paths | Reproducibility, specific experiment |
| **Fresh Start** | `resume_get_bblocks: false` | New experiment, parameters changed |

**Key Benefit**: Never lose data! All runs are preserved with timestamps.
