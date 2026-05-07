# Invention Knowledge Graph Pipeline

A multi-step pipeline for extracting knowledge graphs from research papers and identifying research opportunities (blind spots).

## Running the Pipeline

### Prerequisites

1. Activate the virtual environment from the project root:
```bash
cd /path/to/ai-inventor    # your local clone
source .venv/bin/activate
```

2. Ensure all dependencies are installed:
```bash
uv pip install -e .
```

### Running

From the project root directory with venv activated:
```bash
python -m aii_pipeline.steps._seed_hypo.invention_kg
```

## Configuration (config.yaml)

The pipeline is controlled by `config.yaml` in this directory.

### Pipeline Control

```yaml
# Which steps to run (1-9)
start_step: 1
end_step: 9

# Resume from existing run?
# - true: reuse run_id, skip re-fetching papers
# - false: create new run_id, start fresh
resume: true
run_id: "144_20251218_011448"  # Only used when resume=true
```

### Pipeline Steps

| Step | Name | Description |
|------|------|-------------|
| 1 | sel_topics | Select OpenAlex topics |
| 2 | get_papers | Fetch papers from OpenAlex |
| 3 | clean_papers | Clean and deduplicate |
| 4 | get_triples | Extract concept triples |
| 5 | add_wikidata | Add Wikidata grounding |
| 6 | link_to_papers | Link papers to concepts |
| 7 | gen_hypo_seeds | Generate hypothesis seeds |
| 8 | gen_hypo_prompt | Format opportunity prompts |
| 9 | gen_graphs | Generate visualization data |

### Common Configuration Options

```yaml
# Topics to analyze
sel_topics:
  topics:
    - "Multi-Agent Systems and Negotiation"
    - "Reinforcement Learning in Robotics"
    # Add more topics as needed

# Papers per topic/year
get_papers:
  email: "your@email.com"  # For OpenAlex polite pool
  papers_per_year: 3
  year_range:
    start: 2020
    end: 2025

# Hypothesis generation settings
gen_hypo_seeds:
  blind_spots:
    min_shared_concepts: 1  # Min shared concepts between topics
    max_similarity: 1.0     # Max Jaccard similarity (1.0 = no filter)
```

## Common Workflows

### Fresh Run (All Steps)
```yaml
start_step: 1
end_step: 9
resume: false
```

### Resume from Step 7 (Regenerate Graphs)
```yaml
start_step: 7
end_step: 9
resume: true
run_id: "your_run_id"
```

## Output Data

All output is stored in `data/` directory organized by step and run_id:
```
data/
├── _1_topics/          # Selected topics
├── _2_papers/          # Raw papers from OpenAlex
├── _3_clean_papers/    # Cleaned papers
├── _4_triples/         # Extracted concept triples
├── _5_wikidata/        # Wikidata-grounded triples
├── _6_paper_triples/   # Papers linked with triples
├── _7_hypo_seeds/      # Hypothesis seeds (blind spots)
├── _8_hypo_prompt/     # Formatted prompts
└── _9_graphs/          # Graph JSON for visualization
```
