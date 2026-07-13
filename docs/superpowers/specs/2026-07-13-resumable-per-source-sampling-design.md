# Resumable Per-Source Sampling Design

## Goal

Make MDPCE sampling observable and restartable at source-sample granularity without changing counterfactual optimization, evaluation metrics, or the final output schema.

## Scope

The sampling loop processes source instances independently.  Each source instance is one atomic checkpoint unit.  A completed source is never optimized again after a restart, whether it produced valid counterfactuals or not.  An interruption during an individual source leaves no completion marker, so that source is recomputed on the next run.

## Checkpoint Layout

For an existing result directory `save_path`, sampling writes under `save_path/checkpoints/`:

- `sources/<id>.json`: completion marker containing the source id, completion status, and per-source sampling counters.
- `cfs/<id>.csv`: selected final counterfactuals for a successful source.
- `latents/<id>.csv`: matching latent vectors for a successful source.
- `projection/<id>.csv`: projection-shift records when the source reached projection.
- `posthoc_sparsity/<id>.csv` and `posthoc_diversity/<id>.csv`: optional post-hoc records when enabled.

Every shard is written to a temporary sibling file and atomically replaced.  The source JSON marker is written only after all relevant shards have been committed.  The marker is therefore the sole authority for skip/resume decisions.

## Runtime Behavior

At startup, the sampler discovers valid source markers and skips exactly those source ids.  It shows a progress bar over all requested sources, including resumed sources, and prints a concise completed/total count after every committed source.

When the loop finishes, it merges shards ordered by source id and emits the existing top-level files (`cfs.csv`, `cf_latents.csv`, projection and post-hoc summaries, `sampling_diagnostics.csv`, and downstream summaries) with their current schemas.  No downstream evaluator reads partial checkpoints.  Existing completed result directories without checkpoints remain readable as final results; resume behavior is used only for new or incomplete checkpointed runs.

## Error Handling

Malformed, missing, or inconsistent checkpoint markers are ignored rather than trusted, causing the corresponding source to be recomputed.  Empty successful sources still receive a marker and contribute their zero-valued counters to final diagnostics.

## Tests

Unit tests cover atomic CSV/JSON shard persistence, marker discovery, rejection of malformed markers, source-id ordering during merge, and reconstruction of aggregate diagnostics from per-source counters.  Existing actionability and evaluation tests remain unchanged.

## Non-Goals

This change does not checkpoint optimization iterations, Adam state, models, or black-box training.  It does not alter experiment hyperparameters or rerun historical outputs.
