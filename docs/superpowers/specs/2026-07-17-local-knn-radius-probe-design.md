# Per-Source Local-kNN Radius Probe Design

## Objective

Evaluate whether a per-source latent initialization radius improves the diversity--LOF trade-off of MDPCE without materially reducing generated-counterfactual validity. The probe is exploratory and must not alter the completed main or multiseed results until it passes the stated acceptance criteria.

## Current Baseline

The existing `auto_nn` strategy computes one scalar radius per trained latent model:

\[
r_{\mathrm{global}}=\alpha\,\operatorname{median}_{z\in Z_{\mathrm{train}}} d_1(z,Z_{\mathrm{train}}\setminus z),
\]

with `latent_init_radius_scale=0.2`. Every source instance in a dataset uses this same radius.

## Proposed Strategy

Add a `local_knn` initialization strategy. For source latent representation \(z_{0,i}\), compute

\[
r_i=r_{\mathrm{global}}\cdot
\operatorname{clip}\left(
\frac{d_k(z_{0,i},Z_{\mathrm{train}})}
{\operatorname{median}_{z\in Z_{\mathrm{train}}}d_k(z,Z_{\mathrm{train}}\setminus z)},
0.5,2.0
\right).
\]

The initial implementation uses `k=10`. Flattened token-level prediction-relevant latent vectors are used for all distances, matching the representation optimized by the counterfactual generator. The normalized ratio preserves the scale of the current global radius while adapting it to local density. Clipping prevents extreme radii for outlier sources.

## Interface

Add the following command-line options:

- `--latent_init_radius_strategy local_knn`
- `--latent_init_local_k 10`
- `--latent_init_local_min_factor 0.5`
- `--latent_init_local_max_factor 2.0`
- `--source_start_index 0`

`source_start_index` is applied after selecting eligible non-target test instances. With `source_start_index=100` and `num_samples=20`, the probe uses eligible instances 101--120 and does not overlap the final first-100 evaluation set.

## Sampling and Checkpoint Behavior

The generator resolves a separate radius immediately before optimizing each source instance. Existing `fixed` and `auto_nn` behavior remains unchanged. Each source checkpoint records the resolved radius, local kNN distance, normalization reference, and clipped factor. Resume mode reuses completed source checkpoints and recomputes only unfinished sources under the same configuration.

The merged sampling diagnostics report the minimum, median, mean, and maximum resolved radius for the run. The full configuration, including source offset and local-radius parameters, is saved in `config.json` and encoded sufficiently in the result identity to prevent collision with global-radius runs.

## Probe Protocol

Run paired global-versus-local comparisons on GMC and Lending Club:

- latent-model seed: 42;
- black-box seed: 42;
- eligible source instances: positions 101--120;
- number of source instances: 20;
- requested counterfactuals: 10;
- maximum optimization iterations: 3000;
- representation: Transformer-based CSVAE;
- all dataset-specific generation and selection settings unchanged;
- only the radius strategy differs between paired runs.

The global control is rerun on the same held-out probe sources. Historical probes using the first 20 eligible test instances are not used for this comparison.

## Evaluation and Decision Rule

Report coverage, generated-counterfactual validity, diversity, and LOF for each paired run. The local strategy is considered promising only if:

1. generated-counterfactual validity decreases by no more than 0.02 absolute;
2. diversity is maintained or improved;
3. LOF is reduced; and
4. at least one of the two datasets shows a clear joint diversity--LOF improvement without violating criterion 1.

If the results are mixed, retain `auto_nn` as the final method and report the probe only in internal notes. If the criteria are met, repeat the held-out 20-source comparison with seeds 43 and 44 before considering any full 100-source rerun or manuscript claim.

## Tests

Add focused tests for:

- dense and sparse sources receiving smaller and larger radii, respectively;
- lower and upper factor clipping;
- equality with the global radius when the local/reference ratio is one;
- exclusion of self-neighbors in the training reference statistic;
- correct non-overlapping source selection with `source_start_index=100`;
- unchanged behavior for `fixed` and `auto_nn`;
- checkpoint persistence and resume compatibility for per-source radius diagnostics.

## Scope Boundaries

This probe does not modify the latent model, optimization loss, candidate-selection rule, black-box classifier, completed multiseed results, baseline results, or manuscript. No final-paper claim is added until the held-out probe and subsequent multiseed confirmation support it.
