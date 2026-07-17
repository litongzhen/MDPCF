# CSVAE Learning-Rate and Beta Probe Design

## Objective

Determine whether CSVAE representation quality and downstream counterfactual
quality improve under a less aggressive learning-rate scheduler and a better
controlled KL weight. The goal is to maximize validation-selected performance
under a fair protocol, not to select configurations from test-set outcomes.

## Scope

This phase uses Adult, latent-model seed 42, the fixed seed-42 black box, the
Transformer CSVAE backbone, and 1000 training epochs. It does not modify the
paper or the five-dataset result table.

Four configurations are compared:

| ID | LR scheduler | Beta schedule |
|---|---|---|
| A | Plateau, patience 10, factor 0.95, no explicit floor | Existing adaptive decay |
| B | Plateau, patience 50, factor 0.5, minimum LR 1e-6 | Existing adaptive decay |
| C | Improved scheduler from B | Fixed beta 1e-3 |
| D | Improved scheduler from B | Linear warm-up from 0 to 1e-3 over 300 epochs, then fixed |

## Validation Protocol

- Reserve 10% of the original training partition as validation data using seed 42.
- Do not use the test partition for checkpoint selection or hyperparameter ranking.
- Train with stochastic reparameterization and Gumbel sampling.
- Validate deterministically with latent means and categorical logits.
- Select each run's checkpoint by its complete validation objective.
- Reload the best checkpoint before exporting encoder, decoder, and latent arrays.

## Evaluation

Stage 1 compares best epoch, reconstruction MSE, categorical CE, label loss,
KL-z, KL-w, beta trajectory, learning-rate trajectory, and validation objective.

Stage 2 takes the best two non-reference configurations according to validation
evidence and runs the same 20 held-out source instances. It reports coverage,
generated-counterfactual validity, target probability, diversity, proximity,
and LOF using the existing evaluation pipeline.

The final test partition is used only for this single Stage-2 comparison after
the two configurations have been selected without test metrics.

## Decision Rules

- Prefer a configuration only when it improves validation reconstruction or the
  complete validation objective without collapsing KL or label accuracy.
- Treat changes below 0.1% as practically negligible unless downstream metrics
  show a consistent improvement.
- Do not claim superiority from one seed. A promising configuration must later
  be confirmed with seeds 42, 43, and 44 before entering the paper.
- If no configuration improves the controlled probe, retain the existing
  2000-epoch setting and report that scheduler/beta tuning did not explain the gap.

## Outputs

- Isolated checkpoints under run tags prefixed with `lr_beta_probe_`.
- Per-epoch training histories in each checkpoint directory.
- Training summary under `results/lr_beta_probe/training_summary.csv`.
- Downstream 20-source summary under `results/lr_beta_probe/cf_summary.csv`.
- A short analysis at `results/lr_beta_probe/analysis.md`.

## Safety and Reproducibility

Existing multiseed results and checkpoints are never overwritten. All runs use
the same source selection, fixed black box, and seed. Failed runs stop the queue
and remain inspectable; they are not silently retried.
