# CSVAE Learning-Rate and Beta Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run a controlled Adult probe for CSVAE learning-rate and beta schedules, then evaluate the best two configurations on 20 held-out sources.

**Architecture:** Add testable schedule helpers and CLI configuration without changing existing defaults. Add deterministic validation and a seeded train/validation split used only when explicitly requested. Run four isolated training configurations, summarize validation evidence, then run downstream evaluation for the two validation-selected configurations.

**Tech Stack:** Python, PyTorch, pytest, PowerShell, existing MDPCE training/sampling/evaluation pipeline.

## Global Constraints

- Adult dataset, latent-model seed 42, black-box seed 42, Transformer CSVAE, 1000 epochs.
- Validation fraction is 0.1 and is drawn only from the original training partition.
- Test metrics cannot select hyperparameters.
- Existing formal results and checkpoints cannot be overwritten.
- Existing CLI defaults remain backward compatible.

---

### Task 1: Testable Scheduler and Beta Policies

**Files:**
- Modify: `mdpcf/csvae/main.py`
- Modify: `utils.py`
- Test: `tests/test_csvae_training_policies.py`

**Interfaces:**
- Produces: `beta_for_epoch(strategy, epoch, target_beta, warmup_epochs, adaptive_beta)`.
- Produces CLI options `--lr_scheduler_patience`, `--lr_scheduler_factor`, `--min_lr`, `--beta_schedule`, `--beta_target`, and `--beta_warmup_epochs`.

- [ ] Write failing tests for fixed beta, linear warm-up endpoints, and scheduler argument parsing.
- [ ] Run `python -m pytest tests/test_csvae_training_policies.py -q` and confirm failure.
- [ ] Implement the policy helper and construct `ReduceLROnPlateau` from CLI values.
- [ ] Preserve the existing adaptive beta behavior when `--beta_schedule adaptive` is used.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Independent and Deterministic Validation

**Files:**
- Modify: `mdpcf/csvae/main.py`
- Modify: `mdpcf/csvae/model.py`
- Modify: `utils.py`
- Test: `tests/test_csvae_validation.py`

**Interfaces:**
- Produces: `split_training_validation_indices(size, fraction, seed)`.
- Produces deterministic encoder/decoder validation paths based on latent means and categorical logits.
- Produces CLI option `--validation_fraction` with default 0.0 and probe value 0.1.

- [ ] Write failing tests proving split indices are deterministic, disjoint, and exhaustive.
- [ ] Write a failing test proving deterministic validation returns identical losses on repeated calls.
- [ ] Implement the seeded split without touching the test partition.
- [ ] Implement deterministic validation while preserving stochastic training behavior.
- [ ] Run `python -m pytest tests/test_csvae_validation.py tests/test_runtime_seed.py -q` and confirm success.

### Task 3: Four-Configuration Training Runner

**Files:**
- Create: `scripts/run_lr_beta_probe.ps1`
- Create: `scripts/summarize_lr_beta_probe.py`
- Test: `tests/test_lr_beta_probe_summary.py`

**Interfaces:**
- Consumes the policy CLI from Task 1 and validation CLI from Task 2.
- Produces `results/lr_beta_probe/training_summary.csv`.

- [ ] Write summary tests using synthetic training-history fixtures for A, B, C, and D.
- [ ] Implement the runner with isolated `lr_beta_probe_*` run tags and the fixed multiseed black box.
- [ ] Implement summary extraction for best epoch, losses, LR changes, and beta trajectory.
- [ ] Run the summary tests and a 2-epoch smoke run.
- [ ] Run all four 1000-epoch training configurations and verify every required artifact exists.

### Task 4: Validation-Based Selection and Downstream Probe

**Files:**
- Modify: `scripts/summarize_lr_beta_probe.py`
- Create: `scripts/run_lr_beta_cf_probe.ps1`
- Create: `results/lr_beta_probe/analysis.md`

**Interfaces:**
- Consumes `training_summary.csv` and selects two non-reference configurations without test metrics.
- Produces `results/lr_beta_probe/cf_summary.csv` from 20 common held-out sources.

- [ ] Rank configurations according to the design decision rules and record the selected IDs.
- [ ] Export or reuse the selected best checkpoints under their isolated run tags.
- [ ] Run sampling and evaluation for the same 20 source instances with fixed generation settings.
- [ ] Generate `cf_summary.csv` with coverage, validity, target probability, diversity, proximity, and LOF.
- [ ] Write `analysis.md` distinguishing validation evidence, downstream evidence, limitations, and the next multi-seed gate.

### Task 5: Verification

**Files:**
- Verify all files modified or created above.

**Interfaces:**
- Consumes all prior outputs.
- Produces a reproducibility and integrity verdict.

- [ ] Run all focused tests and `python -m py_compile` for changed Python files.
- [ ] Confirm no active probe process remains and no formal result directory was overwritten.
- [ ] Confirm all four training runs use seed 42 and the same black-box file hash.
- [ ] Confirm Stage-2 configuration selection used no test metrics.
- [ ] Report numerical findings without claiming multi-seed superiority.
