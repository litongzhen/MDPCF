# CRUDS-Aligned CSVAE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train CSVAE with label-related `w` and label-unrelated `z`, then generate counterfactuals by fixing `z_0` and optimizing multiple `w_i` candidates.

**Architecture:** Keep the existing token backbone and decoder, but replace the positive `z` label head with an adversarial `z` predictor and replace the standard-normal `w` KL with a learnable class-conditional Gaussian prior. Save isolated `cruds_aligned_v1` artifacts and move all latent search utilities from `z` to `w` while preserving the existing counterfactual losses.

**Tech Stack:** Python, PyTorch, NumPy, pandas, pytest, PowerShell.

## Global Constraints

- Do not modify baselines or manuscript files.
- Do not delete or overwrite existing checkpoints or result directories.
- Keep `standard_vae` behavior unchanged.
- Use `black_box_seed=42`, `seed=42`, `max_iter=3000` for the first probe.
- Never read ground-truth test labels to construct source latent artifacts.
- Mark new artifacts with `latent_artifact_protocol=cruds_aligned_v1`, `search_latent=w`, and `fixed_latent=z`.

---

### Task 1: Class-Conditional CSVAE Model

**Files:**
- Modify: `mdpcf/csvae/model.py`
- Modify: `mdpcf/csvae/main.py`
- Test: `tests/test_csvae_training_policies.py`
- Test: `tests/test_csvae_validation.py`

**Interfaces:**
- Produces: `Model_CSVAE.class_prior(y) -> (mu_w, logvar_w)` with token-broadcastable tensors.
- Produces: encoder output `(mu_z, logvar_z, z, mu_w, logvar_w, w, z_label_logits)`.
- Produces: `conditional_gaussian_kl(mu_w, logvar_w, prior_mu, prior_logvar)`.

- [ ] **Step 1: Write failing model tests**

Add tests that assert `z` is identical for the same `x` under different `y`, `w` changes with `y`, and class-prior parameters differ by label:

```python
def test_csvae_z_is_label_free_and_w_is_label_conditioned():
    model = make_csvae()
    x_num, x_cat = make_batch()
    out0 = model.encoder(x_num, x_cat, torch.zeros(2), deterministic=True)
    out1 = model.encoder(x_num, x_cat, torch.ones(2), deterministic=True)
    assert torch.allclose(out0[0], out1[0])
    assert not torch.allclose(out0[3], out1[3])

def test_class_conditional_w_prior_selects_label_parameters():
    model = make_csvae()
    mu0, _ = model.class_prior(torch.tensor([0]))
    mu1, _ = model.class_prior(torch.tensor([1]))
    assert not torch.allclose(mu0, mu1)
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_csvae_training_policies.py tests/test_csvae_validation.py -q`

Expected: failure because `class_prior` and conditional KL do not exist and the current positive `z` classification semantics remain.

- [ ] **Step 3: Implement the model and objective**

Add learnable two-class `w_prior_mu` and `w_prior_logvar`, a `class_prior(y)` selector, conditional Gaussian KL, and gradient reversal for `z_label_logits`. Remove the positive classification term from the CSVAE encoder objective and include the adversarial term with an explicit CLI weight `--z_adversary_weight`.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_csvae_training_policies.py tests/test_csvae_validation.py -q`

Expected: all tests pass with finite losses and unchanged standard-VAE assertions.

### Task 2: Artifact Protocol and Label-Safe Encoding

**Files:**
- Modify: `mdpcf/csvae/main.py`
- Modify: `mdpcf/latent_utils.py`
- Modify: `scripts/export_csvae_artifacts.py`
- Test: `tests/test_runtime_seed.py`

**Interfaces:**
- Produces: metadata protocol `cruds_aligned_v1`.
- Produces: deterministic arrays `train_z.npy`, `test_z.npy`, `train_w.npy`, `test_w.npy`.
- Consumes: black-box predicted source labels for `w`, never test ground truth.

- [ ] **Step 1: Write failing artifact tests**

```python
def test_cruds_artifact_protocol_rejects_source_residual_v2(tmp_path):
    write_protocol(tmp_path, "source_residual_v2")
    with pytest.raises(ValueError, match="cruds_aligned_v1"):
        load_generation_inputs(tmp_path, "csvae")

def test_cruds_metadata_declares_search_and_fixed_latents():
    metadata = latent_artifact_metadata("csvae")
    assert metadata["search_latent"] == "w"
    assert metadata["fixed_latent"] == "z"
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_runtime_seed.py -q`

Expected: failure because the current protocol is `source_residual_v2`.

- [ ] **Step 3: Implement isolated artifacts**

Update metadata validation, deterministic export, and checkpoint reuse. Obtain source labels from the fixed black-box predictions when encoding `w`; keep `z=q(z|x)` label-free. Save class-prior and adversary parameters inside `model.pt` and continue saving encoder/decoder wrappers.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_runtime_seed.py -q`

Expected: all protocol and seed tests pass.

### Task 3: Fixed-z, Optimize-w Counterfactual Search

**Files:**
- Modify: `mdpcf/sample.py`
- Test: `tests/test_mdpcf_diversity_loss.py`
- Create: `tests/test_cruds_latent_search.py`

**Interfaces:**
- Produces: `Decoder_Black_Box.set_fixed_z(z)`.
- Produces: `Decoder_Black_Box.decode_w(w_candidates)`.
- Produces: `optimize_w_counterfactuals(w0, z0, ...)` returning optimized `w` and decoded counterfactuals.

- [ ] **Step 1: Write failing search tests**

```python
def test_optimize_w_updates_only_w_candidates():
    decoder = make_decoder_black_box()
    z0 = torch.randn(1, 3, 4, requires_grad=True)
    w0 = torch.randn(1, 3, 4)
    optimized_w, _ = optimize_w_counterfactuals(w0, z0, decoder, **small_args())
    assert optimized_w.shape[-2:] == w0.shape[-2:]
    assert z0.grad is None
    assert decoder.fixed_z.requires_grad is False

def test_initial_candidates_lie_on_w_radius():
    candidates = initialize_latent_candidates(torch.zeros(2, 4), 8, radius=0.2)
    distances = torch.linalg.vector_norm(candidates, dim=-1)
    assert torch.allclose(distances, torch.full_like(distances, 0.2), atol=1e-5)
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_cruds_latent_search.py tests/test_mdpcf_diversity_loss.py -q`

Expected: failure because fixed-z decoding and `optimize_w_counterfactuals` do not exist.

- [ ] **Step 3: Implement search migration**

Rename the search path around its actual semantics, freeze `z_0`, optimize only `w_candidates`, fit radius/GMM/nearest-neighbor plausibility on `train_w`, and update checkpoint column names and sampling metadata to `w_*`. Preserve the existing objective terms and per-source resume behavior.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_cruds_latent_search.py tests/test_mdpcf_diversity_loss.py tests/test_sampling_checkpoints.py tests/test_sample_actionability.py -q`

Expected: all search and resume tests pass.

### Task 4: Probe Runner and Summary

**Files:**
- Create: `scripts/run_cruds_csvae_probe.ps1`
- Create: `scripts/summarize_cruds_csvae_probe.py`
- Modify: `tests/test_experiment_scripts.py`

**Interfaces:**
- Runner datasets: `adult`, `bank`, `lending-club`.
- Runner output: `results/cruds_csvae_probe/runs`.
- Summary output: `results/cruds_csvae_probe/comparison.csv` and `analysis.md`.

- [ ] **Step 1: Write failing script contract test**

Assert the runner fixes both seeds to 42, uses 20 sources and 3000 iterations, passes the new protocol/run tag, and does not reference baseline commands.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_experiment_scripts.py -q`

Expected: failure because the probe scripts do not exist.

- [ ] **Step 3: Implement runner and summary**

Create a resumable PowerShell runner that trains the new CSVAE once per dataset, runs counterfactual generation, evaluates results, and compares them with the corrected current Transformer CSVAE. The summary computes coverage, generated validity, target probability, diversity, proximity, and LOF deltas and evaluates the experiment gate from the design.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_experiment_scripts.py -q`

Expected: all script contract tests pass.

### Task 5: End-to-End Verification and Small Probe

**Files:**
- Modify only if failures require scoped fixes to files listed above.
- Produce: `results/cruds_csvae_probe/comparison.csv`
- Produce: `results/cruds_csvae_probe/analysis.md`

- [ ] **Step 1: Run focused tests**

Run: `pytest tests/test_csvae_training_policies.py tests/test_csvae_validation.py tests/test_runtime_seed.py tests/test_cruds_latent_search.py tests/test_sampling_checkpoints.py tests/test_sample_actionability.py tests/test_experiment_scripts.py -q`

Expected: all pass.

- [ ] **Step 2: Run one-source smoke test**

Run the probe runner for Adult with `NumSamples=1` and confirm training, artifact loading, fixed-z optimization, evaluation, and resume metadata complete without non-finite values.

- [ ] **Step 3: Run the controlled probe**

Run: `powershell -ExecutionPolicy Bypass -File scripts/run_cruds_csvae_probe.ps1 -Resume`

Expected: Adult, Bank, and Lending Club each contain 20 attempted source records and an evaluation summary.

- [ ] **Step 4: Summarize and apply the gate**

Run: `python scripts/summarize_cruds_csvae_probe.py`

Expected: `comparison.csv` contains three datasets and `analysis.md` reports each gate condition without using old-protocol results as new evidence.

- [ ] **Step 5: Run regression tests**

Run: `pytest tests -q`

Expected: all tests pass; any unrelated pre-existing failure is recorded verbatim.
