# MDPCF CSVAE + Diverse Latent Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the MDPCF main VAE path with a CSVAE-backed latent model, keep `w` fixed during counterfactual search, and add latent diversity sampling on `z`.

**Architecture:** MDPCF will keep its current training/sampling entry points, but the latent backend will change from a single VAE to a conditional VAE with two latents: `z` for label-related signal and `w` for instance-specific nuisance signal. Training will save both latent arrays and encoder/decoder checkpoints. Sampling will optimize multiple `z` candidates per negative instance while holding `w = w0` fixed, then decode and score the resulting CFs.

**Tech Stack:** PyTorch, pandas, numpy, existing `dice_ml` pipeline, existing dataset preprocessing helpers.

---

### Task 1: Replace the MDPCF latent backend with CSVAE

**Files:**
- Modify: `D:\CS Learning\因果推断\反事实解释\MDPCF\mdpcf\vae\model.py`

- [ ] **Step 1: Define the CSVAE modules**

```python
class CSVAEEncoder(nn.Module):
    def forward(self, x_num, x_cat, y=None):
        return mu_z, logvar_z, z, mu_w, logvar_w, w, y_logits

class CSVAE_Decoder(nn.Module):
    def forward(self, z, w):
        return recon_x_num, recon_x_cat

class Model_CSVAE(nn.Module):
    def forward(self, x_num, x_cat, y):
        return recon_x_num, recon_x_cat, mu_z, logvar_z, z, mu_w, logvar_w, w, y_logits
```

- [ ] **Step 2: Keep the existing tabular reconstruction behavior**

```python
def csvae_loss(x_num, x_cat, recon_x_num, recon_x_cat, y, y_logits, mu_z, logvar_z, mu_w, logvar_w):
    recon_loss = ...
    y_loss = F.cross_entropy(y_logits, y.long())
    kl_z = ...
    kl_w = ...
    return recon_loss + y_loss + kl_z + kl_w
```

- [ ] **Step 3: Preserve `Encoder_model`, `Decoder_model`, `Encoder_model_Z` compatibility**

```python
class Encoder_model_Z(nn.Module):
    def forward(self, x_num, x_cat):
        return z
```

Keep the legacy classes working for any code path that still imports them, but make their outputs come from the CSVAE `z` path.

- [ ] **Step 4: Verify the file still imports cleanly**

Run: `python -m py_compile mdpcf/vae/model.py`
Expected: no syntax errors.

### Task 2: Rework VAE training to train CSVAE and save both latents

**Files:**
- Modify: `D:\CS Learning\因果推断\反事实解释\MDPCF\mdpcf\vae\main.py`
- Modify: `D:\CS Learning\因果推断\反事实解释\MDPCF\utils_train.py`

- [ ] **Step 1: Swap the imported model constructor**

```python
from mdpcf.vae.model import Model_CSVAE
```

- [ ] **Step 2: Update the training loop to feed labels into the model**

```python
Recon_X_num, Recon_X_cat, mu_z, logvar_z, z, mu_w, logvar_w, w, y_logits = model(batch_num, batch_cat, batch_y)
loss = csvae_loss(...)
```

- [ ] **Step 3: Save the latent arrays needed by sampling**

```python
train_z = z_train.cpu().numpy()
test_z = z_test.cpu().numpy()
train_w = w_train.cpu().numpy()
test_w = w_test.cpu().numpy()
np.save(f"{ckpt_dir}/train_z.npy", train_z)
np.save(f"{ckpt_dir}/test_z.npy", test_z)
np.save(f"{ckpt_dir}/train_w.npy", train_w)
np.save(f"{ckpt_dir}/test_w.npy", test_w)
```

- [ ] **Step 4: Make `get_name_form_args` produce a CSVAE-specific ckpt folder**

```python
output_file_name_vae = "_".join([
    f"num_encoding[{args.num_encoding}]",
    f"num_loss[{args.num_reconstr_loss}]",
    "csvae",
])
```

- [ ] **Step 5: Run a smoke compile**

Run: `python -m py_compile mdpcf/vae/main.py utils_train.py`
Expected: no syntax errors.

### Task 3: Update latent loading helpers for CSVAE checkpoints

**Files:**
- Modify: `D:\CS Learning\因果推断\反事实解释\MDPCF\mdpcf\latent_utils.py`

- [ ] **Step 1: Load the new CSVAE encoder/decoder checkpoints**

```python
train_z = torch.tensor(np.load(f"{vae_ckpt_dir}/train_z.npy")).float()
test_z = torch.tensor(np.load(f"{vae_ckpt_dir}/test_z.npy")).float()
train_w = torch.tensor(np.load(f"{vae_ckpt_dir}/train_w.npy")).float()
test_w = torch.tensor(np.load(f"{vae_ckpt_dir}/test_w.npy")).float()
```

- [ ] **Step 2: Stop flattening token latents**

```python
return train_z, test_z, train_w, test_w, curr_dir, dataset_dir, ckpt_dir, info, num_inverse, cat_inverse, y
```

- [ ] **Step 3: Pass decoder and encoder objects through `info`**

```python
info["pre_encoder"] = pre_encoder
info["pre_decoder"] = pre_decoder
info["token_dim"] = token_dim
```

- [ ] **Step 4: Verify the helper still resolves paths correctly**

Run: `python -m py_compile mdpcf/latent_utils.py`
Expected: no syntax errors.

### Task 4: Make MDPCF sampling optimize diverse `z` while fixing `w`

**Files:**
- Modify: `D:\CS Learning\因果推断\反事实解释\MDPCF\mdpcf\sample.py`

- [ ] **Step 1: Replace the latent dataframe with `z` only**

```python
train_z_df = pd.DataFrame(train_z, columns=[f"z_{i}" for i in range(train_z.shape[1])])
test_z_df = pd.DataFrame(test_z, columns=[f"z_{i}" for i in range(test_z.shape[1])])
```

- [ ] **Step 2: Carry `w0` alongside each negative test instance**

```python
original_w = negative_test_w_df.iloc[indx : indx + 1].to_numpy()
```

- [ ] **Step 3: Generate multiple latent seeds on a sphere around `z0`**

```python
directions = generate_equiangular_unit_vectors(total_CFs, d=train_z.shape[1])
z_seeds = z0 + radius * torch.tensor(directions, device=device).float()
```

- [ ] **Step 4: Optimize each candidate with `w` fixed**

```python
z_cf = z_seed.clone().requires_grad_(True)
decoded = pre_decoder(z_cf, w0)
```

- [ ] **Step 5: Add a diversity term across candidates**

```python
diversity_loss = -torch.pdist(torch.stack(z_candidates), p=2).mean()
total_loss = validity_loss + proximity_loss + diversity_weight * diversity_loss
```

- [ ] **Step 6: Keep the output format unchanged**

Write `cfs.csv` and `original_test_samples.csv` with the same columns as before.

- [ ] **Step 7: Run a focused sample smoke test**

Run: `python main.py --dataname adult --method mdpcf --mode sample --num_samples 5`
Expected: `cfs.csv` is written and at least some rows decode successfully.

### Task 5: Verify the end-to-end CSVAE MDPCF path

**Files:**
- No new files

- [ ] **Step 1: Train on a small dataset slice**

Run: `python main.py --dataname adult --method vae --mode train`
Expected: CSVAE checkpoints and `train_z.npy` / `train_w.npy` / `test_z.npy` / `test_w.npy` exist.

- [ ] **Step 2: Sample counterfactuals from the new checkpoint**

Run: `python main.py --dataname adult --method mdpcf --mode sample --num_samples 10`
Expected: sampling completes without loading the old VAE latent layout.

- [ ] **Step 3: Check that `w` is not being optimized**

Inspect the sampling code path and confirm only `z_cf` has `requires_grad_(True)`.

- [ ] **Step 4: Commit**

```bash
git add mdpcf/vae/model.py mdpcf/vae/main.py mdpcf/latent_utils.py mdpcf/sample.py utils_train.py docs/superpowers/plans/2026-05-24-mdpcf-csvae-diverse-latent.md
git commit -m "feat: switch mdpcf to csvae latent sampling"
```

## Self-Review

- Spec coverage: training, saving, loading, sampling, diversity objective, and verification are each covered by a task.
- Placeholder scan: no `TODO`/`TBD` markers remain.
- Type consistency: `z` and `w` are used consistently as separate tensors, and `w` is fixed during CF optimization.
