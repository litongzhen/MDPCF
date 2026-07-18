# CRUDS-Aligned CSVAE Design

## Objective

Restore the original CSVAE/CRUDS latent-variable semantics while preserving
MDPCE's multi-candidate latent optimization:

- `w` is label-related;
- `z` is label-unrelated source context;
- counterfactual generation fixes `z_0` and optimizes multiple `w_i` candidates.

This change covers the CSVAE model, training objective, latent artifacts, and
MDPCE latent search. It does not modify baselines, the manuscript, or existing
result directories.

## Model Semantics

The generative model is

\[
p_\theta(x,y,w,z)
=p_\theta(x\mid w,z)p_\gamma(w\mid y)p(z)p(y),
\]

with

\[
p(z)=\mathcal N(0,I),\qquad
p_\gamma(w\mid y)=\mathcal N(\mu_y,\operatorname{diag}(\sigma_y^2)).
\]

The inference model uses

\[
q_\phi(z\mid x),\qquad q_\phi(w\mid x,y).
\]

The Transformer or MLP token backbone remains configurable. The backbone emits
feature tokens used by both posterior heads, and the decoder reconstructs each
feature token from the concatenated `(w, z)` representation.

## Training Objective

The CSVAE objective contains:

1. numerical and categorical reconstruction losses;
2. `KL(q(z|x) || N(0,I))`;
3. `KL(q(w|x,y) || p(w|y))` using learnable class-conditional Gaussian
   parameters;
4. an adversarial label predictor on `z` to reduce label information in the
   source-context branch.

The adversary minimizes label cross-entropy from `z`. The encoder receives the
reversed gradient so that it makes the label difficult to predict from `z`.
The existing positive label-prediction loss on `z` is removed because it has the
opposite semantics. Label-related structure in `w` is induced by the
class-conditional prior; label separability of `w` is recorded as a diagnostic,
not used to select a model on test data.

Validation and best-checkpoint selection use only the training validation split.
The validation objective mirrors the training objective and records
reconstruction, both KL terms, adversarial accuracy, and total loss separately.

## Counterfactual Generation

For a source instance `x_0`, obtain

\[
z_0=q_\phi(z\mid x_0).
\]

The black-box source prediction supplies the source class only when an initial
`w_0=q_\phi(w\mid x_0,\hat y_0)` is required. No ground-truth test label is read.
Initialize multiple candidates on the radius-`r` sphere around `w_0`:

\[
w_i^{(0)}=w_0+r u_i,\qquad \lVert u_i\rVert_2=1.
\]

During search, freeze the CSVAE decoder, black-box classifier, and `z_0`; optimize
only the candidate tensor `w_i`. Decode candidates as

\[
x_i^{\mathrm{cf}}=D_\theta(w_i,z_0).
\]

The existing validity, input proximity, latent proximity, diversity,
plausibility, actionability, and post-processing terms remain in place. Latent
plausibility and nearest-neighbor calculations move from the old `z` search
space to the new label-related `w` search space. A target-prior penalty toward
`p(w|y^*)` is supported as a configured loss term but is disabled in the first
isolation probe so that the semantic correction is measured without another
simultaneous algorithm change.

## Artifact Isolation

Introduce a new latent artifact protocol, `cruds_aligned_v1`. A compatible
checkpoint stores:

- encoder and decoder weights;
- class-conditional `w` prior parameters;
- adversarial predictor weights;
- deterministic train/test `z` and `w` means;
- protocol metadata and the source-label inference mode.

Loaders must reject old `source_residual_v2` artifacts for this model. New runs
use a distinct run tag and result root. Existing checkpoints and results remain
untouched and cannot be mixed into new summaries.

## Compatibility

`standard_vae` remains unchanged. The `csvae_backbone` switch continues to
support Transformer and MLP. Existing public command-line arguments remain
accepted where their meaning is unchanged; search-specific configuration and
saved metadata explicitly identify `search_latent=w` and `fixed_latent=z`.

## Verification

Unit tests must cover:

- class-conditional `w` KL for both labels;
- label-free `z` encoding;
- no ground-truth source-label access during artifact generation or sampling;
- frozen `z_0` and trainable `w_i` during counterfactual optimization;
- rejection of old latent artifact protocols;
- checkpoint save/load equivalence;
- Transformer and MLP forward shapes;
- unchanged standard-VAE behavior.

Run an end-to-end one-source smoke test before experiments.

## Experiment Gate

The first experiment is a controlled `seed=42`, fixed-black-box probe on Adult,
Bank, and Lending Club with 20 source instances and `max_iter=3000`. Compare the
new Transformer CSVAE against the current corrected Transformer CSVAE under the
same source selection and counterfactual objective.

Proceed to all datasets and three seeds only if:

- coverage decreases by no more than 0.02 on every probe dataset;
- generated-counterfactual validity decreases by no more than 0.03;
- no dataset shows simultaneous degradation in both diversity and LOF;
- training and inference diagnostics show finite losses and no artifact-protocol
  mismatch.

If the gate fails, inspect class-conditional prior scale, adversarial weight, and
`w/z` dimensionality before changing the counterfactual objective. CVAE is not
part of this change and may be added later only as a separate representation
ablation.
