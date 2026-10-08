# MDPCE: Structured Latent Counterfactual Search

[Chinese Quick Start / 中文运行说明](README.zh-CN.md)

```bash
git clone git@github.com:litongzhen/MDPCF.git
cd MDPCF
```

Research implementation of MDPCE for mixed-type tabular counterfactuals.
The repository and CLI retain the historical name `MDPCF` / `mdpcf`.
The final main method fixes the source `z` branch of a Transformer-based CSVAE
and searches its label-related `w` branch. The standard-VAE ablation also uses
a Transformer; the MLP ablation replaces the CSVAE backbone.

## Environment (Python 3.10)

Existing users can keep their working environment. Do not upload or copy
`.venv` or a Conda environment directory. New users should create their own:

```powershell
conda create -n mdpcf python=3.10
conda activate mdpcf
```

Alternatively, with Python 3.10 installed:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
```

On Windows, install the main-method dependencies in the activated environment:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_environment.ps1 -TorchFlavor cpu
```

For an NVIDIA GPU with a driver compatible with the CUDA 13.0 wheel, replace
`cpu` with `cu130`. The reference experiments used Torch `2.10.0+cu130`.
The installer does not create an environment or alter any other environment.
See the [official Torch wheels](https://pytorch.org/get-started/previous-versions/)
for other hardware. CPU execution is supported but slower.

The equivalent manual installation, also usable on Linux, is:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

`requirements.txt` pins direct dependencies to the observed research versions.
`release/environment-reference.json` records the complete observed package
versions without local paths. It is a diagnostic snapshot, not an installation
lock: transitive dependencies, OS, drivers, hardware and numerical kernels can
still differ. Fresh isolated installation has not yet been validated.

## Optional Baselines

The main method does not require TensorFlow or installation of CARLA/DiCE.
To install the adapted baselines in the same environment on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_environment.ps1 -TorchFlavor cpu -Baselines
```

Or use the manual equivalent:

```bash
python -m pip install -r requirements-baselines.txt
python -m pip install --no-deps -e baselines/dice/DiCE-official
python -m pip install --no-deps -e baselines/CARLA
python scripts/check_environment.py --baselines --skip-dicoflex
```

`--no-deps` is intentional: CARLA's historical dependency metadata requests
obsolete Torch/TensorFlow versions. This project uses adapted PyTorch baseline
paths and the separately specified dependencies. Legacy metadata conflicts may
therefore appear in `pip check`; successful import checks do not establish that
every unused upstream CARLA method is compatible. DiCoFlex source is loaded
from `baselines/DiCoFlex-main`. That upstream source is not redistributed in this
code-only release because no license file was found in the supplied source or
its upstream repository. Obtain it separately from
[ofurman/DiCoFlex](https://github.com/ofurman/DiCoFlex), check its use terms, and
place it at that path before running DiCoFlex or the full baseline import check.
Once it is available, run `python scripts/check_environment.py --baselines`
without `--skip-dicoflex` to check all five baseline adapters.

## Data

Raw data, generated splits and models are not committed. The current automatic
downloader handles Adult only:

```bash
python download_dataset.py
python process_dataset.py --dataname adult
```

Other datasets require obtaining the original files and placing them at the
`data_path` specified in `data/Info/<dataset>.json`, then running
`process_dataset.py --dataname <dataset>`. Check the dataset's access and
redistribution conditions before sharing it. Kaggle access, where applicable,
requires the user's own account/credentials, which must never be committed.
Dataset preprocessing and source order are part of the protocol; installing
packages alone does not reproduce the paper's results.

| Dataset | Required original file(s) |
|---|---|
| Adult | `data/adult/adult.data`, `data/adult/adult.test` (automatic download) |
| Bank | `data/bank/bank.csv` |
| Default | `data/default/default of credit card clients.xls` |
| GMC | `data/gmc/cs-training.csv` |
| Lending Club | `data/lending-club/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv` |

File names, header/column format and case must match `data/Info/*.json`.
After supplying a dataset, preprocess it with its CLI name, e.g.
`python process_dataset.py --dataname bank`.

## Minimal Run

For a complete Windows Adult workflow after installation and data preparation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_quickstart.ps1 -Python python -Gpu -1
```

This trains/checks the black box, trains/reuses the Transformer CSVAE, generates
3 sets of 10 counterfactuals and evaluates them, all under the isolated
`quickstart_e2000` run tag. Use `-Gpu 0` with the CUDA environment. The default 2000
training epochs can take time; `-Epochs 20` checks the workflow only and is not
an experiment-quality model. Re-run with exactly the same arguments to resume.
The wrapper automatically separates run tags by training epochs and output
directories by epochs/source count, so a 20-epoch smoke model cannot overwrite
the 2000-epoch quickstart model. Other changed settings still require a separate
run tag/output directory.

Run all commands from the repository root, using the same configuration for
training, sampling and evaluation. This example is a smoke workflow, not the
complete dataset-specific paper configuration:

```bash
python main.py --dataname adult --method black_box --mode train --gpu -1 --run_tag manual_smoke --seed 42 --black_box_seed 42
python main.py --dataname adult --method csvae --mode train --gpu -1 --run_tag manual_smoke --seed 42 --black_box_seed 42
python main.py --dataname adult --method mdpcf --mode sample --gpu -1 --run_tag manual_smoke --seed 42 --black_box_seed 42 --num_samples 3 --save_path results/manual_smoke
python main.py --dataname adult --method mdpcf --mode evaluate --gpu -1 --run_tag manual_smoke --seed 42 --black_box_seed 42 --num_samples 3 --save_path results/manual_smoke
```

Sampling saves one checkpoint per completed source. Re-running the same
configuration and output path resumes completed sources; do not change settings
inside an existing run directory. Generated outputs stay under the specified
`save_path` and are ignored by Git.

Look recursively under `results/quickstart/` (or your `save_path`):
`cfs.csv` contains the returned counterfactuals, `original_test_samples.csv`
contains their sources, `config.json` records the run settings, and
`evaluation_summary.csv` reports coverage, generated validity, valid-CF yield,
target probability, diversity, proximity and LOF. The CSV column
`validity_expected` is valid-CF yield. Single-output methods have different
output capacity from the main K=10 method.

If sampling is interrupted, completed sources are retained; the interrupted
source may need to restart. Resume the same run rather than deleting artifacts.
A missing-classifier error means the black-box stage was not run with the same
dataset, run tag and classifier settings. A missing-data error means the raw
dataset has not yet been preprocessed. For an import error, run
`python scripts/check_environment.py` in the selected Python environment.

## Paper Protocol

Formal evidence consists of five datasets, three generation/latent seeds
42/43/44, fixed black-box seed 42, 100 common eligible sources, K=10, M=50,
2000 latent-training epochs, best validation checkpoint, and generation bounds
500--3000 iterations with the configured early stopping.
Adult/Bank/Lending Club target class 1; Default/GMC target class 0.

The main/ablation runner trains the fixed black box and latent models as needed:

```powershell
./scripts/run_final_main_ablations.ps1 -Python python -Resume
./scripts/run_final_baselines.ps1 -Python python -Resume
```

Run the baseline queue after the main queue has prepared the fixed classifiers.
These are long research jobs, not quick installation tests. Final dataset-specific
settings are defined in the runners. Small tuning probes are exploratory and are
not substitutes for the formal three-seed results.
Curated metric summaries and sanitized run configurations are under `release/`;
raw counterfactuals and pretrained checkpoints are not included in the default
source release.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

Do not run repository-wide test discovery: vendored projects have independent
dependencies. Some tests require optional baseline packages.

## Project Structure

- `main.py`, `utils.py`, `utils_train.py`: CLI and preprocessing/configuration.
- `mdpcf/csvae/`, `mdpcf/sample.py`: representation learning and generation.
- `mdpcf/checkpoints.py`: resumable source checkpoints.
- `evaluation_framework/`: shared metrics.
- `scripts/`, `tests/`: reproducible runners and first-party regression tests.
- `baselines/`: adapted third-party implementations; retain their notices.
- The manuscript draft is kept locally and is not included in this code release.
- `docs/GITHUB_RELEASE.md`: publication checklist.

## Acknowledgments and Licensing

This implementation adapts [TABSYN](https://github.com/amazon-science/tabsyn),
[DiCE](https://github.com/interpretml/DiCE),
[CARLA](https://github.com/carla-recourse/CARLA), and
[DiCoFlex](https://github.com/ofurman/DiCoFlex), obtained separately.
The Transformer tabular architecture is adapted from TABSYN; CSVAE-specific
branches and objectives are implemented in this project.
Keep all third-party license and attribution files. A license for first-party
code has not yet been selected; do not assume that publishing a GitHub repository
automatically grants an open-source license. Complete the release checklist
before announcing an unrestricted public code release.
