## Setup

Create a conda environment

```
conda create -n mdpcf python=3.10
conda activate mdpcf
```

Install dependencies

```
pip install -r requirements.txt
```

Download and process datasets

```
python download_dataset.py
python process_dataset.py
```

Install local DiCE optimization framework and local CARLA framework

```
cd baselines/dice/DiCE-main
pip install -e .
```

```
cd baselines/CARLA
pip install -e .
```

Train VAE model

```
python main.py --dataname [NAME_OF_DATASET] --method vae --mode train
```

## Usage

To generate counterfactuals with MDPCF run the following command:

```
python main.py --dataname [NAME_OF_DATASET] --method mdpcf --mode sample --num_samples [NUMBER_OF_SAMPLES]
```

The same command can be used to generate counterfactuals using a competitor method, e.g.:

```
python main.py --dataname [NAME_OF_DATASET] --method wachter --mode sample --num_samples [NUMBER_OF_SAMPLES]
```

The resulting counterfactuals are saved as `.csv` files under the directory `/counterfactual_results`.

For calculation of all metrics (evaluation), given the generated csv files, run the following command:

```
python main.py --dataname [NAME_OF_DATASET] --method wachter --mode evaluate --num_samples [NUMBER_OF_SAMPLES]
```

## General code structure

- [/baselines](./baselines/) contains all baseline methods used in our experiments. Each folder contains a `sample.py` file which finds and saves counterfactuals.
- [/utils.py](./utils.py) contains all hyperparameter definitions.
- [/evaluation_framework/evaluate.py](./evaluation_framework/evaluate.py) is used to calculate all metrics, given saved counterfactuals.
- [/data/Info](./data/Info) contains all necessary information for each dataset.
- [/download_dataset.py](./download_dataset.py) download the datasets from the provided information.
- [/process_dataset.py](./process_dataset.py) pre-processes the datasets.
- [/lamda_loss_plot_all.py](./lamda_loss_plot_all.py) runs the ablation study experiment.
- [/feature_usage_plot.py](./feature_usage_plot.py) runs the feature-usage experiment.
- [/shapley.py](./shapley.py) runs the shapley-values experiment.

## Acknowledgments

In our code we use and alter versions of the following repositories.

- [TABSYN](https://github.com/amazon-science/tabsyn) (VAE structure)
- [DiCE](https://github.com/interpretml/DiCE) (under `/baselines/dice/DiCE-main`), SGD optimization for counterfactual generation.
- [CARLA](https://github.com/carla-recourse/CARLA) (under `/baselines`), the CARLA framework is used for evaluation against baseline methods.
