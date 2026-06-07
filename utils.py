import argparse
import importlib
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


METHOD_DISPATCH = {
    "csvae": {
        "default_mode": "train",
        "modes": {
            "train": "mdpcf.csvae.main",
        },
    },
    "black_box": {
        "default_mode": "train",
        "modes": {
            "train": "mdpcf.csvae.train_black_box",
            "run": "mdpcf.csvae.run_black_box",
            "test": "mdpcf.csvae.test_black_box",
            "check": "mdpcf.csvae.test_black_box",
        },
    },
    "evaluate": {
        "default_mode": "run",
        "modes": {
            "run": "evaluation_framework.evaluate",
            "test": "evaluation_framework.evaluate",
        },
    },
    "mdpcf": {
        "default_mode": "sample",
        "modes": {
            "sample": "mdpcf.sample",
        },
    },
    "dice": {
        "default_mode": "sample",
        "modes": {
            "sample": "baselines.dice.sample",
        },
    },
    "revise": {
        "default_mode": "sample",
        "modes": {
            "sample": "baselines.revise.sample",
        },
    },
    "cchvae": {
        "default_mode": "sample",
        "modes": {
            "sample": "baselines.cchvae.sample",
        },
    },
    "wachter": {
        "default_mode": "sample",
        "modes": {
            "sample": "baselines.wachter.sample",
        },
    },
}


def resolve_mode(method, mode):
    """Resolve the effective mode for a method, using its configured default."""
    method_config = METHOD_DISPATCH.get(method)
    if method_config is None:
        available_methods = ", ".join(sorted(METHOD_DISPATCH))
        raise ValueError(
            f"Unknown method '{method}'. Available methods: {available_methods}"
        )

    return mode or method_config["default_mode"]


def prepare_runtime_args(args):
    """Populate runtime-only fields derived from CLI arguments and environment."""
    args.mode = resolve_mode(args.method, args.mode)
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"
    return args


def execute_function(method, mode):
    """Lazy-import and return the entry-point `main` for a given method/mode."""
    resolved_mode = resolve_mode(method, mode)
    module_path = METHOD_DISPATCH[method]["modes"].get(resolved_mode)
    if module_path is None:
        available_modes = ", ".join(sorted(METHOD_DISPATCH[method]["modes"]))
        raise ValueError(
            f"Unknown mode '{resolved_mode}' for method '{method}'. "
            f"Available modes: {available_modes}"
        )

    module = importlib.import_module(module_path)
    return module.main


def get_args():
    parser = argparse.ArgumentParser(
        description="MDPCF / CSVAE counterfactual generation pipeline"
    )

    general_group = parser.add_argument_group("General")
    black_box_group = parser.add_argument_group("Black-box classifier")
    csvae_group = parser.add_argument_group("CSVAE training")
    sample_group = parser.add_argument_group("Counterfactual sampling")
    gmm_group = parser.add_argument_group("GMM plausibility")
    dice_group = parser.add_argument_group("DiCE")
    evaluate_group = parser.add_argument_group("Evaluation")

    general_group.add_argument(
        "--dataname", type=str, default="adult", help="Dataset name."
    )
    general_group.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Mode for the selected method; defaults depend on method.",
    )
    general_group.add_argument(
        "--method",
        type=str,
        default="mdpcf",
        help="Method to run: mdpcf, csvae, black_box, evaluate, or a baseline.",
    )
    general_group.add_argument(
        "--gpu", type=int, default=0, help="GPU index; use -1 to force CPU."
    )
    general_group.add_argument(
        "--verbose", type=str2bool, default=False, help="Enable verbose logging."
    )
    general_group.add_argument(
        "--seed", type=int, default=42, help="Seed for numpy and torch."
    )

    black_box_group.add_argument(
        "--hidden_dims",
        type=int,
        default=16,
        help="Hidden dimension size of the black-box MLP.",
    )
    black_box_group.add_argument(
        "--black_box_epochs",
        type=int,
        default=100,
        help="Number of black-box classifier training epochs.",
    )

    csvae_group.add_argument(
        "--max_beta", type=float, default=1e-2, help="Maximum KL beta."
    )
    csvae_group.add_argument(
        "--min_beta", type=float, default=1e-5, help="Minimum KL beta."
    )
    csvae_group.add_argument(
        "--lambd", type=float, default=0.7, help="Weight of the supervised label term."
    )
    csvae_group.add_argument(
        "--gumbel_softmax",
        type=str2bool,
        default=True,
        help="Use differentiable Gumbel-Softmax for categorical decoding.",
    )
    csvae_group.add_argument(
        "--tau", type=float, default=1.0, help="Temperature for Gumbel-Softmax."
    )
    csvae_group.add_argument(
        "--sigmoid",
        type=str2bool,
        default=True,
        help="Apply sigmoid to continuous outputs during decoding.",
    )
    csvae_group.add_argument(
        "--num_reconstr_loss",
        type=str,
        default="L2",
        help="Reconstruction loss for continuous features.",
    )
    csvae_group.add_argument(
        "--kl_weight",
        type=float,
        default=-1,
        help="Override KL weight; negative keeps the scheduled beta.",
    )
    csvae_group.add_argument(
        "--csvae_epochs",
        type=int,
        default=4000,
        help="Number of CSVAE training epochs.",
    )
    csvae_group.add_argument(
        "--plot_gradients",
        type=str2bool,
        default=False,
        help="Save gradient plots during optimization-related runs.",
    )

    sample_group.add_argument(
        "--save_path",
        type=str,
        default="counterfactual_results",
        help="Directory for generated counterfactual results.",
    )
    sample_group.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of test instances to generate counterfactuals for.",
    )
    sample_group.add_argument(
        "--dice_optimization",
        type=str2bool,
        default=True,
        help="Use DiCE-style joint latent optimization.",
    )
    sample_group.add_argument(
        "--num_encoding",
        type=str,
        default="min_max_torch",
        help="Continuous feature encoding used by the generative model.",
    )
    sample_group.add_argument(
        "--decode_before_loss",
        type=str2bool,
        default=True,
        help="Compute proximity loss after decoding to feature space.",
    )
    sample_group.add_argument(
        "--proximity_weight_input",
        type=float,
        default=1.0,
        help="Weight of proximity loss in decoded feature space.",
    )
    sample_group.add_argument(
        "--proximity_weight_latent",
        type=float,
        default=1.0,
        help="Weight of proximity loss in latent space.",
    )
    sample_group.add_argument(
        "--diversity_weight",
        type=float,
        default=1.0,
        help="Weight of diversity regularization across CF candidates.",
    )
    sample_group.add_argument(
        "--latent_init_radius",
        type=float,
        default=0.5,
        help="Initialization radius around the source latent code.",
    )
    sample_group.add_argument(
        "--proximity_latent_loss",
        type=str,
        default="L2",
        help="Latent-space proximity loss type.",
    )
    sample_group.add_argument(
        "--proximity_input_loss",
        type=str,
        default=None,
        help="Input-space proximity loss type.",
    )
    sample_group.add_argument(
        "--plausibility_weight_latent",
        type=float,
        default=-1.0,
        help="Weight of latent plausibility regularization.",
    )
    sample_group.add_argument(
        "--plausibility_latent_loss",
        type=str,
        default="GMM",
        help="Latent plausibility loss type.",
    )
    sample_group.add_argument(
        "--immutable",
        type=str2bool,
        default=False,
        help="Enforce immutable feature constraints during sampling.",
    )
    sample_group.add_argument(
        "--total_CFs",
        type=int,
        default=1,
        help="Number of counterfactuals to generate per instance.",
    )
    sample_group.add_argument(
        "--target_class",
        type=int,
        default=1,
        help="Target class index for counterfactual generation.",
    )
    sample_group.add_argument(
        "--min_iter", type=int, default=500, help="Minimum optimization iterations."
    )
    sample_group.add_argument(
        "--max_iter", type=int, default=5000, help="Maximum optimization iterations."
    )

    gmm_group.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples used to fit the GMM plausibility model.",
    )
    gmm_group.add_argument(
        "--components",
        type=int,
        default=5,
        help="Number of Gaussian mixture components.",
    )
    gmm_group.add_argument(
        "--dims", type=int, default=2, help="Latent dimensionality for the GMM."
    )
    gmm_group.add_argument(
        "--iterations", type=int, default=20_000, help="Number of GMM optimization steps."
    )
    gmm_group.add_argument(
        "--family", type=str, default="full", help="Covariance family of the mixture model."
    )
    gmm_group.add_argument(
        "--log_freq", type=int, default=5_000, help="Logging frequency in optimization steps."
    )
    gmm_group.add_argument(
        "--radius", type=float, default=8.0, help="L1 bound for sampled latent points."
    )
    gmm_group.add_argument(
        "--mixture_lr",
        type=float,
        default=3e-5,
        help="Learning rate for mixture weights.",
    )
    gmm_group.add_argument(
        "--component_lr",
        type=float,
        default=1e-2,
        help="Learning rate for component means and covariances.",
    )
    gmm_group.add_argument(
        "--visualize",
        type=bool,
        default=False,
        help="Visualize GMM fitting progress.",
    )

    dice_group.add_argument(
        "--dice_method",
        type=str,
        default="gradient",
        help="DiCE backend or optimization strategy.",
    )
    dice_group.add_argument(
        "--dice_post_hoc_sparsity",
        type=str2bool,
        default=False,
        help="Apply DiCE post-hoc sparsity enhancement.",
    )

    evaluate_group.add_argument(
        "--pyod", type=str2bool, default=True, help="Use PyOD for outlier detection."
    )
    evaluate_group.add_argument(
        "--get_stats", type=str2bool, default=True, help="Print summary statistics."
    )
    evaluate_group.add_argument(
        "--get_changed",
        type=str2bool,
        default=False,
        help="Report which features changed in generated counterfactuals.",
    )

    args = parser.parse_args()

    return args
