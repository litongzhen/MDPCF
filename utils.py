import argparse
import argparse
import importlib
import os
import random

import numpy as np
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
    "mdpcf": {
        "default_mode": "sample",
        "modes": {
            "sample": "mdpcf.sample",
            "evaluate": "evaluation_framework.evaluate",
        },
    },
    "tabcf": {
        "default_mode": "sample",
        "modes": {
            "train": "baselines.tabcf_original.train",
            "sample": "baselines.tabcf_original.sample",
            "evaluate": "evaluation_framework.evaluate",
        },
    },
    "dice": {
        "default_mode": "sample",
        "modes": {
            "sample": "baselines.dice.sample",
            "evaluate": "evaluation_framework.evaluate",
        },
    },
    "revise": {
        "default_mode": "sample",
        "modes": {
            "sample": "baselines.revise.sample",
            "evaluate": "evaluation_framework.evaluate",
        },
    },
    "cchvae": {
        "default_mode": "sample",
        "modes": {
            "sample": "baselines.cchvae.sample",
            "evaluate": "evaluation_framework.evaluate",
        },
    },
    "dicoflex": {
        "default_mode": "sample",
        "modes": {
            "sample": "baselines.dicoflex.sample",
            "evaluate": "evaluation_framework.evaluate",
        },
    },
    "wachter": {
        "default_mode": "sample",
        "modes": {
            "sample": "baselines.wachter.sample",
            "evaluate": "evaluation_framework.evaluate",
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
    set_global_seed(args.seed)
    args.mode = resolve_mode(args.method, args.mode)
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
    else:
        args.device = "cpu"
    return args


def set_global_seed(seed):
    """Seed all stochastic libraries used by the MDPCF experiment pipeline."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
        help="Method to run: mdpcf, csvae, black_box, or a baseline.",
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
    general_group.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional tag that isolates experiment artifacts from legacy runs.",
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
        default=30,
        help="Number of black-box classifier training epochs.",
    )
    black_box_group.add_argument(
        "--black_box_seed",
        type=int,
        default=None,
        help="Optional seed identifying a fixed black-box artifact for generation runs.",
    )

    csvae_group.add_argument(
        "--max_beta", type=float, default=1e-2, help="Maximum KL beta."
    )
    csvae_group.add_argument(
        "--latent_model",
        type=str,
        default="csvae",
        choices=["csvae", "standard_vae", "cvae"],
        help="Latent model for MDPCF: csvae uses two structured branches; standard_vae is label-free; cvae is a single-latent class-conditional VAE.",
    )
    csvae_group.add_argument(
        "--csvae_backbone",
        type=str,
        default="transformer",
        choices=["transformer", "mlp"],
        help="Backbone inside the CSVAE encoder/decoder; mlp is for transformer-module ablation.",
    )
    csvae_group.add_argument(
        "--mlp_parameter_match",
        type=str2bool,
        default=False,
        help="Match each MLP backbone to the corresponding Transformer parameter count.",
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
        default=2000,
        help="Number of CSVAE training epochs.",
    )
    csvae_group.add_argument(
        "--token_dim",
        type=int,
        default=4,
        help="Transformer feature-token width used by the latent model.",
    )
    csvae_group.add_argument(
        "--latent_dim",
        type=int,
        default=4,
        help="Per-token z dimension; currently required to equal token_dim.",
    )
    csvae_group.add_argument(
        "--lr_scheduler_patience", type=int, default=10,
        help="ReduceLROnPlateau patience for latent-model training.",
    )
    csvae_group.add_argument(
        "--lr_scheduler_factor", type=float, default=0.95,
        help="ReduceLROnPlateau multiplicative learning-rate factor.",
    )
    csvae_group.add_argument(
        "--min_lr", type=float, default=0.0,
        help="Minimum latent-model learning rate.",
    )
    csvae_group.add_argument(
        "--beta_schedule", type=str, default="adaptive",
        choices=["adaptive", "fixed", "linear_warmup"],
        help="KL beta schedule used during latent-model training.",
    )
    csvae_group.add_argument(
        "--beta_target", type=float, default=1e-3,
        help="KL beta used by fixed and linear-warmup schedules.",
    )
    csvae_group.add_argument(
        "--beta_warmup_epochs", type=int, default=300,
        help="Epochs required for the linear beta warm-up.",
    )
    csvae_group.add_argument(
        "--z_adversary_weight",
        type=float,
        default=1.0,
        help="Gradient-reversal strength used to remove label information from z.",
    )
    csvae_group.add_argument(
        "--z_adversary_class_balance",
        type=str2bool,
        default=False,
        help="Use inverse-frequency class weights for the z adversary loss.",
    )
    csvae_group.add_argument(
        "--z_adversary_training",
        choices=["joint_grl", "crud_alternating"],
        default="joint_grl",
        help="Train the z adversary jointly with GRL or with CRUD-style alternating updates.",
    )
    csvae_group.add_argument(
        "--z_adversary_steps",
        type=int,
        default=1,
        help="Adversary updates performed before each CRUD-style encoder update.",
    )
    csvae_group.add_argument(
        "--z_adversary_normalize",
        type=str2bool,
        default=False,
        help="Layer-normalize flattened z before CRUD-style adversary updates.",
    )
    csvae_group.add_argument(
        "--w_dim",
        type=int,
        default=2,
        help="Dimension of the global label-related CSVAE latent w.",
    )
    csvae_group.add_argument(
        "--w_structure",
        choices=["global", "token"],
        default="global",
        help="Represent the label-related w as one global vector or one vector per feature token.",
    )
    csvae_group.add_argument(
        "--w_kl_multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to beta for the class-conditional w KL term.",
    )
    csvae_group.add_argument(
        "--validation_fraction", type=float, default=0.0,
        help="Fraction of the training partition reserved for checkpoint validation.",
    )
    csvae_group.add_argument(
        "--reuse_checkpoint",
        type=str2bool,
        default=False,
        help="Skip latent-model training when all checkpoint and latent artifacts exist.",
    )
    csvae_group.add_argument(
        "--source_residual_mode",
        type=str,
        default="encoder_predicted_label",
        choices=["encoder_predicted_label"],
        help="Inference-time construction used for cached CSVAE source residuals.",
    )
    csvae_group.add_argument(
        "--source_label_mode",
        type=str,
        default="black_box_prediction",
        choices=["black_box_prediction"],
        help="Labels used to encode the label-related w artifact at inference.",
    )
    csvae_group.add_argument(
        "--plot_gradients",
        type=str2bool,
        default=False,
        help="Save gradient plots during optimization-related runs.",
    )
    csvae_group.add_argument(
        "--reparam",
        type=str2bool,
        default=True,
        help="Use original TABCF mean/std latent parameterization.",
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
        "--source_start",
        type=int,
        default=0,
        help="Offset within fixed black-box non-target test instances.",
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
        "--diversity_loss",
        type=str,
        choices=["average_distance", "dpp", "dpp_det", "dpp_log"],
        default="average_distance",
        help="Diversity loss across CF candidates.",
    )
    sample_group.add_argument(
        "--latent_init_radius",
        type=float,
        default=0.5,
        help="Initialization radius around the source latent code.",
    )
    sample_group.add_argument(
        "--latent_init_radius_strategy",
        type=str,
        choices=["fixed", "auto_nn"],
        default="fixed",
        help="Strategy used to set the latent initialization radius.",
    )
    sample_group.add_argument(
        "--latent_init_radius_scale",
        type=float,
        default=1.0,
        help="Scale multiplier for auto_nn latent initialization radius.",
    )
    sample_group.add_argument(
        "--target_prior_init_weight",
        type=float,
        default=0.0,
        help="Interpolate w initialization from the source posterior toward the target-class prior.",
    )
    sample_group.add_argument(
        "--latent_init_radius_max_samples",
        type=int,
        default=5000,
        help="Maximum training latents used to estimate auto_nn radius.",
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
        "--validity_loss",
        type=str,
        choices=["cross_entropy", "hinge"],
        default="cross_entropy",
        help="Validity loss for counterfactual optimization.",
    )
    sample_group.add_argument(
        "--validity_margin",
        type=float,
        default=1.0,
        help="Margin used by hinge validity loss.",
    )
    sample_group.add_argument(
        "--categorical_straight_through",
        type=str2bool,
        default=False,
        help="Use hard one-hot categorical values in the forward pass with softmax gradients.",
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
        "--enforce_constraints",
        type=str2bool,
        default=False,
        help="Apply hard actionability/domain/integer constraints during MDPCF sampling.",
    )
    sample_group.add_argument(
        "--total_CFs",
        type=int,
        default=1,
        help="Number of counterfactuals to generate per instance.",
    )
    sample_group.add_argument(
        "--candidate_CFs",
        type=int,
        default=0,
        help="Number of optimized candidate CFs before selecting total_CFs outputs. 0 uses total_CFs.",
    )
    sample_group.add_argument(
        "--cf_selection_metric",
        type=str,
        choices=[
            "mixed_distance",
            "category_first",
            "dpp",
            "target_prob",
            "lof_dpp",
            "lof_distance",
        ],
        default="mixed_distance",
        help="Criterion used to select total_CFs outputs from a larger candidate pool.",
    )
    sample_group.add_argument(
        "--trajectory_quality_selection",
        type=str2bool,
        default=False,
        help="Select the best valid candidate-set snapshot across the optimization trajectory.",
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
        "--max_iter", type=int, default=3000, help="Maximum optimization iterations."
    )
    sample_group.add_argument(
        "--dicoflex_epochs",
        type=int,
        default=10,
        help="Number of epochs for the DiCoFlex conditional flow.",
    )
    sample_group.add_argument(
        "--dicoflex_n_nearest",
        type=int,
        default=32,
        help="Number of target-class neighbours used to train DiCoFlex.",
    )
    sample_group.add_argument(
        "--dicoflex_train_samples",
        type=int,
        default=2000,
        help="Maximum number of training rows used by DiCoFlex to build neighbour pairs. 0 uses all rows.",
    )
    sample_group.add_argument(
        "--dicoflex_p_value",
        type=float,
        default=2.0,
        help="p value used by DiCoFlex for neighbour selection and generation.",
    )
    sample_group.add_argument(
        "--dicoflex_temperature",
        type=float,
        default=0.8,
        help="Sampling temperature used by DiCoFlex.",
    )
    sample_group.add_argument(
        "--dicoflex_prob_threshold",
        type=float,
        default=0.55,
        help="Minimum black-box target probability for DiCoFlex training neighbours.",
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
    dice_group.add_argument(
        "--posthoc_sparsity",
        type=str2bool,
        default=False,
        help="Apply DiCE-style post-hoc sparsity to continuous features after MDPCF generation.",
    )
    dice_group.add_argument(
        "--posthoc_diversity",
        type=str2bool,
        default=False,
        help="Apply post-hoc categorical diversity refinement after MDPCF generation.",
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
