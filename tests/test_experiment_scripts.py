import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from utils import get_args
from utils_train import get_optim_params_form_args


class DummyEncoder:
    def __init__(self):
        self.categories_ = [
            ["Private", "Self-emp"],
            ["HS-grad", "Bachelors", "Masters"],
        ]

    def get_feature_names(self, categorical):
        return [
            "workclass_Private",
            "workclass_Self-emp",
            "education_HS-grad",
            "education_Bachelors",
            "education_Masters",
        ]


class DummyDataset:
    categorical = ["workclass", "education"]
    encoder = DummyEncoder()


class ExperimentScriptsTest(unittest.TestCase):
    def test_standard_cvae_probe_uses_matched_independent_protocol(self):
        repo_root = Path(__file__).resolve().parents[1]
        runner = repo_root / "scripts" / "run_standard_cvae_probe.ps1"

        self.assertTrue(runner.is_file())
        content = runner.read_text(encoding="utf-8")
        self.assertIn('@("adult", "default", "gmc")', content)
        self.assertIn('"--latent_model", "cvae"', content)
        self.assertIn('"--csvae_epochs", "2000"', content)
        self.assertIn('"--validation_fraction", "0.1"', content)
        self.assertIn('"--token_dim", "4"', content)
        self.assertIn('"--latent_dim", "4"', content)
        self.assertIn('[int]$Seed = 42', content)
        self.assertIn('[int]$BlackBoxSeed = 42', content)
        self.assertIn('[int]$SourceStart = 110', content)
        self.assertIn('[int]$NumSamples = 10', content)
        self.assertIn('[int]$MaxIter = 3000', content)
        self.assertIn('"--candidate_CFs", "50"', content)
        self.assertIn('"--cf_selection_metric", "lof_dpp"', content)
        self.assertNotIn('"--trajectory_quality_selection", "True"', content)
        self.assertIn('"default" { return 0 }', content)
        self.assertIn('"gmc" { return 0 }', content)

    def test_trajectory_quality_probe_uses_common_d4_sources_without_training(self):
        repo_root = Path(__file__).resolve().parents[1]
        runner = repo_root / "scripts" / "run_trajectory_quality_probe.ps1"

        self.assertTrue(runner.is_file())
        content = runner.read_text(encoding="utf-8")
        self.assertIn('@("adult", "default", "gmc")', content)
        self.assertIn('@("csvae", "standard_vae")', content)
        self.assertIn('"--trajectory_quality_selection", "True"', content)
        self.assertIn('"--cf_selection_metric", "lof_dpp"', content)
        self.assertIn('[int]$SourceStart = 110', content)
        self.assertIn('[int]$NumSamples = 10', content)
        self.assertIn('[int]$MaxIter = 3000', content)
        self.assertIn('"default" { return 0 }', content)
        self.assertIn('"gmc" { return 0 }', content)
        self.assertNotIn('"--mode", "train"', content)

    def test_d6_probe_uses_dimension_matched_isolated_protocol(self):
        repo_root = Path(__file__).resolve().parents[1]
        runner = repo_root / "scripts" / "run_d6_latent_probe.ps1"
        summary = repo_root / "scripts" / "summarize_d6_latent_probe.py"

        self.assertTrue(runner.is_file())
        self.assertTrue(summary.is_file())
        content = runner.read_text(encoding="utf-8")
        self.assertIn('@("adult", "default", "gmc")', content)
        self.assertIn('"--token_dim", "6"', content)
        self.assertIn('"--latent_dim", "6"', content)
        self.assertIn('"--w_dim", "6"', content)
        self.assertIn('[int]$SourceStart = 110', content)
        self.assertIn('[int]$NumSamples = 10', content)
        self.assertIn('[int]$MaxIter = 3000', content)
        self.assertIn('[int]$BlackBoxSeed = 42', content)
        self.assertIn('"default" { return 0 }', content)
        self.assertIn('"gmc" { return 0 }', content)

    def test_csvae_search_parameter_probe_uses_disjoint_screening_protocol(self):
        repo_root = Path(__file__).resolve().parents[1]
        runner = repo_root / "scripts" / "run_csvae_search_parameter_probe.ps1"

        self.assertTrue(runner.is_file())
        content = runner.read_text(encoding="utf-8")
        self.assertIn('@("adult", "bank", "lending-club")', content)
        self.assertIn("[int]$NumSamples = 10", content)
        self.assertIn("[int]$SourceStart = 100", content)
        self.assertIn("[int]$MaxIter = 1000", content)
        self.assertIn("[int]$TargetClass = 1", content)
        self.assertIn('"--target_class", "$TargetClass"', content)
        self.assertIn('[string]$ResultsRoot = "results\\csvae_search_parameter_probe\\stage_a"', content)
        self.assertIn("[string[]]$ConfigIds = @()", content)
        self.assertIn("@(0.1, 0.2, 0.3)", content)
        self.assertIn("@(1.0, 2.0, 3.0)", content)
        self.assertIn('"--candidate_CFs", "50"', content)
        self.assertIn('"--cf_selection_metric", "lof_dpp"', content)
        self.assertIn('"--run_tag", $RunTag', content)
        self.assertNotIn('"--mode", "train"', content)
        self.assertNotIn('"--source_start", "0"', content)

    def test_csvae_search_selection_probe_uses_only_stage_a_finalists(self):
        repo_root = Path(__file__).resolve().parents[1]
        runner = repo_root / "scripts" / "run_csvae_search_selection_probe.ps1"

        self.assertTrue(runner.is_file())
        content = runner.read_text(encoding="utf-8")
        self.assertIn('"results\\csvae_search_parameter_probe\\stage_b_selected.json"', content)
        self.assertIn('@("dpp", "lof_dpp", "category_first")', content)
        self.assertIn("[int]$NumSamples = 10", content)
        self.assertIn("[int]$SourceStart = 100", content)
        self.assertIn("[int]$MaxIter = 3000", content)
        self.assertIn('"--candidate_CFs", "50"', content)
        self.assertNotIn('"--mode", "train"', content)

    def test_csvae_search_confirmation_uses_disjoint_holdout_sources(self):
        repo_root = Path(__file__).resolve().parents[1]
        runner = repo_root / "scripts" / "run_csvae_search_confirmation.ps1"

        self.assertTrue(runner.is_file())
        content = runner.read_text(encoding="utf-8")
        self.assertIn('"results\\csvae_search_parameter_probe\\confirmation_selected.json"', content)
        self.assertIn("[int]$NumSamples = 10", content)
        self.assertIn("[int]$SourceStart = 110", content)
        self.assertIn("[int]$MaxIter = 3000", content)
        self.assertIn('"--candidate_CFs", "50"', content)
        self.assertNotIn('"--mode", "train"', content)

    def test_cruds_probe_runner_uses_the_isolated_fixed_seed_protocol(self):
        repo_root = Path(__file__).resolve().parents[1]
        runner = repo_root / "scripts" / "run_cruds_csvae_probe.ps1"
        summary = repo_root / "scripts" / "summarize_cruds_csvae_probe.py"

        self.assertTrue(runner.is_file())
        self.assertTrue(summary.is_file())
        content = runner.read_text(encoding="utf-8")
        self.assertIn('@("adult", "bank", "lending-club")', content)
        self.assertIn('[int]$Seed = 42', content)
        self.assertIn('[int]$BlackBoxSeed = 42', content)
        self.assertIn('[int]$NumSamples = 20', content)
        self.assertIn('[int]$MaxIter = 3000', content)
        self.assertIn('"cruds_csvae_probe"', content)
        self.assertIn('"--validation_fraction", "0.1"', content)
        self.assertNotIn('"--method", "dice"', content.lower())
        self.assertNotIn('"--method", "wachter"', content.lower())

    def test_cruds_configuration_runner_uses_disjoint_probe_sources(self):
        repo_root = Path(__file__).resolve().parents[1]
        runner = repo_root / "scripts" / "run_cruds_globalw_probe.ps1"

        content = runner.read_text(encoding="utf-8")

        self.assertIn("[int]$SourceStart = 100", content)
        self.assertIn('"--source_start", "$SourceStart"', content)
        self.assertIn("[int]$Config.source_start -eq $SourceStart", content)

    def test_evaluation_resolves_disjoint_mdpcf_probe_directory(self):
        repo_root = Path(__file__).resolve().parents[1]
        evaluator = repo_root / "evaluation_framework" / "evaluate.py"

        content = evaluator.read_text(encoding="utf-8")

        self.assertIn('source_start = getattr(args, "source_start", 0)', content)
        self.assertIn('f"{output_folder_name}_from_{source_start}"', content)

    def test_mdpcf_constraints_are_disabled_by_default(self):
        with patch("sys.argv", ["main.py"]):
            args = get_args()

        self.assertFalse(args.enforce_constraints)

    def test_mdpcf_output_name_marks_constraints_only_when_enabled(self):
        base_args = {
            "proximity_weight_input": 1.0,
            "diversity_weight": 1.0,
            "diversity_loss": "average_distance",
            "latent_init_radius": 0.5,
            "hidden_dims": 16,
            "method": "mdpcf",
            "proximity_weight_latent": 0.1,
            "proximity_latent_loss": "L2",
            "plausibility_weight_latent": -1.0,
            "plausibility_latent_loss": "GMM",
            "proximity_input_loss": "reconstruction",
            "validity_loss": "hinge",
            "validity_margin": 1.0,
            "categorical_straight_through": True,
            "posthoc_sparsity": True,
            "posthoc_diversity": True,
            "candidate_CFs": 30,
            "total_CFs": 10,
            "cf_selection_metric": "category_first",
            "immutable": False,
        }

        unconstrained = get_optim_params_form_args(
            SimpleNamespace(**base_args, enforce_constraints=False)
        )
        constrained = get_optim_params_form_args(
            SimpleNamespace(**base_args, enforce_constraints=True)
        )

        self.assertNotIn("constraints", unconstrained)
        self.assertIn("_constraints", constrained)

    def test_carla_postprocess_projects_invalid_one_hot_blocks(self):
        from baselines.carla_postprocess import project_encoded_categories

        encoded = pd.DataFrame(
            {
                "age": [0.5, 0.7],
                "workclass_Private": [0.0, 0.2],
                "workclass_Self-emp": [0.0, 0.8],
                "education_HS-grad": [0.1, 0.0],
                "education_Bachelors": [0.7, 0.0],
                "education_Masters": [0.2, 0.0],
            }
        )

        projected = project_encoded_categories(encoded, DummyDataset())

        self.assertEqual(projected.loc[0, ["workclass_Private", "workclass_Self-emp"]].tolist(), [1.0, 0.0])
        self.assertEqual(projected.loc[1, ["workclass_Private", "workclass_Self-emp"]].tolist(), [0.0, 1.0])
        self.assertEqual(projected.loc[0, ["education_HS-grad", "education_Bachelors", "education_Masters"]].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(projected.loc[1, ["education_HS-grad", "education_Bachelors", "education_Masters"]].tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(encoded.loc[0, "workclass_Private"], 0.0)

    def test_dice_sample_handles_no_counterfactuals(self):
        repo_root = Path(__file__).resolve().parents[1]
        dice_sample = repo_root / "baselines" / "dice" / "sample.py"

        content = dice_sample.read_text(encoding="utf-8")

        self.assertIn("No valid DiCE counterfactuals generated", content)

    def test_dice_sample_uses_official_dice_source_tree(self):
        repo_root = Path(__file__).resolve().parents[1]
        dice_sample = repo_root / "baselines" / "dice" / "sample.py"
        official_dice = repo_root / "baselines" / "dice" / "DiCE-official"

        content = dice_sample.read_text(encoding="utf-8")

        self.assertTrue(official_dice.exists())
        self.assertIn("DiCE-official", content)
        self.assertIn("load_official_dice", content)
        self.assertIn("sys.path.insert(0, str(dice_root))", content)

    def test_dice_sample_uses_shared_optimization_path_builder(self):
        repo_root = Path(__file__).resolve().parents[1]
        dice_sample = repo_root / "baselines" / "dice" / "sample.py"

        content = dice_sample.read_text(encoding="utf-8")

        self.assertIn("get_optim_params_form_args", content)
        self.assertIn("optimization_parameters = get_optim_params_form_args(args)", content)
        self.assertNotIn('f"proximity_weight_input[{proximity_weight_input}]_hidden_dims[{hidden_dims}]"', content)

    def test_dice_sample_outputs_feature_columns_without_target(self):
        repo_root = Path(__file__).resolve().parents[1]
        dice_sample = repo_root / "baselines" / "dice" / "sample.py"

        content = dice_sample.read_text(encoding="utf-8")

        self.assertIn('column_names = info["column_names"].copy()', content)
        self.assertIn('cols = ["id"] + column_names', content)
        self.assertNotIn('cols = ["id"] + info["column_names"]', content)

    def test_dice_sample_passes_diversity_weight_to_dice(self):
        repo_root = Path(__file__).resolve().parents[1]
        dice_sample = repo_root / "baselines" / "dice" / "sample.py"

        content = dice_sample.read_text(encoding="utf-8")

        self.assertIn("diversity_weight = args.diversity_weight", content)
        self.assertIn('"diversity_weight": diversity_weight', content)

    def test_dice_sample_builds_actionability_constraints(self):
        repo_root = Path(__file__).resolve().parents[1]
        dice_sample = repo_root / "baselines" / "dice" / "sample.py"

        content = dice_sample.read_text(encoding="utf-8")

        self.assertIn("def build_dice_actionability_args", content)
        self.assertIn("features_to_vary", content)
        self.assertIn('info.get("immutable", [])', content)
        self.assertIn('info.get("monotone_increase", [])', content)
        self.assertIn('info.get("ordinal_monotone_increase", [])', content)
        self.assertIn("dice_call_args.update(", content)
        self.assertIn("build_dice_actionability_args(original_instance, info, column_names)", content)

    def test_dice_sample_projects_integers_and_saves_probabilities(self):
        repo_root = Path(__file__).resolve().parents[1]
        dice_sample = repo_root / "baselines" / "dice" / "sample.py"

        content = dice_sample.read_text(encoding="utf-8")

        self.assertIn("def project_integer_features", content)
        self.assertIn("def get_target_probabilities", content)
        self.assertIn("torch.softmax", content)
        self.assertIn("cfs = project_integer_features(cfs, info)", content)
        self.assertIn("target_probs_final = get_target_probabilities", content)

    def test_dice_sample_uses_method_specific_constructor_and_args(self):
        repo_root = Path(__file__).resolve().parents[1]
        dice_sample = repo_root / "baselines" / "dice" / "sample.py"

        content = dice_sample.read_text(encoding="utf-8")

        self.assertIn('if dice_method == "gradient":', content)
        self.assertIn("exp = dice_ml.Dice(data_dice, dice_model, method=dice_method)", content)
        self.assertNotIn("method=dice_method, verbose=verbose", content)
        self.assertIn('dice_args["maxiterations"] = max_iter', content)

    def test_dice_sample_does_not_pass_device_to_official_dice_model(self):
        repo_root = Path(__file__).resolve().parents[1]
        dice_sample = repo_root / "baselines" / "dice" / "sample.py"

        content = dice_sample.read_text(encoding="utf-8")

        self.assertNotIn("device=device", content)

    def test_dice_sample_keeps_official_dice_pytorch_model_on_cpu(self):
        repo_root = Path(__file__).resolve().parents[1]
        dice_sample = repo_root / "baselines" / "dice" / "sample.py"

        content = dice_sample.read_text(encoding="utf-8")

        self.assertIn('dice_device = "cpu"', content)
        self.assertIn("BBMLPCLF(input_shape).to(dice_device)", content)
        self.assertIn('torch.load(model_filename, map_location="cpu")', content)

    def test_dice_sample_does_not_pass_internal_index_to_official_dice(self):
        repo_root = Path(__file__).resolve().parents[1]
        dice_sample = repo_root / "baselines" / "dice" / "sample.py"

        content = dice_sample.read_text(encoding="utf-8")

        self.assertNotIn('dice_call_args["index"] = indx', content)
        self.assertIn('cfs["id"] = indx', content)

    def test_adult_mdpcf_script_contains_reproducible_pipeline(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_adult_mdpcf.ps1"

        content = script.read_text(encoding="utf-8")

        self.assertIn("param(", content)
        self.assertIn("[switch]$TrainBlackBox", content)
        self.assertIn("[switch]$TrainCSVAE", content)
        self.assertIn('"--method" "black_box"', content)
        self.assertIn('"--method" "csvae"', content)
        self.assertIn('"--method", "mdpcf"', content)
        self.assertIn('"--mode" "sample"', content)
        self.assertIn('"--mode" "evaluate"', content)
        self.assertIn('"--posthoc_sparsity", "True"', content)
        self.assertIn('"--categorical_straight_through", "True"', content)
        self.assertIn('"--get_stats" "False"', content)
        self.assertIn("Start-Transcript", content)
        self.assertIn("logs", content)
        self.assertIn("evaluation_summary.csv", content)

    def test_adult_all_methods_script_matches_tabcf_protocol(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_adult_all_methods.ps1"

        content = script.read_text(encoding="utf-8")

        self.assertIn("param(", content)
        self.assertIn('"mdpcf"', content)
        self.assertIn('"dice"', content)
        self.assertIn('"tabcf"', content)
        self.assertIn('"wachter"', content)
        self.assertIn('"revise"', content)
        self.assertIn('"cchvae"', content)
        self.assertIn("[switch]$TrainTABCF", content)
        self.assertIn('[string]$DiceMethod = "genetic"', content)
        self.assertIn("[double]$DiceProximityWeight = 0.5", content)
        self.assertIn('"--method", "tabcf"', content)
        self.assertIn('"--mode", "train"', content)
        self.assertIn('"--dice_method", "$DiceMethod"', content)
        self.assertIn('"--proximity_weight_input", "$DiceProximityWeight"', content)
        self.assertIn('"--mode" "sample"', content)
        self.assertIn('"--mode" "evaluate"', content)
        self.assertIn('"--get_stats" "False"', content)
        self.assertIn("scripts\\collect_evaluation_summaries.py", content)
        self.assertIn("results\\evaluation_summaries.csv", content)
        self.assertIn("Start-Transcript", content)
        self.assertIn("ResolvedMethods", content)
        self.assertIn("-split ','", content)

    def test_cchvae_sample_uses_local_carla_source_tree(self):
        repo_root = Path(__file__).resolve().parents[1]
        cchvae_sample = repo_root / "baselines" / "cchvae" / "sample.py"
        carla_compat = repo_root / "baselines" / "carla_compat.py"
        local_carla = repo_root / "baselines" / "CARLA"

        content = cchvae_sample.read_text(encoding="utf-8")
        compat_content = carla_compat.read_text(encoding="utf-8")

        self.assertTrue(local_carla.exists())
        self.assertIn("from baselines.carla_compat import CARLA_ROOT, TrainTestCsvCatalog", content)
        self.assertIn("load_carla_source", compat_content)
        self.assertIn("baselines\" / \"CARLA", compat_content)
        self.assertIn("DiCE-official", compat_content)
        self.assertIn("for source_root in [dice_root, carla_root]", compat_content)
        self.assertIn("sys.path.insert(0, str(source_root))", compat_content)
        self.assertIn('"tensorflow.contrib"', compat_content)
        self.assertIn("np.float = float", compat_content)
        self.assertIn("OneHotEncoder.get_feature_names", compat_content)
        self.assertIn("class TrainTestCsvCatalog", compat_content)
        self.assertNotIn("C-CHVAE-official", content)

    def test_cchvae_sample_outputs_probabilities_and_handles_empty_results(self):
        repo_root = Path(__file__).resolve().parents[1]
        cchvae_sample = repo_root / "baselines" / "cchvae" / "sample.py"

        content = cchvae_sample.read_text(encoding="utf-8")

        self.assertIn("torch.softmax", content)
        self.assertIn('column_names = info["column_names"].copy()', content)
        self.assertIn("No valid CCHVAE counterfactuals generated", content)
        self.assertIn('pd.DataFrame(columns=cols + ["target_prob"])', content)
        self.assertNotIn("exit(0)", content)

    def test_wachter_sample_outputs_probabilities_and_handles_empty_results(self):
        repo_root = Path(__file__).resolve().parents[1]
        wachter_sample = repo_root / "baselines" / "wachter" / "sample.py"
        wachter_model = (
            repo_root
            / "baselines"
            / "CARLA"
            / "carla"
            / "recourse_methods"
            / "catalog"
            / "wachter"
            / "model.py"
        )

        content = wachter_sample.read_text(encoding="utf-8")
        model_content = wachter_model.read_text(encoding="utf-8")

        self.assertIn("torch.softmax", content)
        self.assertIn('column_names = info["column_names"].copy()', content)
        self.assertIn("No valid Wachter counterfactuals generated", content)
        self.assertIn('pd.DataFrame(columns=cols + ["target_prob"])', content)
        self.assertNotIn("exit(0)", content)
        self.assertIn("negative_label=1 - target_class", model_content)

    def test_reverse_target_carla_baselines_use_target_aware_validation(self):
        repo_root = Path(__file__).resolve().parents[1]
        catalog_root = (
            repo_root
            / "baselines"
            / "CARLA"
            / "carla"
            / "recourse_methods"
            / "catalog"
        )

        revise_content = (catalog_root / "revise" / "model.py").read_text(
            encoding="utf-8"
        )
        cchvae_content = (catalog_root / "cchvae" / "model.py").read_text(
            encoding="utf-8"
        )
        revise_sample = (repo_root / "baselines" / "revise" / "sample.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("negative_label=1 - target_class", revise_content)
        self.assertIn("negative_label=source_label", cchvae_content)
        self.assertIn("No valid ReVISE counterfactuals generated", revise_sample)
        self.assertIn('pd.DataFrame(columns=cols + ["target_prob"])', revise_sample)
        self.assertNotIn("exit(0)", revise_sample)

    def test_formal_baseline_queue_runs_three_method_seeds_in_isolated_directories(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_final_baselines.ps1"
        content = script.read_text(encoding="utf-8")

        self.assertIn("[int[]]$Seeds = @(42, 43, 44)", content)
        self.assertIn("foreach ($Seed in $Seeds)", content)
        self.assertIn('"--seed", "$Seed"', content)
        self.assertIn('"seed[$Seed]"', content)
        self.assertIn("[int]$Config.seed -eq $Seed", content)

    def test_formal_baselines_use_runtime_target_source_and_black_box_protocol(self):
        repo_root = Path(__file__).resolve().parents[1]
        sample_files = [
            repo_root / "baselines" / method / "sample.py"
            for method in ("dice", "cchvae", "revise", "wachter")
        ]

        for sample_file in sample_files:
            content = sample_file.read_text(encoding="utf-8")
            self.assertIn("target_class_index = int(args.target_class)", content)
            self.assertIn("source_start", content)
            self.assertIn("select_source_indices(", content)
            self.assertIn("get_black_box_model_path(", content)
            self.assertIn("save_run_config(args, save_path)", content)

        dice_content = sample_files[0].read_text(encoding="utf-8")
        revise_content = sample_files[2].read_text(encoding="utf-8")
        wachter_content = sample_files[3].read_text(encoding="utf-8")
        self.assertIn('"desired_class": target_class_index', dice_content)
        self.assertIn(
            'hyperparameters["target_class"] = target_one_hot', revise_content
        )
        self.assertIn('hyperparameters["y_target"] = target_one_hot', wachter_content)

    def test_final_baseline_runner_covers_only_reproducible_methods(self):
        repo_root = Path(__file__).resolve().parents[1]
        runner = repo_root / "scripts" / "run_final_baselines.ps1"

        self.assertTrue(runner.is_file())
        content = runner.read_text(encoding="utf-8")
        self.assertIn('@("adult", "bank", "default", "gmc", "lending-club")', content)
        self.assertIn('@("dice", "dicoflex", "cchvae", "revise", "wachter")', content)
        self.assertIn('[string]$RunTag = "final_main_ablation_e2000"', content)
        self.assertIn('[int]$BlackBoxSeed = 42', content)
        self.assertIn('"default" { return 0 }', content)
        self.assertIn('"gmc" { return 0 }', content)
        self.assertIn('"--source_start", "0"', content)
        self.assertIn('"--target_class", "$TargetClass"', content)
        self.assertIn('[switch]$Resume', content)
        self.assertNotIn('"tabcf"', content.lower())

    def test_tabcf_original_is_registered_as_baseline_method(self):
        repo_root = Path(__file__).resolve().parents[1]
        utils_py = repo_root / "utils.py"

        content = utils_py.read_text(encoding="utf-8")

        self.assertIn('"tabcf"', content)
        self.assertIn('"train": "baselines.tabcf_original.train"', content)
        self.assertIn('"sample": "baselines.tabcf_original.sample"', content)
        self.assertIn('"evaluate": "evaluation_framework.evaluate"', content)
        self.assertIn("--reparam", content)

    def test_tabcf_original_wrapper_uses_local_tabcf_main(self):
        repo_root = Path(__file__).resolve().parents[1]
        sample_wrapper = repo_root / "baselines" / "tabcf_original" / "sample.py"
        train_wrapper = repo_root / "baselines" / "tabcf_original" / "train.py"
        loader = repo_root / "baselines" / "tabcf_original" / "_loader.py"

        sample_content = sample_wrapper.read_text(encoding="utf-8")
        train_content = train_wrapper.read_text(encoding="utf-8")
        loader_content = loader.read_text(encoding="utf-8")

        self.assertIn("TABCF-main", loader_content)
        self.assertIn("DiCE-main", loader_content)
        self.assertIn("repo_root / \"baselines\" / \"TABCF-main\"", loader_content)
        self.assertIn('tabcf_root / "baselines" / "dice" / "DiCE-main"', loader_content)
        self.assertIn("tabcf.sample", sample_content)
        self.assertIn("tabcf.vae.main", train_content)
        self.assertIn('args.device = "cpu"', sample_content)
        self.assertIn("args.gpu = -1", sample_content)

    def test_tabcf_original_loader_patches_scheduler_verbose_compatibility(self):
        repo_root = Path(__file__).resolve().parents[1]
        loader = repo_root / "baselines" / "tabcf_original" / "_loader.py"

        loader_content = loader.read_text(encoding="utf-8")

        self.assertIn("def patch_reduce_lr_on_plateau_verbose", loader_content)
        self.assertIn('kwargs.pop("verbose", None)', loader_content)
        self.assertLess(
            loader_content.index("patch_reduce_lr_on_plateau_verbose()"),
            loader_content.index("importlib.import_module(module_name)"),
        )

    def test_tabcf_original_loader_patches_dataloader_workers_for_windows(self):
        repo_root = Path(__file__).resolve().parents[1]
        loader = repo_root / "baselines" / "tabcf_original" / "_loader.py"

        loader_content = loader.read_text(encoding="utf-8")

        self.assertIn("def patch_torch_dataloader_workers", loader_content)
        self.assertIn('kwargs["num_workers"] = 0', loader_content)
        self.assertLess(
            loader_content.index("patch_torch_dataloader_workers()"),
            loader_content.index("importlib.import_module(module_name)"),
        )

    def test_tabcf_original_loader_does_not_patch_custom_dice_latent_api(self):
        repo_root = Path(__file__).resolve().parents[1]
        loader = repo_root / "baselines" / "tabcf_original" / "_loader.py"

        loader_content = loader.read_text(encoding="utf-8")

        self.assertNotIn("patch_dice_model_device_argument", loader_content)
        self.assertNotIn("patch_dice_pytorch_get_feature_weights", loader_content)

    def test_tabcf_original_loader_patches_custom_dice_object_dtype(self):
        repo_root = Path(__file__).resolve().parents[1]
        loader = repo_root / "baselines" / "tabcf_original" / "_loader.py"

        loader_content = loader.read_text(encoding="utf-8")

        self.assertIn("def patch_custom_dice_pytorch_object_dtype", loader_content)
        self.assertIn("def transform_to_float_tensor", loader_content)
        self.assertIn("astype(float)", loader_content)
        self.assertLess(
            loader_content.index("patch_custom_dice_pytorch_object_dtype()"),
            loader_content.index("importlib.import_module(module_name)"),
        )

    def test_tabcf_original_loader_patches_custom_dice_tensor_assignment(self):
        repo_root = Path(__file__).resolve().parents[1]
        loader = repo_root / "baselines" / "tabcf_original" / "_loader.py"

        loader_content = loader.read_text(encoding="utf-8")

        self.assertIn("def patch_custom_dice_tensor_assignment", loader_content)
        self.assertIn("def to_python_float", loader_content)
        self.assertIn("DicePyTorch.initialize_CFs = initialize_CFs", loader_content)
        self.assertLess(
            loader_content.index("patch_custom_dice_tensor_assignment()"),
            loader_content.index("importlib.import_module(module_name)"),
        )

    def test_evaluate_uses_tabcf_original_result_path(self):
        repo_root = Path(__file__).resolve().parents[1]
        evaluate_py = repo_root / "evaluation_framework" / "evaluate.py"

        content = evaluate_py.read_text(encoding="utf-8")

        self.assertIn('method == "tabcf"', content)
        self.assertIn("get_tabcf_original_name_form_args", content)
        self.assertIn("get_tabcf_original_optim_params_form_args", content)
        self.assertIn("TABCF", content)

    def test_evaluate_light_summary_returns_before_legacy_metrics(self):
        repo_root = Path(__file__).resolve().parents[1]
        evaluate_py = repo_root / "evaluation_framework" / "evaluate.py"

        content = evaluate_py.read_text(encoding="utf-8")

        return_pos = content.index("if not get_stats and not get_changed:")
        legacy_metrics_pos = content.index("valid_cfs = []")
        self.assertLess(return_pos, legacy_metrics_pos)


if __name__ == "__main__":
    unittest.main()
