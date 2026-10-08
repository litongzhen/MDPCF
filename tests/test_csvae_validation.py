import torch

from mdpcf.csvae.main import validation_objective


def test_csvae_validation_objective_matches_training_objective():
    result = validation_objective(
        "csvae",
        torch.tensor(1.0),
        torch.tensor(2.0),
        torch.tensor(3.0),
        torch.tensor(4.0),
        torch.tensor(5.0),
        0.1,
    )

    assert result.item() == torch.tensor(8.7).item()


def test_standard_vae_validation_objective_excludes_csvae_terms():
    result = validation_objective(
        "standard_vae",
        torch.tensor(1.0),
        torch.tensor(2.0),
        torch.tensor(3.0),
        torch.tensor(400.0),
        torch.tensor(500.0),
        0.1,
    )

    assert result.item() == torch.tensor(3.3).item()


def test_csvae_validation_objective_supports_stronger_w_kl():
    result = validation_objective(
        "csvae",
        torch.tensor(1.0),
        torch.tensor(2.0),
        torch.tensor(3.0),
        torch.tensor(4.0),
        torch.tensor(5.0),
        0.1,
        w_kl_multiplier=5.0,
    )

    assert result.item() == torch.tensor(10.3).item()


def test_crud_alternating_validation_uses_confusion_not_classifier_ce():
    result = validation_objective(
        "csvae",
        torch.tensor(1.0),
        torch.tensor(2.0),
        torch.tensor(3.0),
        torch.tensor(4.0),
        torch.tensor(500.0),
        0.1,
        adversary_training="crud_alternating",
        adversary_confusion=torch.tensor(-0.6),
        adversary_weight=2.0,
    )

    assert result.item() == torch.tensor(2.5).item()
