from unittest.mock import patch

import numpy as np
import torch

from mdpcf.sample import fit_latent_gmm_plausibility


def test_latent_gmm_falls_back_to_diagonal_covariance_on_singular_fit():
    real_gmm = __import__("mdpcf.sample", fromlist=["GaussianMixture"]).GaussianMixture
    calls = []

    class FailingFullGMM:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs["covariance_type"])
            self.covariance_type = kwargs["covariance_type"]
            self._model = real_gmm(*args, **kwargs)

        def fit(self, values):
            if self.covariance_type == "full":
                raise ValueError("singular covariance")
            self._model.fit(values)
            self.means_ = self._model.means_
            self.covariances_ = self._model.covariances_
            self.weights_ = self._model.weights_
            return self

    train_z = np.random.default_rng(0).normal(size=(100, 4)).astype(np.float32)
    result = None
    with patch("mdpcf.sample.GaussianMixture", FailingFullGMM):
        result = fit_latent_gmm_plausibility(
            train_z,
            n_components=3,
            max_samples=100,
            device=torch.device("cpu"),
        )

    assert calls == ["full", "diag"]
    assert result["precisions"].shape == (3, 4, 4)
