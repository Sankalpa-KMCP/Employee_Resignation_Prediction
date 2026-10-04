"""Checks for the custom AP stopping loop and preprocessing boundaries."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.models.mlp_experiment import make_preprocessor, refit, train_candidate, training_weights


class EpochModel:
    def __init__(self):
        self.calls = 0
        self.loss_ = 1.0

    def partial_fit(self, *args, **kwargs):
        self.calls += 1
        return self

    def predict_proba(self, x):
        return np.tile([.9, .1], (len(x), 1))


class MLPExperimentTests(unittest.TestCase):
    def test_ap_stopping_and_fresh_refit_length(self):
        config = {"fixed": {"max_epochs": 10, "patience": 2, "minimum_ap_improvement": 1e-6}}
        candidate = {"weighting": "none"}
        x, y = np.ones((4, 2)), np.array([0, 0, 0, 1])
        selection_model, fresh_model = EpochModel(), EpochModel()
        with patch("src.models.mlp_experiment.make_model", side_effect=[selection_model, fresh_model]), \
             patch("src.models.mlp_experiment.average_precision_score", side_effect=[.10, .14, .13, .12]):
            ap, epoch, history = train_candidate(x, y, x, y, candidate, config)
            model, losses = refit(x, y, candidate, config, epoch)
        self.assertEqual((ap, epoch, len(history)), (.14, 2, 4))
        self.assertEqual(selection_model.calls, 4)
        self.assertIs(model, fresh_model)
        self.assertEqual((fresh_model.calls, len(losses)), (2, 2))

    def test_training_statistics_and_unknown_category(self):
        prep = make_preprocessor(["number", "category"], ["category"])
        prep.fit(pd.DataFrame({"number": [1., 3.], "category": ["a", "b"]}))
        transformed = prep.transform(pd.DataFrame({"number": [100.], "category": ["new"]}))
        self.assertEqual(prep.named_transformers_["numerical"].mean_[0], 2.)
        np.testing.assert_allclose(transformed, [[98., 0., 0.]])

    def test_balanced_weights_preserve_mean_and_class_ratio(self):
        y = np.array([0] * 90 + [1] * 10)
        weights = training_weights(y, "balanced")
        self.assertAlmostEqual(weights.mean(), 1.)
        self.assertAlmostEqual(weights[y == 1][0] / weights[y == 0][0], 9.)
        self.assertIsNone(training_weights(y, "none"))


if __name__ == "__main__":
    unittest.main()
