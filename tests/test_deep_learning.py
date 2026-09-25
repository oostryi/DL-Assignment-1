import unittest

import torch

import numpy as np
import pandas as pd

from deep_learning import CreditRiskNet, DecoupledAdamW, FeatureGate, TabularPreprocessor, Config, build_model


class DeepLearningComponentsTest(unittest.TestCase):
    def test_feature_gate_has_trainable_parameters_and_gradients(self):
        layer = FeatureGate(4)
        output = layer(torch.ones(3, 4)).sum()
        output.backward()
        self.assertIsNotNone(layer.logits.grad)
        self.assertIsNotNone(layer.scale.grad)
        self.assertEqual(tuple(layer(torch.ones(3, 4)).shape), (3, 4))

    def test_model_output_shapes(self):
        model = CreditRiskNet(5, [3, 7], hidden_dim=32, depth=1, cross_layers=1, dropout=0.0)
        x = torch.cat([torch.randn(8, 5), torch.tensor([[1, 2]] * 8, dtype=torch.float32)], dim=1)
        self.assertEqual(tuple(model(x).shape), (8,))

    def test_single_matrix_preprocessing_and_alternative_models(self):
        frame = pd.DataFrame({"age": [20.0, np.nan, 40.0], "type": ["A", "B", None]})
        preprocessor = TabularPreprocessor(["type"], ["age"]).fit(frame)
        x = preprocessor.transform(frame)
        self.assertEqual(x.shape, (3, 2))
        self.assertTrue(np.isfinite(x).all())
        for name in ("transformer",):
            config = Config(model=name, hidden_dim=32, depth=1)
            model = build_model(config, 1, preprocessor.cardinalities)
            self.assertEqual(tuple(model(torch.from_numpy(x)).shape), (3,))

    def test_missing_indicators_are_fold_local_and_before_categories(self):
        training = pd.DataFrame({"age": [20.0, np.nan, 40.0], "income": [1.0, 2.0, 3.0], "type": ["A", "B", None]})
        validation = pd.DataFrame({"age": [np.nan, 25.0], "income": [np.nan, 4.0], "type": ["C", "A"]})
        preprocessor = TabularPreprocessor(["type"], ["age", "income"], True).fit(training)
        self.assertEqual(preprocessor.missing_indicator_columns, ["age"])
        self.assertEqual(preprocessor.num_numeric_features, 3)
        x = preprocessor.transform(validation)
        self.assertEqual(x.shape, (2, 4))
        self.assertEqual(x[:, 2].tolist(), [1.0, 0.0])
        self.assertEqual(x[0, 3], 0.0)  # Unseen category.
        self.assertTrue(np.isfinite(x).all())
        model = build_model(Config(model="transformer", hidden_dim=32, depth=1), preprocessor.num_numeric_features, preprocessor.cardinalities)
        self.assertEqual(tuple(model(torch.from_numpy(x)).shape), (2,))

    def test_piecewise_numeric_embeddings_use_training_fold_bins(self):
        training = pd.DataFrame({"amount": [0.0, 1.0, 2.0, 3.0], "constant": [5.0] * 4, "missing": [1.0, np.nan, 2.0, 3.0], "type": ["A"] * 4})
        validation = pd.DataFrame({"amount": [100.0], "constant": [5.0], "missing": [np.nan], "type": ["A"]})
        preprocessor = TabularPreprocessor(["type"], ["amount", "constant", "missing"], True, numeric_bins=4).fit(training)
        self.assertEqual(preprocessor.numeric_bin_edges.shape, (3, 5))
        self.assertEqual(preprocessor.numeric_bin_mask.shape, (3, 4))
        self.assertEqual(preprocessor.numeric_bin_mask[1].sum(), 1.0)
        self.assertLess(preprocessor.numeric_bin_edges[0].max(), 100.0)
        x = preprocessor.transform(validation)
        model = build_model(
            Config(model="transformer", hidden_dim=32, depth=1, numeric_embedding="ple", numeric_bins=4),
            preprocessor.num_numeric_features,
            preprocessor.cardinalities,
            (preprocessor.numeric_bin_edges, preprocessor.numeric_bin_mask),
        )
        result = model(torch.from_numpy(x))
        self.assertEqual(tuple(result.shape), (1,))
        self.assertEqual(model.tokenizer.num_piecewise, 3)
        self.assertEqual(model.tokenizer.numeric_weight.shape, (1, 32))
        result.sum().backward()
        self.assertTrue(torch.isfinite(model.tokenizer.piecewise_weight.grad).all())

    def test_tabm_keeps_independent_submodel_logits(self):
        training = pd.DataFrame({
            "amount": [0.0, 1.0, 2.0, 3.0],
            "sparse": [1.0, np.nan, 2.0, 3.0],
            "type": ["A", "B", "A", "B"],
        })
        preprocessor = TabularPreprocessor(
            ["type"], ["amount", "sparse"], True, numeric_bins=4
        ).fit(training)
        x = torch.from_numpy(preprocessor.transform(training))
        model = build_model(
            Config(model="tabm", hidden_dim=64, depth=2, dropout=0.1,
                   numeric_embedding="ple", numeric_bins=4, tabm_k=4),
            preprocessor.num_numeric_features,
            preprocessor.cardinalities,
            (preprocessor.numeric_bin_edges, preprocessor.numeric_bin_mask),
        )
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (4, 4))
        target = torch.tensor([0.0, 1.0, 0.0, 1.0])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target[:, None].expand_as(logits)
        )
        loss.backward()
        self.assertTrue(torch.isfinite(model.tabm.output.weight.grad).all())

    def test_custom_optimizer_updates_parameter(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = DecoupledAdamW([parameter], lr=0.01, weight_decay=0.01)
        (parameter.square().sum()).backward()
        before = parameter.detach().clone()
        optimizer.step()
        self.assertLess(parameter.item(), before.item())


if __name__ == "__main__":
    unittest.main()
