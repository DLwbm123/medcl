"""Synthetic hand-calculated checks only; no real labels, files, or models."""

import json
import math
import unittest

import numpy as np

from medcl.metrics import continual_summary, federated_summary, score_cases


class KernelChecks(unittest.TestCase):
    def test_classification_cases_and_client_weighting(self):
        prediction = np.array([1] + [0] * 9)
        target = np.ones(10, dtype=int)
        cases = score_cases("classification", prediction, target, [(0, 1), (1, 10), (10, 10)])
        self.assertEqual([case["score"] for case in cases], [1.0, 0.0, None])
        self.assertEqual([case["n_samples"] for case in cases], [1, 9, 0])
        summary = federated_summary([1.0, 0.0, 1.0, None], [1, 9, 0, 3], "classification")
        self.assertEqual(summary["client_macro"], 0.5)
        self.assertEqual(summary["sample_weighted_accuracy"], 0.1)
        self.assertEqual(summary["worst_client"], 0.0)
        self.assertAlmostEqual(summary["client_std"], 0.5)
        self.assertEqual(summary["client_gap"], 1.0)
        self.assertEqual((summary["available_clients"], summary["total_clients"]), (2, 4))

    def test_dice_empty_and_case_voxel_aggregation(self):
        prediction = np.zeros((2, 2, 2), dtype=int)
        target = prediction.copy()
        prediction[1, 0, 0] = 1
        cases = score_cases("segmentation", prediction, target, [(0, 0), (0, 1), (1, 2)], classes=(1, 2))
        self.assertIsNone(cases[0]["score"])
        self.assertIsNone(cases[0]["per_class"])
        self.assertEqual(cases[1]["per_class"], {"1": 1.0, "2": 1.0})
        self.assertAlmostEqual(cases[2]["per_class"]["1"], 1e-5 / (1 + 1e-5))
        self.assertAlmostEqual(cases[2]["score"], (1 + 1e-5 / (1 + 1e-5)) / 2)
        prediction = np.array([[[1, 1], [1, 1]], [[0, 0], [0, 0]]])
        target = np.array([[[1, 0], [0, 0]], [[1, 0], [0, 0]]])
        cases = score_cases("segmentation", prediction, target, [(0, 2)], classes=(1,))
        self.assertAlmostEqual(cases[0]["score"], (2 + 1e-5) / (6 + 1e-5))
        included = score_cases("segmentation", prediction, target, [(0, 2)], classes=(0, 1))[0]
        background = (6 + 1e-5) / (10 + 1e-5)
        self.assertAlmostEqual(included["score"], (background + (2 + 1e-5) / (6 + 1e-5)) / 2)
        self.assertEqual(score_cases("segmentation", np.zeros((1, 2, 2, 2), dtype=int),
                                     np.zeros((1, 2, 2, 2), dtype=int), [(0, 1)], classes=(1,))[0]["score"], 1.0)

    def test_tre_spacing_and_lower_is_better(self):
        prediction = np.array([[[3.0, 2.0], [0.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]])
        cases = score_cases("registration", prediction, np.zeros_like(prediction), [(0, 2), (2, 2)], spacing=(1.0, 2.0))
        self.assertEqual(cases[0]["score"], 2.0)  # Distances: 5, 0, 2, 1 mm.
        self.assertIsNone(cases[1]["score"])
        summary = federated_summary([2.0, 4.0], [2, 5], "registration", direction="lower")
        self.assertEqual(summary["client_macro"], 3.0)
        self.assertEqual(summary["worst_client"], 4.0)
        self.assertAlmostEqual(summary["client_std"], 1.0)
        self.assertIsNone(summary["sample_weighted_accuracy"])
        matrix = [[10.0, 18.0, None], [9.0, 20.0, 28.0], [8.0, 18.0, 30.0]]
        summary = continual_summary(matrix, "lower", [10.0, 20.0, 30.0], True)
        self.assertAlmostEqual(summary["Final average"]["value"], 56 / 3)
        self.assertEqual(summary["BWT"]["value"], 2.0)
        self.assertEqual(summary["Forgetting"]["value"], -1.5)
        self.assertEqual(summary["FWT"]["value"], 2.0)
        self.assertIsNone(summary["BWTR"]["value"])

    def test_custom_task_order_is_explicit_matrix_reordering(self):
        # Stored columns A,B,C; acquisition order C,A,B. Rows are stage order.
        stored = np.array([[0.1, 0.2, 0.8], [0.7, 0.2, 0.7], [0.6, 0.9, 0.6]])
        ordered = stored[:, [2, 0, 1]]
        summary = continual_summary(ordered, random_baseline=[0.1, 0.1, 0.1], allow_unseen=True)
        self.assertAlmostEqual(summary["Final average"]["value"], 0.7)
        self.assertAlmostEqual(summary["BWT"]["value"], -0.15)
        self.assertAlmostEqual(summary["Forgetting"]["value"], 0.15)
        self.assertAlmostEqual(summary["FWT"]["value"], 0.05)
        self.assertAlmostEqual(summary["BWTR"]["value"], (-0.2 / 0.8 - 0.1 / 0.7) / 2)

    def test_missing_stages_are_not_compacted_or_imputed(self):
        summary = continual_summary([[0.8, None, None], [None, 0.7, None], [0.9, 0.8, 0.6]])
        self.assertAlmostEqual(summary["BWT"]["value"], 0.1)
        self.assertIsNone(summary["Forgetting"]["value"])
        missing_stage = continual_summary([[0.8, None, None], [None, None, None], [0.6, 0.7, 0.9]])
        self.assertAlmostEqual(missing_stage["Final average"]["value"], 2.2 / 3)
        self.assertIsNone(missing_stage["BWT"]["value"])
        self.assertIsNone(missing_stage["Forgetting"]["value"])
        missing_final = continual_summary([[0.8, None], [None, 0.7]])
        self.assertIsNone(missing_final["Final average"]["value"])
        improved = continual_summary([[0.5, None], [0.6, 0.7]])
        self.assertAlmostEqual(improved["Forgetting"]["value"], -0.1)

    def test_gates_and_no_data(self):
        for kwargs in ({}, {"allow_unseen": True}, {"allow_unseen": True, "random_baseline": [0.1, None]}):
            self.assertIsNone(continual_summary([[0.5, 0.2], [0.6, 0.7]], **kwargs)["FWT"]["value"])
        self.assertIsNone(continual_summary([[0.5, None], [0.6, 0.7]], random_baseline=[0.1, 0.1], allow_unseen=True)["FWT"]["value"])
        self.assertIsNone(continual_summary([[0.0, None], [0.6, 0.7]])["BWTR"]["value"])
        empty = federated_summary([None, 0.8], [2, 0], "segmentation")
        self.assertEqual((empty["available_clients"], empty["total_clients"]), (0, 2))
        self.assertTrue(all(empty[key] is None for key in ("client_macro", "sample_weighted_accuracy", "worst_client", "client_std", "client_gap")))
        self.assertTrue(all(item["value"] is None for item in continual_summary([]).values()))
        self.assertEqual(continual_summary([[0.8]])["Final average"]["value"], 0.8)
        self.assertIsNone(continual_summary([[0.8]])["BWT"]["value"])
        self.assertEqual(score_cases("classification", np.array([], dtype=int), np.array([], dtype=int), [(0, 0)])[0]["n_samples"], 0)
        json.dumps({"federated": empty, "continual": continual_summary([[None]])}, allow_nan=False)

    def test_invalid_inputs(self):
        invalid_calls = [
            lambda: score_cases("classification", [1.0], [1.0], [(0, 1)]),
            lambda: score_cases("classification", [[1]], [[1]], [(0, 1)]),
            lambda: score_cases("classification", [1, 2], [1], [(0, 1)]),
            lambda: score_cases("classification", [1, 2], [1, 2], [(0, 1)]),
            lambda: score_cases("classification", [1, 2], [1, 2], [(0, 1), (0, 2)]),
            lambda: score_cases("classification", [1], [1], [(False, 1)]),
            lambda: score_cases("segmentation", np.zeros((1, 2, 2), dtype=int), np.zeros((1, 2, 2), dtype=int), [(0, 1)]),
            lambda: score_cases("segmentation", np.zeros((1, 2, 2), dtype=int), np.zeros((1, 2, 2), dtype=int), [(0, 1)], classes=(1, 1)),
            lambda: score_cases("registration", np.zeros((1, 1, 2)), np.zeros((1, 1, 2)), [(0, 1)]),
            lambda: score_cases("registration", np.zeros((1, 1, 2)), np.zeros((1, 1, 2)), [(0, 1)], spacing=(1, 0)),
            lambda: score_cases("registration", np.array([[[math.nan, 1.0]]]), np.zeros((1, 1, 2)), [(0, 1)], spacing=(1, 1)),
            lambda: continual_summary([[0.1, 0.2]]),
            lambda: continual_summary([[math.nan]]),
            lambda: continual_summary([[True]]),
            lambda: continual_summary([[0.1]], direction="automatic"),
            lambda: continual_summary([[0.1]], random_baseline=[]),
            lambda: continual_summary([[0.1]], allow_unseen="true"),
            lambda: federated_summary([math.inf], [0], "classification"),
            lambda: federated_summary([0.5], [1.0], "classification"),
            lambda: federated_summary([0.5], [-1], "classification"),
            lambda: federated_summary([0.5], [True], "classification"),
            lambda: federated_summary([0.5], [1, 2], "classification"),
        ]
        for index, call in enumerate(invalid_calls):
            with self.subTest(index=index), self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
