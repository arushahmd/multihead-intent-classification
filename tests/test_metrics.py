from __future__ import annotations

import unittest

from intent_classifier.data import LabelHierarchy
from intent_classifier.evaluation import (
    classification_report_rows,
    compute_metrics,
    confusion_matrix_rows,
    macro_f1,
)


class MetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hierarchy = LabelHierarchy.from_mapping(
            {"cart": ["add", "remove"], "menu": ["browse"]}
        )

    def test_main_sub_joint_and_valid_pair_metrics(self) -> None:
        metrics = compute_metrics(
            ["cart", "cart", "menu", "menu"],
            ["cart", "cart", "menu", "cart"],
            ["add", "remove", "browse", "browse"],
            ["add", "add", "browse", "browse"],
            hierarchy=self.hierarchy,
        )
        self.assertAlmostEqual(metrics.main_accuracy, 0.75)
        self.assertAlmostEqual(metrics.sub_accuracy, 0.75)
        self.assertAlmostEqual(metrics.joint_exact_match, 0.5)
        self.assertAlmostEqual(metrics.valid_pair_rate or 0.0, 0.75)

    def test_macro_f1_counts_false_positive_only_labels(self) -> None:
        self.assertAlmostEqual(macro_f1(["a", "a"], ["a", "b"]), 1 / 3)

    def test_confusion_and_report_are_deterministic(self) -> None:
        labels, matrix = confusion_matrix_rows(["b", "a", "a"], ["a", "a", "b"])
        self.assertEqual(labels, ["a", "b"])
        self.assertEqual(matrix, [[1, 1], [1, 0]])
        report = classification_report_rows(["b", "a", "a"], ["a", "a", "b"])
        self.assertEqual([row["label"] for row in report], ["a", "b"])

    def test_empty_or_mismatched_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compute_metrics([], [], [], [])
        with self.assertRaises(ValueError):
            compute_metrics(["a"], ["a", "b"], ["x"], ["x"])


if __name__ == "__main__":
    unittest.main()
