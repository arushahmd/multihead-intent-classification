from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from intent_classifier.experiments import (
    ExperimentConfig,
    OptimizationConfig,
    load_experiment_config,
    select_best_run,
)


class ExperimentTests(unittest.TestCase):
    def test_default_configuration_is_valid(self) -> None:
        config = ExperimentConfig(
            name="baseline",
            description="test",
            encoder_name_or_path="local-encoder",
            dataset_path="data.csv",
            hierarchy_path="hierarchy.json",
            output_root="runs",
        )
        config.validate()

    def test_test_or_final_metric_cannot_select_best_run(self) -> None:
        rows = [
            {
                "run_id": "a",
                "validation_joint_exact_match": 0.7,
                "final_test_joint_exact_match": 0.9,
            }
        ]
        for metric in ("final_test_joint_exact_match", "test_joint_exact_match"):
            with self.subTest(metric=metric), self.assertRaises(ValueError):
                select_best_run(rows, metric=metric)
        with self.assertRaises(ValueError):
            OptimizationConfig(metric_for_best_model="final_test_joint_exact_match").validate()

    def test_best_run_uses_validation_then_validation_loss(self) -> None:
        rows = [
            {
                "run_id": "first",
                "validation_joint_exact_match": 0.8,
                "validation_loss": 0.4,
                "final_test_joint_exact_match": 0.99,
            },
            {
                "run_id": "second",
                "validation_joint_exact_match": 0.8,
                "validation_loss": 0.3,
                "final_test_joint_exact_match": 0.1,
            },
        ]
        self.assertEqual(select_best_run(rows)["run_id"], "second")

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML is not installed")
    def test_repository_yaml_loads(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_experiment_config(
            root / "configs" / "experiments.yaml", "baseline_distilbert"
        )
        self.assertEqual(config.split.group_key, "normalized_text")
        self.assertEqual(config.training.metric_for_best_model, "validation_joint_exact_match")


if __name__ == "__main__":
    unittest.main()
