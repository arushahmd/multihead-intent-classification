from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

from intent_classifier.reporting import sha256_file


class HistoricalEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.results = cls.root / "results"

    def test_curated_metrics_are_exact(self) -> None:
        payload = json.loads(
            (self.results / "historical" / "run_002" / "common_eval_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(payload["rows"], 16050)
        self.assertEqual(payload["metrics"]["main_accuracy"], 0.9543302180685358)
        self.assertEqual(payload["metrics"]["sub_accuracy"], 0.6632398753894081)
        self.assertEqual(payload["metrics"]["joint_exact_match"], 0.6456697819314642)

        with (self.results / "historical_summary.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        internal = {
            row["metric"]: float(row["value"])
            for row in rows
            if row["evaluation_context"] == "historical_internal_split"
        }
        self.assertEqual(internal["validation_joint_accuracy"], 0.9310344827586207)
        self.assertEqual(internal["test_main_accuracy"], 0.9603448275862069)
        self.assertEqual(internal["test_sub_accuracy"], 0.9362068965517242)
        self.assertEqual(internal["test_joint_accuracy"], 0.9258620689655173)

    def test_preserved_figure_hash_is_stable(self) -> None:
        figure = self.results / "figures" / "historical_training_loss.png"
        self.assertEqual(
            sha256_file(figure),
            "7b5da27d3c01e56be9e63782e228292425a16d8200368e2fafe5751537f4a145",
        )

    def test_raw_historical_exports_are_not_distributed(self) -> None:
        prohibited_names = {
            "predictions_with_correctness.csv",
            "predictions_only.csv",
            "ground_truth_only.csv",
            "main_failures_only.csv",
            "sub_failures_only.csv",
            "joint_failures_only.csv",
        }
        found = {path.name for path in self.results.rglob("*") if path.is_file()}
        self.assertFalse(prohibited_names & found)


if __name__ == "__main__":
    unittest.main()
