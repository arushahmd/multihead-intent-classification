from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from intent_classifier.reporting import (
    safe_output_path,
    sha256_directory,
    sha256_file,
    write_json_artifact,
)


class ReportingTests(unittest.TestCase):
    def test_safe_relative_path_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = write_json_artifact(directory, "reports/metrics.json", {"score": 1.0})
            self.assertTrue(output.is_file())
            self.assertEqual(json.loads(output.read_text())["score"], 1.0)
            self.assertEqual(len(sha256_file(output)), 64)
            first_hash = sha256_directory(directory)
            self.assertEqual(first_hash, sha256_directory(directory))

    def test_absolute_and_traversing_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                safe_output_path(directory, Path(directory).resolve() / "outside.json")
            with self.assertRaises(ValueError):
                safe_output_path(directory, "../outside.json")


if __name__ == "__main__":
    unittest.main()
