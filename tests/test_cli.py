from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


class CliAndImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]

    def test_package_imports_without_loading_model_dependencies(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(self.root / "src")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import intent_classifier; print(intent_classifier.__version__)",
            ],
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "0.1.0")

    def test_all_cli_help_commands_are_offline(self) -> None:
        environment = os.environ.copy()
        environment.update({"TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"})
        for script in ("train.py", "evaluate.py", "predict.py"):
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, str(self.root / "scripts" / script), "--help"],
                    cwd=self.root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
