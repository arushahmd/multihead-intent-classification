"""Command-line wrapper for intent-classifier training."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    from intent_classifier.training import train_cli

    return train_cli()


if __name__ == "__main__":
    raise SystemExit(main())
