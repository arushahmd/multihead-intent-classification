"""Experiment configuration, manifests, registry, and validation-only model selection."""

from __future__ import annotations

import csv
import json
import platform
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ALLOWED_SELECTION_METRICS = {
    "validation_main_accuracy",
    "validation_sub_accuracy",
    "validation_joint_exact_match",
    "validation_main_macro_f1",
    "validation_sub_macro_f1",
    "validation_valid_pair_rate",
}


@dataclass(frozen=True, slots=True)
class SplitConfig:
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    group_key: str = "normalized_text"

    def validate(self) -> None:
        if abs(self.train_fraction + self.validation_fraction + self.test_fraction - 1) > 1e-9:
            raise ValueError("Split fractions must sum to 1.0.")
        if min(self.train_fraction, self.validation_fraction, self.test_fraction) <= 0:
            raise ValueError("Split fractions must be positive.")
        if self.group_key != "normalized_text":
            raise ValueError("The public pipeline requires group_key: normalized_text.")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    max_length: int = 64
    dropout: float = 0.20
    main_loss_weight: float = 1.0
    sub_loss_weight: float = 1.0

    def validate(self) -> None:
        if self.max_length < 1:
            raise ValueError("max_length must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in the interval [0, 1).")
        if min(self.main_loss_weight, self.sub_loss_weight) < 0:
            raise ValueError("Loss weights cannot be negative.")
        if self.main_loss_weight == self.sub_loss_weight == 0:
            raise ValueError("At least one loss weight must be positive.")


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    epochs: int = 6
    train_batch_size: int = 16
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 2
    metric_for_best_model: str = "validation_joint_exact_match"
    greater_is_better: bool = True

    def validate(self) -> None:
        if self.metric_for_best_model not in _ALLOWED_SELECTION_METRICS:
            raise ValueError(
                "Best-run selection must use an approved validation metric, not test/report data."
            )
        if not self.greater_is_better:
            raise ValueError("Approved validation selection metrics must be maximized.")
        positive = (
            self.learning_rate,
            self.epochs,
            self.train_batch_size,
            self.eval_batch_size,
            self.gradient_accumulation_steps,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Learning rate, epochs, and batch controls must be positive.")
        if self.weight_decay < 0 or not 0 <= self.warmup_ratio < 1:
            raise ValueError("Invalid weight_decay or warmup_ratio.")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    description: str
    encoder_name_or_path: str
    dataset_path: str
    hierarchy_path: str
    output_root: str
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: OptimizationConfig = field(default_factory=OptimizationConfig)

    def validate(self) -> None:
        sanitize_identifier(self.name)
        for name, value in (
            ("encoder_name_or_path", self.encoder_name_or_path),
            ("dataset_path", self.dataset_path),
            ("hierarchy_path", self.hierarchy_path),
            ("output_root", self.output_root),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} cannot be blank.")
        self.split.validate()
        self.model.validate()
        self.training.validate()


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Evidence needed to interpret and reproduce a future experiment run."""

    schema_version: int
    run_id: str
    status: str
    git_commit_sha: str | None
    dataset_sha256: str
    dataset_row_count: int
    split_seed: int
    split_membership_path: str
    split_membership_sha256: str
    label_hierarchy_sha256: str
    encoder_name_or_path: str
    hyperparameters: Mapping[str, Any]
    runtime: Mapping[str, Any]
    validation_metrics: Mapping[str, float] = field(default_factory=dict)
    final_test_metrics: Mapping[str, float] = field(default_factory=dict)
    model_artifact_sha256: str | None = None
    evaluation_artifact_hashes: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sanitize_identifier(value: str) -> str:
    """Validate run identifiers before using them in paths or registries."""

    candidate = str(value).strip()
    if not _SAFE_IDENTIFIER.fullmatch(candidate) or candidate in {".", ".."}:
        raise ValueError(
            "Identifiers must contain only letters, numbers, '.', '_', or '-' and cannot traverse."
        )
    return candidate


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"'{field_name}' must be a mapping.")
    return value


def load_experiment_config(path: str | Path, name: str) -> ExperimentConfig:
    """Load one named YAML experiment and validate model-selection policy."""

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Reading YAML requires the declared PyYAML dependency.") from exc

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    experiments = _mapping(payload, "root").get("experiments")
    experiments = _mapping(experiments, "experiments")
    if name not in experiments:
        raise KeyError(f"Unknown experiment '{name}'.")
    raw = _mapping(experiments[name], name)
    split = _mapping(raw.get("split", {}), "split")
    model = _mapping(raw.get("model", {}), "model")
    training = _mapping(raw.get("training", {}), "training")
    config = ExperimentConfig(
        name=name,
        description=str(raw.get("description", "")),
        encoder_name_or_path=str(raw.get("encoder_name_or_path", "")),
        dataset_path=str(raw.get("dataset_path", "")),
        hierarchy_path=str(raw.get("hierarchy_path", "")),
        output_root=str(raw.get("output_root", "")),
        split=SplitConfig(**split),
        model=ModelConfig(**model),
        training=OptimizationConfig(**training),
    )
    config.validate()
    return config


def create_run_directory(output_root: str | Path, run_id: str) -> Path:
    """Create an isolated run directory with stable artifact subdirectories."""

    safe_id = sanitize_identifier(run_id)
    root = Path(output_root)
    run_dir = root / safe_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for child in ("config", "splits", "model", "evaluation"):
        (run_dir / child).mkdir()
    return run_dir


def current_git_sha(repository: str | Path = ".") -> str | None:
    """Return the current commit SHA, or None in a repository without commits."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(repository),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def runtime_information() -> dict[str, Any]:
    """Collect compact interpreter, platform, and relevant dependency versions."""

    package_names = ("accelerate", "numpy", "PyYAML", "torch", "transformers")
    versions: dict[str, str] = {}
    for package in package_names:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": versions,
    }


def write_manifest(path: str | Path, manifest: RunManifest) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def select_best_run(
    rows: Iterable[Mapping[str, Any]],
    *,
    metric: str = "validation_joint_exact_match",
) -> Mapping[str, Any]:
    """Select a run using validation evidence only; final/test metrics are report-only."""

    if metric not in _ALLOWED_SELECTION_METRICS:
        raise ValueError(
            f"Selection metric '{metric}' is not allowed. Use a validation/model-selection metric."
        )
    candidates = list(rows)
    if not candidates:
        raise ValueError("Cannot select a best run from an empty registry.")
    for row in candidates:
        if metric not in row or row[metric] in {None, ""}:
            raise ValueError(f"Registry row is missing selection metric '{metric}'.")

    def rank_key(row: Mapping[str, Any]) -> tuple[float, float, str]:
        score = float(row[metric])
        raw_loss = row.get("validation_loss")
        validation_loss = float(raw_loss) if raw_loss not in {None, ""} else float("inf")
        return (-score, validation_loss, str(row.get("run_id", "")))

    return min(candidates, key=rank_key)


def write_registry(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write a deterministic registry while retaining report-only test columns."""

    records = list(rows)
    if not records:
        raise ValueError("Registry must contain at least one run.")
    fieldnames = sorted({key for record in records for key in record})
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(records, key=lambda value: str(value.get("run_id", ""))):
            writer.writerow(row)


def read_registry(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def allowed_selection_metrics() -> frozenset[str]:
    return frozenset(_ALLOWED_SELECTION_METRICS)
