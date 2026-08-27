"""Training orchestration with immutable split evidence and validation-only selection."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from intent_classifier.data import (
    DatasetSplits,
    IntentRecord,
    LabelHierarchy,
    dataset_sha256,
    load_intent_csv,
    split_by_normalized_text,
    write_split_membership,
)
from intent_classifier.evaluation import compute_metrics, write_evaluation_artifacts
from intent_classifier.experiments import (
    RunManifest,
    create_run_directory,
    current_git_sha,
    load_experiment_config,
    read_registry,
    runtime_information,
    select_best_run,
    write_manifest,
    write_registry,
)
from intent_classifier.reporting import (
    hash_artifacts,
    sha256_directory,
    sha256_file,
    write_json_artifact,
)


class EncodedIntentDataset:
    """Small Trainer-compatible dataset that defers padding to the batch collator."""

    def __init__(
        self,
        records: Sequence[IntentRecord],
        *,
        tokenizer: Any,
        main_to_id: Mapping[str, int],
        sub_to_id: Mapping[str, int],
        max_length: int,
    ) -> None:
        self.records = tuple(records)
        self.tokenizer = tokenizer
        self.main_to_id = main_to_id
        self.sub_to_id = sub_to_id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        encoded = self.tokenizer(
            record.text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["labels_main"] = self.main_to_id[record.main_intent]
        encoded["labels_sub"] = self.sub_to_id[record.sub_intent]
        return encoded


class MultiHeadDataCollator:
    """Pad tokenizer fields and attach both integer label tensors."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        model_features: list[dict[str, Any]] = []
        main_labels: list[int] = []
        sub_labels: list[int] = []
        for feature in features:
            item = dict(feature)
            main_labels.append(int(item.pop("labels_main")))
            sub_labels.append(int(item.pop("labels_sub")))
            model_features.append(item)
        batch = self.tokenizer.pad(model_features, padding=True, return_tensors="pt")
        batch["labels_main"] = torch.tensor(main_labels, dtype=torch.long)
        batch["labels_sub"] = torch.tensor(sub_labels, dtype=torch.long)
        return batch


def set_reproducible_seed(seed: int) -> None:
    """Seed available random-number generators without requiring optional imports."""

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _decode_predictions(
    prediction_output: Any,
    *,
    main_labels: Sequence[str],
    sub_labels: Sequence[str],
) -> tuple[list[str], list[str]]:
    import numpy as np

    raw_predictions = prediction_output.predictions
    if not isinstance(raw_predictions, tuple) or len(raw_predictions) < 2:
        raise RuntimeError("Expected separate main- and sub-intent logits from the model.")
    main_ids = np.asarray(raw_predictions[0]).argmax(axis=-1)
    sub_ids = np.asarray(raw_predictions[1]).argmax(axis=-1)
    return (
        [main_labels[int(index)] for index in main_ids],
        [sub_labels[int(index)] for index in sub_ids],
    )


def _trainer_metrics(
    main_labels: Sequence[str],
    sub_labels: Sequence[str],
    hierarchy: LabelHierarchy,
) -> Any:
    def callback(evaluation_prediction: Any) -> dict[str, float]:
        import numpy as np

        predictions = evaluation_prediction.predictions
        labels = evaluation_prediction.label_ids
        if not isinstance(predictions, tuple) or not isinstance(labels, tuple):
            raise RuntimeError("Trainer must return two prediction and two label arrays.")
        pred_main = [main_labels[int(index)] for index in np.asarray(predictions[0]).argmax(-1)]
        pred_sub = [sub_labels[int(index)] for index in np.asarray(predictions[1]).argmax(-1)]
        true_main = [main_labels[int(index)] for index in np.asarray(labels[0])]
        true_sub = [sub_labels[int(index)] for index in np.asarray(labels[1])]
        metrics = compute_metrics(true_main, pred_main, true_sub, pred_sub, hierarchy=hierarchy)
        return {
            "main_accuracy": metrics.main_accuracy,
            "sub_accuracy": metrics.sub_accuracy,
            "joint_exact_match": metrics.joint_exact_match,
            "main_macro_f1": metrics.main_macro_f1,
            "sub_macro_f1": metrics.sub_macro_f1,
            "valid_pair_rate": float(metrics.valid_pair_rate or 0.0),
        }

    return callback


def _metric_values(raw: Mapping[str, Any], prefix: str) -> dict[str, float]:
    translated: dict[str, float] = {}
    for name in (
        "main_accuracy",
        "sub_accuracy",
        "joint_exact_match",
        "main_macro_f1",
        "sub_macro_f1",
        "loss",
    ):
        source_key = f"{prefix}_{name}"
        if source_key in raw:
            translated[name] = float(raw[source_key])
    return translated


def _build_dataset(
    records: Sequence[IntentRecord],
    *,
    tokenizer: Any,
    hierarchy: LabelHierarchy,
    max_length: int,
) -> EncodedIntentDataset:
    return EncodedIntentDataset(
        records,
        tokenizer=tokenizer,
        main_to_id=hierarchy.main_to_id,
        sub_to_id=hierarchy.sub_to_id,
        max_length=max_length,
    )


def _write_aggregate_split_report(
    run_dir: Path,
    split_name: str,
    records: Sequence[IntentRecord],
    pred_main: Sequence[str],
    pred_sub: Sequence[str],
    hierarchy: LabelHierarchy,
) -> dict[str, float | None]:
    true_main = [record.main_intent for record in records]
    true_sub = [record.sub_intent for record in records]
    metrics = compute_metrics(
        true_main,
        pred_main,
        true_sub,
        pred_sub,
        hierarchy=hierarchy,
    )
    write_evaluation_artifacts(
        run_dir / "evaluation" / split_name,
        metrics=metrics,
        true_main=true_main,
        pred_main=pred_main,
        true_sub=true_sub,
        pred_sub=pred_sub,
    )
    return metrics.as_dict()


def train_experiment(
    config_path: str | Path,
    experiment_name: str,
    *,
    run_id: str | None = None,
    allow_download: bool = False,
) -> Path:
    """Execute an explicitly requested training run and preserve its complete identity.

    This function is not called by repository tests. Network access is disabled unless the caller
    deliberately passes ``allow_download=True``.
    """

    from intent_classifier.modeling import MultiHeadIntentModel, require_ml_backend

    require_ml_backend()
    from transformers import AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments

    config = load_experiment_config(config_path, experiment_name)
    set_reproducible_seed(config.split.seed)
    records = load_intent_csv(config.dataset_path)
    hierarchy = LabelHierarchy.from_json(config.hierarchy_path)
    hierarchy.validate_records(records)
    splits: DatasetSplits = split_by_normalized_text(
        records,
        train_fraction=config.split.train_fraction,
        validation_fraction=config.split.validation_fraction,
        test_fraction=config.split.test_fraction,
        seed=config.split.seed,
    )

    effective_run_id = run_id or datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = create_run_directory(config.output_root, effective_run_id)
    membership_path = run_dir / "splits" / "membership.csv"
    write_split_membership(membership_path, splits.membership)
    write_json_artifact(run_dir, "config/experiment.json", asdict(config))

    manifest = RunManifest(
        schema_version=1,
        run_id=effective_run_id,
        status="running",
        git_commit_sha=current_git_sha(),
        dataset_sha256=dataset_sha256(config.dataset_path),
        dataset_row_count=len(records),
        split_seed=config.split.seed,
        split_membership_path="splits/membership.csv",
        split_membership_sha256=sha256_file(membership_path),
        label_hierarchy_sha256=hierarchy.sha256,
        encoder_name_or_path=config.encoder_name_or_path,
        hyperparameters=asdict(config),
        runtime=runtime_information(),
    )
    write_manifest(run_dir / "run_manifest.json", manifest)

    tokenizer = AutoTokenizer.from_pretrained(
        config.encoder_name_or_path, local_files_only=not allow_download
    )
    model = MultiHeadIntentModel.from_encoder_pretrained(
        config.encoder_name_or_path,
        main_labels=hierarchy.main_labels,
        sub_labels=hierarchy.sub_labels,
        dropout=config.model.dropout,
        main_loss_weight=config.model.main_loss_weight,
        sub_loss_weight=config.model.sub_loss_weight,
        local_files_only=not allow_download,
    )
    datasets = {
        name: _build_dataset(
            values,
            tokenizer=tokenizer,
            hierarchy=hierarchy,
            max_length=config.model.max_length,
        )
        for name, values in splits.as_dict().items()
    }

    args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.train_batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model=config.training.metric_for_best_model.removeprefix("validation_"),
        greater_is_better=config.training.greater_is_better,
        seed=config.split.seed,
        data_seed=config.split.seed,
        report_to="none",
        save_safetensors=True,
        label_names=["labels_main", "labels_sub"],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=MultiHeadDataCollator(tokenizer),
        compute_metrics=_trainer_metrics(hierarchy.main_labels, hierarchy.sub_labels, hierarchy),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config.training.early_stopping_patience)
        ],
    )
    trainer.train()

    model_dir = run_dir / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    validation_output = trainer.predict(datasets["validation"], metric_key_prefix="validation")
    test_output = trainer.predict(datasets["test"], metric_key_prefix="test")
    validation_main, validation_sub = _decode_predictions(
        validation_output,
        main_labels=hierarchy.main_labels,
        sub_labels=hierarchy.sub_labels,
    )
    test_main, test_sub = _decode_predictions(
        test_output,
        main_labels=hierarchy.main_labels,
        sub_labels=hierarchy.sub_labels,
    )
    validation_metrics = _write_aggregate_split_report(
        run_dir,
        "validation",
        splits.validation,
        validation_main,
        validation_sub,
        hierarchy,
    )
    test_metrics = _write_aggregate_split_report(
        run_dir, "test", splits.test, test_main, test_sub, hierarchy
    )

    model_hash = sha256_directory(model_dir)
    evaluation_paths = [
        path.relative_to(run_dir) for path in (run_dir / "evaluation").rglob("*") if path.is_file()
    ]
    final_manifest = RunManifest(
        **{
            **manifest.to_dict(),
            "status": "complete",
            "validation_metrics": validation_metrics,
            "final_test_metrics": test_metrics,
            "model_artifact_sha256": model_hash,
            "evaluation_artifact_hashes": hash_artifacts(run_dir, evaluation_paths),
        }
    )
    write_manifest(run_dir / "run_manifest.json", final_manifest)

    registry_path = Path(config.output_root) / "registry.csv"
    rows: list[dict[str, Any]] = read_registry(registry_path) if registry_path.exists() else []
    row: dict[str, Any] = {
        "run_id": effective_run_id,
        "status": "complete",
        "validation_main_accuracy": validation_metrics["main_accuracy"],
        "validation_sub_accuracy": validation_metrics["sub_accuracy"],
        "validation_joint_exact_match": validation_metrics["joint_exact_match"],
        "validation_main_macro_f1": validation_metrics["main_macro_f1"],
        "validation_sub_macro_f1": validation_metrics["sub_macro_f1"],
        "validation_valid_pair_rate": validation_metrics["valid_pair_rate"],
        "validation_loss": _metric_values(validation_output.metrics, "validation").get("loss", ""),
        "final_test_main_accuracy": test_metrics["main_accuracy"],
        "final_test_sub_accuracy": test_metrics["sub_accuracy"],
        "final_test_joint_exact_match": test_metrics["joint_exact_match"],
        "selection_metric": config.training.metric_for_best_model,
    }
    rows = [existing for existing in rows if existing.get("run_id") != effective_run_id]
    rows.append(row)
    write_registry(registry_path, rows)
    best = select_best_run(rows, metric=config.training.metric_for_best_model)
    write_json_artifact(config.output_root, "best_run.json", dict(best))
    return run_dir


def _build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a shared-encoder multi-head intent classifier."
    )
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--experiment", default="baseline_distilbert")
    parser.add_argument("--run-id", help="Optional safe run identifier.")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit the intentionally selected encoder/tokenizer to be downloaded.",
    )
    return parser


def train_cli(argv: Sequence[str] | None = None) -> int:
    args = _build_train_parser().parse_args(argv)
    run_dir = train_experiment(
        args.config,
        args.experiment,
        run_id=args.run_id,
        allow_download=args.allow_download,
    )
    print(json.dumps({"run_directory": str(run_dir)}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(train_cli())
