"""Reusable hierarchical intent metrics and evaluation reporting."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from intent_classifier.data import LabelHierarchy, load_intent_csv


@dataclass(frozen=True, slots=True)
class IntentMetrics:
    """Aggregate metrics for two independent prediction heads."""

    main_accuracy: float
    sub_accuracy: float
    joint_exact_match: float
    main_macro_f1: float
    sub_macro_f1: float
    valid_pair_rate: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


def _validate_equal_nonempty(**sequences: Sequence[str]) -> int:
    lengths = {name: len(values) for name, values in sequences.items()}
    if not lengths or next(iter(lengths.values())) == 0:
        raise ValueError("Metric inputs must contain at least one example.")
    if len(set(lengths.values())) != 1:
        rendered = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"Metric inputs must have equal lengths ({rendered}).")
    return next(iter(lengths.values()))


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    count = _validate_equal_nonempty(y_true=y_true, y_pred=y_pred)
    correct = sum(truth == prediction for truth, prediction in zip(y_true, y_pred, strict=True))
    return correct / count


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Compute unweighted per-label F1 over labels present in truth or predictions."""

    _validate_equal_nonempty(y_true=y_true, y_pred=y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    scores: list[float] = []
    for label in labels:
        true_positive = sum(
            truth == label and prediction == label
            for truth, prediction in zip(y_true, y_pred, strict=True)
        )
        false_positive = sum(
            truth != label and prediction == label
            for truth, prediction in zip(y_true, y_pred, strict=True)
        )
        false_negative = sum(
            truth == label and prediction != label
            for truth, prediction in zip(y_true, y_pred, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else (2 * true_positive) / denominator)
    return sum(scores) / len(scores)


def compute_metrics(
    true_main: Sequence[str],
    pred_main: Sequence[str],
    true_sub: Sequence[str],
    pred_sub: Sequence[str],
    *,
    hierarchy: LabelHierarchy | None = None,
) -> IntentMetrics:
    """Compute main, sub, and joint metrics from string labels."""

    count = _validate_equal_nonempty(
        true_main=true_main,
        pred_main=pred_main,
        true_sub=true_sub,
        pred_sub=pred_sub,
    )
    joint = (
        sum(
            true_main_value == pred_main_value and true_sub_value == pred_sub_value
            for true_main_value, pred_main_value, true_sub_value, pred_sub_value in zip(
                true_main, pred_main, true_sub, pred_sub, strict=True
            )
        )
        / count
    )
    valid_pair_rate = None
    if hierarchy is not None:
        valid_pair_rate = (
            sum(
                hierarchy.is_valid_pair(main, sub)
                for main, sub in zip(pred_main, pred_sub, strict=True)
            )
            / count
        )
    return IntentMetrics(
        main_accuracy=accuracy(true_main, pred_main),
        sub_accuracy=accuracy(true_sub, pred_sub),
        joint_exact_match=joint,
        main_macro_f1=macro_f1(true_main, pred_main),
        sub_macro_f1=macro_f1(true_sub, pred_sub),
        valid_pair_rate=valid_pair_rate,
    )


def confusion_matrix_rows(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Iterable[str] | None = None
) -> tuple[list[str], list[list[int]]]:
    """Build a deterministic confusion matrix without a scikit-learn dependency."""

    _validate_equal_nonempty(y_true=y_true, y_pred=y_pred)
    ordered = sorted(set(labels) if labels is not None else set(y_true) | set(y_pred))
    index = {label: position for position, label in enumerate(ordered)}
    matrix = [[0 for _ in ordered] for _ in ordered]
    for truth, prediction in zip(y_true, y_pred, strict=True):
        if truth not in index or prediction not in index:
            raise ValueError("Provided labels do not cover all true and predicted values.")
        matrix[index[truth]][index[prediction]] += 1
    return ordered, matrix


def classification_report_rows(
    y_true: Sequence[str], y_pred: Sequence[str]
) -> list[dict[str, str | int | float]]:
    """Return deterministic per-label precision, recall, F1, and support rows."""

    _validate_equal_nonempty(y_true=y_true, y_pred=y_pred)
    true_counts = Counter(y_true)
    predicted_counts = Counter(y_pred)
    rows: list[dict[str, str | int | float]] = []
    for label in sorted(set(y_true) | set(y_pred)):
        true_positive = sum(
            truth == label and prediction == label
            for truth, prediction in zip(y_true, y_pred, strict=True)
        )
        precision = true_positive / predicted_counts[label] if predicted_counts[label] else 0.0
        recall = true_positive / true_counts[label] if true_counts[label] else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": true_counts[label],
            }
        )
    return rows


def write_evaluation_artifacts(
    output_dir: str | Path,
    *,
    metrics: IntentMetrics,
    true_main: Sequence[str],
    pred_main: Sequence[str],
    true_sub: Sequence[str],
    pred_sub: Sequence[str],
) -> None:
    """Write compact aggregate artifacts for a newly evaluated model."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(metrics.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, truth, prediction in (
        ("main", true_main, pred_main),
        ("sub", true_sub, pred_sub),
    ):
        report_path = output / f"classification_report_{name}.csv"
        with report_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("label", "precision", "recall", "f1_score", "support")
            )
            writer.writeheader()
            writer.writerows(classification_report_rows(truth, prediction))

        labels, matrix = confusion_matrix_rows(truth, prediction)
        with (output / f"confusion_{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true_label", *labels])
            for label, row in zip(labels, matrix, strict=True):
                writer.writerow([label, *row])


def _build_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a local multi-head intent model against a labeled CSV."
    )
    parser.add_argument("--model", required=True, help="Local saved-model directory.")
    parser.add_argument("--data", required=True, help="CSV with text/main_intent/sub_intent.")
    parser.add_argument("--hierarchy", required=True, help="Canonical hierarchy JSON.")
    parser.add_argument("--output", required=True, help="Directory for aggregate reports.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def evaluate_cli(argv: Sequence[str] | None = None) -> int:
    """Console entry point; model dependencies are imported only after argument parsing."""

    args = _build_evaluate_parser().parse_args(argv)
    from intent_classifier.inference import IntentPredictor

    records = load_intent_csv(args.data)
    hierarchy = LabelHierarchy.from_json(args.hierarchy)
    hierarchy.validate_records(records)
    predictor = IntentPredictor.from_local(
        args.model, hierarchy=hierarchy, device=args.device, local_files_only=True
    )
    predictions = predictor.predict_many(
        [record.text for record in records], batch_size=args.batch_size
    )
    metrics = compute_metrics(
        [record.main_intent for record in records],
        [prediction.main_intent for prediction in predictions],
        [record.sub_intent for record in records],
        [prediction.sub_intent for prediction in predictions],
        hierarchy=hierarchy,
    )
    write_evaluation_artifacts(
        args.output,
        metrics=metrics,
        true_main=[record.main_intent for record in records],
        pred_main=[prediction.main_intent for prediction in predictions],
        true_sub=[record.sub_intent for record in records],
        pred_sub=[prediction.sub_intent for prediction in predictions],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(evaluate_cli())
