"""Local, device-scoped inference for serialized multi-head intent models."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from intent_classifier.data import LabelHierarchy


def _require_ml_backend() -> None:
    from intent_classifier.modeling import require_ml_backend

    require_ml_backend()


@dataclass(frozen=True, slots=True)
class IntentPrediction:
    """One decoded prediction with independent head confidence scores."""

    text: str
    main_intent: str
    sub_intent: str
    main_confidence: float
    sub_confidence: float
    invalid_hierarchy_pair: bool

    def as_dict(self) -> dict[str, str | float | bool]:
        return asdict(self)


def validate_text_inputs(texts: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize the API shape while preserving text and rejecting blank inputs."""

    values: Sequence[str] = (texts,) if isinstance(texts, str) else texts
    if not values:
        raise ValueError("At least one input text is required.")
    cleaned: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise TypeError(f"Input at position {index} is not a string.")
        if not value.strip():
            raise ValueError(f"Input at position {index} is blank.")
        cleaned.append(value)
    return tuple(cleaned)


def resolve_device(requested: str = "auto") -> str:
    """Resolve a device without changing PyTorch's process-wide default device."""

    _require_ml_backend()
    import torch

    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


class IntentPredictor:
    """Batched prediction wrapper around a tokenizer, model, and canonical hierarchy."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        hierarchy: LabelHierarchy,
        device: str = "auto",
        max_length: int = 64,
    ) -> None:
        _require_ml_backend()
        if max_length < 1:
            raise ValueError("max_length must be positive.")
        self.tokenizer = tokenizer
        self.model = model
        self.hierarchy = hierarchy
        self.device = resolve_device(device)
        self.max_length = max_length

        model_main = tuple(getattr(model.config, "main_labels", ()))
        model_sub = tuple(getattr(model.config, "sub_labels", ()))
        if model_main != hierarchy.main_labels or model_sub != hierarchy.sub_labels:
            raise ValueError("Model label mappings do not match the supplied hierarchy.")
        self.main_id_to_label = dict(enumerate(model_main))
        self.sub_id_to_label = dict(enumerate(model_sub))
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_local(
        cls,
        model_path: str | Path,
        *,
        hierarchy: LabelHierarchy,
        device: str = "auto",
        max_length: int = 64,
        local_files_only: bool = True,
    ) -> IntentPredictor:
        """Load tokenizer and model artifacts from a local directory only by default."""

        _require_ml_backend()
        from transformers import AutoTokenizer

        from intent_classifier.modeling import MultiHeadIntentModel

        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Local model directory does not exist: {path}")
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=local_files_only)
        model = MultiHeadIntentModel.from_pretrained(str(path), local_files_only=local_files_only)
        return cls(
            tokenizer=tokenizer,
            model=model,
            hierarchy=hierarchy,
            device=device,
            max_length=max_length,
        )

    def predict_one(self, text: str) -> IntentPrediction:
        return self.predict_many(text, batch_size=1)[0]

    def predict_many(
        self, texts: str | Sequence[str], *, batch_size: int = 32
    ) -> tuple[IntentPrediction, ...]:
        """Predict one text or a sequence without mutating global device or cache state."""

        import torch

        values = validate_text_inputs(texts)
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        predictions: list[IntentPrediction] = []
        for start in range(0, len(values), batch_size):
            batch = values[start : start + batch_size]
            encoded = self.tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                outputs = self.model(**encoded)
                main_probabilities = torch.softmax(outputs.main_logits, dim=-1)
                sub_probabilities = torch.softmax(outputs.sub_logits, dim=-1)
                main_confidence, main_ids = main_probabilities.max(dim=-1)
                sub_confidence, sub_ids = sub_probabilities.max(dim=-1)

            for text, main_id, sub_id, main_score, sub_score in zip(
                batch,
                main_ids.tolist(),
                sub_ids.tolist(),
                main_confidence.tolist(),
                sub_confidence.tolist(),
                strict=True,
            ):
                main_label = self.main_id_to_label[main_id]
                sub_label = self.sub_id_to_label[sub_id]
                predictions.append(
                    IntentPrediction(
                        text=text,
                        main_intent=main_label,
                        sub_intent=sub_label,
                        main_confidence=float(main_score),
                        sub_confidence=float(sub_score),
                        invalid_hierarchy_pair=not self.hierarchy.is_valid_pair(
                            main_label, sub_label
                        ),
                    )
                )
        return tuple(predictions)


def _build_predict_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict intents with a local saved model.")
    parser.add_argument("--model", required=True, help="Local saved-model directory.")
    parser.add_argument("--hierarchy", required=True, help="Canonical hierarchy JSON.")
    parser.add_argument(
        "--text", action="append", required=True, help="Text to classify; repeatable."
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def predict_cli(argv: Sequence[str] | None = None) -> int:
    args = _build_predict_parser().parse_args(argv)
    hierarchy = LabelHierarchy.from_json(args.hierarchy)
    predictor = IntentPredictor.from_local(
        args.model, hierarchy=hierarchy, device=args.device, local_files_only=True
    )
    predictions = predictor.predict_many(args.text, batch_size=args.batch_size)
    print(json.dumps([prediction.as_dict() for prediction in predictions], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(predict_cli())
