"""Hugging Face-compatible shared-encoder, two-head intent model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import nn
    from torch.nn import CrossEntropyLoss
    from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
    from transformers.utils import ModelOutput
except ImportError as exc:  # Keep lightweight modules importable for data/report tooling.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    CrossEntropyLoss = None  # type: ignore[assignment]
    AutoConfig = AutoModel = None  # type: ignore[assignment]
    PretrainedConfig = PreTrainedModel = object  # type: ignore[assignment,misc]
    ModelOutput = object  # type: ignore[assignment,misc]
    _BACKEND_IMPORT_ERROR: ImportError | None = exc
else:
    _BACKEND_IMPORT_ERROR = None


def is_ml_backend_available() -> bool:
    """Return whether PyTorch and Transformers imported successfully."""

    return _BACKEND_IMPORT_ERROR is None


def require_ml_backend() -> None:
    """Fail with an actionable message when model dependencies are unavailable."""

    if _BACKEND_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Model operations require the declared 'torch' and 'transformers' dependencies."
        ) from _BACKEND_IMPORT_ERROR


if is_ml_backend_available():

    class MultiHeadIntentConfig(PretrainedConfig):
        """Serializable configuration for an encoder and two independent label heads."""

        model_type = "multihead-intent-classifier"

        def __init__(
            self,
            *,
            encoder_config: Mapping[str, Any] | None = None,
            encoder_name_or_path: str | None = None,
            main_labels: Sequence[str] = (),
            sub_labels: Sequence[str] = (),
            dropout: float = 0.2,
            main_loss_weight: float = 1.0,
            sub_loss_weight: float = 1.0,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            if not 0 <= dropout < 1:
                raise ValueError("dropout must be in the interval [0, 1).")
            if main_loss_weight < 0 or sub_loss_weight < 0:
                raise ValueError("Loss weights must be non-negative.")
            if main_loss_weight == 0 and sub_loss_weight == 0:
                raise ValueError("At least one loss weight must be positive.")

            self.encoder_config = dict(encoder_config or {})
            self.encoder_name_or_path = encoder_name_or_path
            self.main_labels = sorted(str(label) for label in main_labels)
            self.sub_labels = sorted(str(label) for label in sub_labels)
            if len(set(self.main_labels)) != len(self.main_labels):
                raise ValueError("main_labels contains duplicates.")
            if len(set(self.sub_labels)) != len(self.sub_labels):
                raise ValueError("sub_labels contains duplicates.")
            self.dropout = float(dropout)
            self.main_loss_weight = float(main_loss_weight)
            self.sub_loss_weight = float(sub_loss_weight)
            self.keys_to_ignore_at_inference = ["main_loss", "sub_loss"]

        @property
        def num_main_labels(self) -> int:
            return len(self.main_labels)

        @property
        def num_sub_labels(self) -> int:
            return len(self.sub_labels)

        @property
        def main_id_to_label(self) -> dict[int, str]:
            return dict(enumerate(self.main_labels))

        @property
        def sub_id_to_label(self) -> dict[int, str]:
            return dict(enumerate(self.sub_labels))

    @dataclass
    class MultiHeadIntentOutput(ModelOutput):
        """Outputs from both heads and their optional component losses."""

        loss: torch.Tensor | None = None
        main_logits: torch.Tensor | None = None
        sub_logits: torch.Tensor | None = None
        main_loss: torch.Tensor | None = None
        sub_loss: torch.Tensor | None = None

    class MultiHeadIntentModel(PreTrainedModel):
        """Shared transformer encoder with independent main- and sub-intent heads."""

        config_class = MultiHeadIntentConfig
        base_model_prefix = "encoder"

        def __init__(
            self, config: MultiHeadIntentConfig, *, encoder: nn.Module | None = None
        ) -> None:
            super().__init__(config)
            injected_encoder = encoder is not None
            if encoder is None:
                if not config.encoder_config:
                    raise ValueError(
                        "encoder_config is required when an encoder module is not injected."
                    )
                encoder_payload = dict(config.encoder_config)
                model_type = encoder_payload.pop("model_type", None)
                if not model_type:
                    raise ValueError("encoder_config must include its Hugging Face model_type.")
                encoder_hf_config = AutoConfig.for_model(model_type, **encoder_payload)
                encoder = AutoModel.from_config(encoder_hf_config)

            hidden_size = getattr(getattr(encoder, "config", None), "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(getattr(encoder, "config", None), "dim", None)
            if hidden_size is None:
                raise ValueError("Injected encoder config must expose hidden_size or dim.")
            if config.num_main_labels < 1 or config.num_sub_labels < 1:
                raise ValueError("Both main_labels and sub_labels must be non-empty.")

            self.encoder = encoder
            self.dropout = nn.Dropout(config.dropout)
            self.main_classifier = nn.Linear(hidden_size, config.num_main_labels)
            self.sub_classifier = nn.Linear(hidden_size, config.num_sub_labels)
            if injected_encoder:
                self._init_weights(self.main_classifier)
                self._init_weights(self.sub_classifier)
            else:
                self.post_init()

        @classmethod
        def from_encoder_pretrained(
            cls,
            encoder_name_or_path: str,
            *,
            main_labels: Sequence[str],
            sub_labels: Sequence[str],
            dropout: float = 0.2,
            main_loss_weight: float = 1.0,
            sub_loss_weight: float = 1.0,
            local_files_only: bool = False,
            **encoder_kwargs: Any,
        ) -> MultiHeadIntentModel:
            """Intentionally load an encoder for a new training run.

            Set ``local_files_only=True`` to guarantee that Transformers will not access the
            network. Plain model construction never resolves an encoder name over the network.
            """

            encoder_config = AutoConfig.from_pretrained(
                encoder_name_or_path,
                local_files_only=local_files_only,
                **encoder_kwargs,
            )
            encoder = AutoModel.from_pretrained(
                encoder_name_or_path,
                config=encoder_config,
                local_files_only=local_files_only,
                **encoder_kwargs,
            )
            config = MultiHeadIntentConfig(
                encoder_config=encoder_config.to_dict(),
                encoder_name_or_path=encoder_name_or_path,
                main_labels=main_labels,
                sub_labels=sub_labels,
                dropout=dropout,
                main_loss_weight=main_loss_weight,
                sub_loss_weight=sub_loss_weight,
            )
            return cls(config, encoder=encoder)

        @classmethod
        def from_local_artifacts(cls, model_path: str, **kwargs: Any) -> MultiHeadIntentModel:
            """Load a serialized model while explicitly prohibiting network fallback."""

            return cls.from_pretrained(model_path, local_files_only=True, **kwargs)

        def forward(
            self,
            input_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            labels_main: torch.Tensor | None = None,
            labels_sub: torch.Tensor | None = None,
            **encoder_kwargs: Any,
        ) -> MultiHeadIntentOutput:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **encoder_kwargs,
            )
            sequence = encoder_outputs.last_hidden_state
            shared_representation = self.dropout(sequence[:, 0, :])
            main_logits = self.main_classifier(shared_representation)
            sub_logits = self.sub_classifier(shared_representation)

            main_loss = None
            sub_loss = None
            combined_loss = None
            loss_function = CrossEntropyLoss()
            if labels_main is not None:
                main_loss = loss_function(main_logits, labels_main)
            if labels_sub is not None:
                sub_loss = loss_function(sub_logits, labels_sub)
            if main_loss is not None and sub_loss is not None:
                combined_loss = (
                    self.config.main_loss_weight * main_loss
                    + self.config.sub_loss_weight * sub_loss
                )
            elif main_loss is not None:
                combined_loss = self.config.main_loss_weight * main_loss
            elif sub_loss is not None:
                combined_loss = self.config.sub_loss_weight * sub_loss

            return MultiHeadIntentOutput(
                loss=combined_loss,
                main_logits=main_logits,
                sub_logits=sub_logits,
                main_loss=main_loss,
                sub_loss=sub_loss,
            )

else:

    class MultiHeadIntentConfig:  # type: ignore[no-redef]
        """Dependency guard used when model libraries are not installed."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_ml_backend()

    class MultiHeadIntentModel:  # type: ignore[no-redef]
        """Dependency guard used when model libraries are not installed."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            require_ml_backend()
