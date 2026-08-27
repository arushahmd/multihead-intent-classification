from __future__ import annotations

import tempfile
import types
import unittest

from intent_classifier.modeling import (
    MultiHeadIntentConfig,
    MultiHeadIntentModel,
    is_ml_backend_available,
)


@unittest.skipUnless(is_ml_backend_available(), "torch and transformers are not installed")
class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch
        from torch import nn

        class TinyEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = types.SimpleNamespace(hidden_size=8)
                self.embedding = nn.Embedding(31, 8)

            def forward(self, input_ids=None, attention_mask=None, **_kwargs):
                del attention_mask
                return types.SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        self.torch = torch
        self.encoder = TinyEncoder()
        self.config = MultiHeadIntentConfig(
            encoder_config={"model_type": "bert", "hidden_size": 8},
            main_labels=["menu", "cart"],
            sub_labels=["browse", "add", "remove"],
            dropout=0.0,
        )

    def test_forward_shapes(self) -> None:
        model = MultiHeadIntentModel(self.config, encoder=self.encoder)
        output = model(input_ids=self.torch.tensor([[1, 2, 3], [4, 5, 6]]))
        self.assertEqual(tuple(output.main_logits.shape), (2, 2))
        self.assertEqual(tuple(output.sub_logits.shape), (2, 3))

    def test_combined_loss_is_sum_of_head_losses(self) -> None:
        model = MultiHeadIntentModel(self.config, encoder=self.encoder)
        output = model(
            input_ids=self.torch.tensor([[1, 2], [3, 4]]),
            labels_main=self.torch.tensor([0, 1]),
            labels_sub=self.torch.tensor([1, 2]),
        )
        self.assertTrue(
            self.torch.allclose(output.loss, output.main_loss + output.sub_loss, atol=1e-7)
        )

    def test_config_label_ids_are_deterministic(self) -> None:
        self.assertEqual(self.config.main_labels, ["cart", "menu"])
        self.assertEqual(self.config.sub_labels, ["add", "browse", "remove"])
        self.assertEqual(self.config.main_id_to_label, {0: "cart", 1: "menu"})

    def test_tiny_model_serializes_and_reloads_locally(self) -> None:
        from transformers import BertConfig

        encoder_config = BertConfig(
            vocab_size=31,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
        config = MultiHeadIntentConfig(
            encoder_config=encoder_config.to_dict(),
            main_labels=["cart", "menu"],
            sub_labels=["add", "browse", "remove"],
            dropout=0.0,
        )
        model = MultiHeadIntentModel(config)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = MultiHeadIntentModel.from_local_artifacts(directory)
            output = restored(input_ids=self.torch.tensor([[1, 2, 3]]))
        self.assertEqual(tuple(output.main_logits.shape), (1, 2))
        self.assertEqual(tuple(output.sub_logits.shape), (1, 3))


if __name__ == "__main__":
    unittest.main()
