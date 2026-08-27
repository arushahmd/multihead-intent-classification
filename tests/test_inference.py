from __future__ import annotations

import unittest

from intent_classifier.data import LabelHierarchy
from intent_classifier.inference import IntentPrediction, validate_text_inputs


class InferenceContractTests(unittest.TestCase):
    def test_single_and_batch_inputs(self) -> None:
        self.assertEqual(validate_text_inputs("hello"), ("hello",))
        self.assertEqual(validate_text_inputs(["hello", "thanks"]), ("hello", "thanks"))

    def test_blank_empty_and_non_string_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_text_inputs([])
        with self.assertRaises(ValueError):
            validate_text_inputs([" "])
        with self.assertRaises(TypeError):
            validate_text_inputs(["valid", 7])  # type: ignore[list-item]

    def test_invalid_hierarchy_pair_can_be_flagged(self) -> None:
        hierarchy = LabelHierarchy.from_mapping({"cart": ["add"], "menu": ["browse"]})
        prediction = IntentPrediction(
            text="browse",
            main_intent="cart",
            sub_intent="browse",
            main_confidence=0.8,
            sub_confidence=0.7,
            invalid_hierarchy_pair=not hierarchy.is_valid_pair("cart", "browse"),
        )
        self.assertTrue(prediction.invalid_hierarchy_pair)


if __name__ == "__main__":
    unittest.main()
