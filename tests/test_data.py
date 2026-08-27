from __future__ import annotations

import csv
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from intent_classifier.data import (
    ContradictoryLabelError,
    DatasetValidationError,
    HierarchyValidationError,
    IntentRecord,
    LabelHierarchy,
    assert_no_normalized_text_overlap,
    audit_duplicates,
    build_label_mappings,
    load_intent_csv,
    normalize_text,
    split_by_normalized_text,
)


class DatasetValidationTests(unittest.TestCase):
    def write_csv(self, fieldnames: list[str], rows: list[dict[str, str]]) -> Path:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as temporary:
            writer = csv.DictWriter(temporary, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        return Path(temporary.name)

    def test_missing_required_column_is_rejected(self) -> None:
        path = self.write_csv(
            ["text", "main_intent"], [{"text": "hello", "main_intent": "conversation"}]
        )
        with self.assertRaisesRegex(DatasetValidationError, "sub_intent"):
            load_intent_csv(path)

    def test_blank_required_values_are_rejected(self) -> None:
        for field in ("text", "main_intent", "sub_intent"):
            with self.subTest(field=field):
                row = {
                    "text": "hello",
                    "main_intent": "conversation",
                    "sub_intent": "greeting",
                }
                row[field] = "   "
                path = self.write_csv(list(row), [row])
                with self.assertRaises(DatasetValidationError):
                    load_intent_csv(path)

    def test_normalized_duplicates_are_reported(self) -> None:
        records = (
            IntentRecord("  HELLO   there ", "conversation", "greeting"),
            IntentRecord("hello there", "conversation", "greeting"),
            IntentRecord("Thanks", "conversation", "gratitude"),
        )
        audit = audit_duplicates(records)
        self.assertEqual(audit.duplicate_groups, 1)
        self.assertEqual(audit.duplicate_rows, 1)
        self.assertEqual(normalize_text("Ａ  B"), "a b")

    def test_contradictory_normalized_labels_are_rejected(self) -> None:
        records = (
            IntentRecord("Show MENU", "menu", "browse_menu"),
            IntentRecord(" show   menu ", "cart", "show_cart"),
        )
        with self.assertRaises(ContradictoryLabelError):
            split_by_normalized_text(records)


class HierarchyAndSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.hierarchy = LabelHierarchy.from_json(cls.root / "configs" / "label_hierarchy.json")
        cls.records = load_intent_csv(cls.root / "data" / "sample" / "intents.csv")

    def test_canonical_hierarchy_has_expected_cardinality(self) -> None:
        self.assertEqual(len(self.hierarchy.main_labels), 5)
        self.assertEqual(len(self.hierarchy.sub_labels), 34)
        self.hierarchy.validate_records(self.records)

    def test_sample_has_three_fictional_examples_per_sub_intent(self) -> None:
        counts = Counter(record.sub_intent for record in self.records)
        self.assertEqual(set(counts), set(self.hierarchy.sub_labels))
        self.assertEqual(set(counts.values()), {3})

    def test_duplicate_sub_ownership_is_rejected(self) -> None:
        with self.assertRaises(HierarchyValidationError):
            LabelHierarchy.from_mapping({"first": ["shared"], "second": ["shared"]})

    def test_label_mapping_is_deterministic(self) -> None:
        reversed_records = tuple(reversed(self.records))
        self.assertEqual(build_label_mappings(self.records), build_label_mappings(reversed_records))
        self.assertEqual(build_label_mappings(self.records)[0], self.hierarchy.main_to_id)
        self.assertEqual(build_label_mappings(self.records)[1], self.hierarchy.sub_to_id)

    def test_split_is_deterministic_under_input_reordering(self) -> None:
        first = split_by_normalized_text(self.records, seed=42)
        shuffled = list(self.records)
        random.Random(9).shuffle(shuffled)
        second = split_by_normalized_text(shuffled, seed=42)
        self.assertEqual(first.membership, second.membership)

    def test_normalized_duplicate_group_never_crosses_splits(self) -> None:
        records = list(self.records)
        original = records[0]
        records.append(
            IntentRecord(
                text=f"  {original.text.upper()}  ",
                main_intent=original.main_intent,
                sub_intent=original.sub_intent,
            )
        )
        splits = split_by_normalized_text(records, seed=12)
        assert_no_normalized_text_overlap(splits)
        group_splits = {
            item.split for item in splits.membership if item.group_id == original.group_id
        }
        self.assertEqual(len(group_splits), 1)

    def test_split_preserves_every_row(self) -> None:
        splits = split_by_normalized_text(self.records, seed=42)
        self.assertEqual(
            len(splits.train) + len(splits.validation) + len(splits.test), len(self.records)
        )
        self.assertTrue(splits.train)
        self.assertTrue(splits.validation)
        self.assertTrue(splits.test)


if __name__ == "__main__":
    unittest.main()
