"""Dataset validation, label hierarchy management, and leakage-resistant splitting."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = ("text", "main_intent", "sub_intent")
_WHITESPACE = re.compile(r"\s+")


class DatasetValidationError(ValueError):
    """Raised when intent data is missing required fields or contains invalid values."""


class ContradictoryLabelError(DatasetValidationError):
    """Raised when the same normalized utterance has more than one label pair."""


class HierarchyValidationError(ValueError):
    """Raised when a label hierarchy is ambiguous or incompatible with data."""


@dataclass(frozen=True, slots=True)
class IntentRecord:
    """One validated hierarchical intent example."""

    text: str
    main_intent: str
    sub_intent: str
    source_row: int | None = None

    @property
    def normalized_text(self) -> str:
        return normalize_text(self.text)

    @property
    def group_id(self) -> str:
        return hashlib.sha256(self.normalized_text.encode("utf-8")).hexdigest()

    @property
    def stable_id(self) -> str:
        payload = "\x1f".join((self.normalized_text, self.main_intent, self.sub_intent))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DuplicateAudit:
    """Summary of normalized-text duplication in a validated dataset."""

    duplicate_groups: int
    duplicate_rows: int
    contradictory_groups: int


@dataclass(frozen=True, slots=True)
class SplitMembership:
    """Stable membership record for one normalized-text group."""

    group_id: str
    split: str
    row_count: int
    main_intent: str
    sub_intent: str


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    """Group-disjoint train, validation, and test records plus stable membership."""

    train: tuple[IntentRecord, ...]
    validation: tuple[IntentRecord, ...]
    test: tuple[IntentRecord, ...]
    membership: tuple[SplitMembership, ...]

    def as_dict(self) -> dict[str, tuple[IntentRecord, ...]]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


@dataclass(frozen=True, slots=True)
class LabelHierarchy:
    """Validated one-to-many mapping from main intents to globally unique sub-intents."""

    main_to_sub: Mapping[str, tuple[str, ...]]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Sequence[str]]) -> LabelHierarchy:
        if not mapping:
            raise HierarchyValidationError(
                "The label hierarchy must define at least one main intent."
            )

        canonical: dict[str, tuple[str, ...]] = {}
        owner_by_sub: dict[str, str] = {}
        for raw_main, raw_subs in mapping.items():
            main = str(raw_main).strip()
            if not main:
                raise HierarchyValidationError("Main-intent labels cannot be blank.")
            if main in canonical:
                raise HierarchyValidationError(f"Duplicate main-intent label: {main}")

            subs = tuple(sorted(str(value).strip() for value in raw_subs))
            if not subs or any(not value for value in subs):
                raise HierarchyValidationError(f"Main intent '{main}' has blank or no sub-intents.")
            if len(set(subs)) != len(subs):
                raise HierarchyValidationError(
                    f"Main intent '{main}' contains duplicate sub-intents."
                )
            for sub in subs:
                prior_owner = owner_by_sub.get(sub)
                if prior_owner is not None:
                    raise HierarchyValidationError(
                        f"Sub-intent '{sub}' belongs to both '{prior_owner}' and '{main}'."
                    )
                owner_by_sub[sub] = main
            canonical[main] = subs

        return cls(main_to_sub={main: canonical[main] for main in sorted(canonical)})

    @classmethod
    def from_json(cls, path: str | Path) -> LabelHierarchy:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        mapping = payload.get("main_to_sub")
        if not isinstance(mapping, dict):
            raise HierarchyValidationError("Hierarchy JSON must contain a 'main_to_sub' object.")
        return cls.from_mapping(mapping)

    @property
    def main_labels(self) -> tuple[str, ...]:
        return tuple(self.main_to_sub)

    @property
    def sub_labels(self) -> tuple[str, ...]:
        return tuple(sorted(sub for subs in self.main_to_sub.values() for sub in subs))

    @property
    def main_to_id(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.main_labels)}

    @property
    def sub_to_id(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.sub_labels)}

    @property
    def sub_to_main(self) -> dict[str, str]:
        return {sub: main for main, subs in self.main_to_sub.items() for sub in subs}

    def is_valid_pair(self, main_intent: str, sub_intent: str) -> bool:
        return self.sub_to_main.get(sub_intent) == main_intent

    def validate_records(self, records: Iterable[IntentRecord]) -> None:
        for record in records:
            if record.main_intent not in self.main_to_sub:
                raise HierarchyValidationError(
                    f"Unknown main intent '{record.main_intent}' at row {record.source_row}."
                )
            if record.sub_intent not in self.sub_to_main:
                raise HierarchyValidationError(
                    f"Unknown sub-intent '{record.sub_intent}' at row {record.source_row}."
                )
            if not self.is_valid_pair(record.main_intent, record.sub_intent):
                expected = self.sub_to_main[record.sub_intent]
                raise HierarchyValidationError(
                    f"Sub-intent '{record.sub_intent}' belongs to '{expected}', not "
                    f"'{record.main_intent}' (row {record.source_row})."
                )

    def canonical_json(self) -> str:
        payload = {"main_to_sub": {main: list(subs) for main, subs in self.main_to_sub.items()}}
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    """Normalize an utterance for duplicate grouping without altering source text."""

    normalized = unicodedata.normalize("NFKC", str(text)).casefold().strip()
    return _WHITESPACE.sub(" ", normalized)


def validate_records(records: Iterable[IntentRecord]) -> tuple[IntentRecord, ...]:
    """Validate required values and reject contradictory normalized-text labels."""

    validated: list[IntentRecord] = []
    labels_by_text: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for position, record in enumerate(records, start=1):
        text = str(record.text).strip()
        main = str(record.main_intent).strip()
        sub = str(record.sub_intent).strip()
        row = record.source_row if record.source_row is not None else position
        if not text:
            raise DatasetValidationError(f"Blank text at row {row}.")
        if not main:
            raise DatasetValidationError(f"Blank main_intent at row {row}.")
        if not sub:
            raise DatasetValidationError(f"Blank sub_intent at row {row}.")
        clean = IntentRecord(text=text, main_intent=main, sub_intent=sub, source_row=row)
        labels_by_text[clean.normalized_text].add((main, sub))
        validated.append(clean)

    if not validated:
        raise DatasetValidationError("The dataset contains no examples.")

    contradictions = {text: pairs for text, pairs in labels_by_text.items() if len(pairs) > 1}
    if contradictions:
        example_pairs = next(iter(contradictions.values()))
        raise ContradictoryLabelError(
            "The same normalized text has contradictory labels: "
            + ", ".join(f"{main}/{sub}" for main, sub in sorted(example_pairs))
        )
    return tuple(validated)


def audit_duplicates(records: Iterable[IntentRecord]) -> DuplicateAudit:
    """Count repeated normalized utterances after validating label consistency."""

    validated = validate_records(records)
    counts: dict[str, int] = defaultdict(int)
    for record in validated:
        counts[record.normalized_text] += 1
    duplicate_counts = [count for count in counts.values() if count > 1]
    return DuplicateAudit(
        duplicate_groups=len(duplicate_counts),
        duplicate_rows=sum(count - 1 for count in duplicate_counts),
        contradictory_groups=0,
    )


def load_intent_csv(path: str | Path) -> tuple[IntentRecord, ...]:
    """Load and validate a UTF-8 CSV with the canonical three-column schema."""

    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        missing = [column for column in REQUIRED_COLUMNS if column not in columns]
        if missing:
            raise DatasetValidationError(
                f"Missing required columns: {', '.join(missing)}. Found: {', '.join(columns)}"
            )
        records = [
            IntentRecord(
                text=row.get("text", ""),
                main_intent=row.get("main_intent", ""),
                sub_intent=row.get("sub_intent", ""),
                source_row=index,
            )
            for index, row in enumerate(reader, start=2)
        ]
    return validate_records(records)


def dataset_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of the exact dataset bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_label_mappings(
    records: Iterable[IntentRecord],
) -> tuple[dict[str, int], dict[str, int]]:
    """Build input-order-independent label-to-ID mappings."""

    validated = validate_records(records)
    main_labels = sorted({record.main_intent for record in validated})
    sub_labels = sorted({record.sub_intent for record in validated})
    return (
        {label: index for index, label in enumerate(main_labels)},
        {label: index for index, label in enumerate(sub_labels)},
    )


def _stable_tie_order(seed: int, stratum: tuple[str, str]) -> tuple[str, ...]:
    names = ("train", "validation", "test")
    return tuple(
        sorted(
            names,
            key=lambda name: hashlib.sha256(
                f"{seed}\x1f{stratum[0]}\x1f{stratum[1]}\x1f{name}".encode()
            ).hexdigest(),
        )
    )


def _split_fractions(train: float, validation: float, test: float) -> dict[str, float]:
    values = {"train": train, "validation": validation, "test": test}
    if any(not math.isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError("All split fractions must be positive finite values.")
    if not math.isclose(sum(values.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Train, validation, and test fractions must sum to 1.0.")
    return values


def split_by_normalized_text(
    records: Iterable[IntentRecord],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> DatasetSplits:
    """Create deterministic, approximately stratified, normalized-text group splits.

    Entire duplicate groups stay together. Allocation is performed separately within each
    main/sub stratum, using stable hashes for ordering and deterministic squared-error balancing
    against requested row fractions. Exact ratios are not guaranteed when groups are large.
    """

    validated = validate_records(records)
    fractions = _split_fractions(train_fraction, validation_fraction, test_fraction)

    groups: dict[str, list[IntentRecord]] = defaultdict(list)
    for record in validated:
        groups[record.normalized_text].append(record)

    strata: dict[tuple[str, str], list[tuple[str, list[IntentRecord]]]] = defaultdict(list)
    for normalized, members in groups.items():
        label_pair = (members[0].main_intent, members[0].sub_intent)
        strata[label_pair].append((normalized, members))

    assigned_records: dict[str, list[IntentRecord]] = defaultdict(list)
    membership: list[SplitMembership] = []
    for stratum in sorted(strata):
        stratum_groups = strata[stratum]
        stratum_groups.sort(
            key=lambda item: (
                -len(item[1]),
                hashlib.sha256(f"{seed}\x1f{item[0]}".encode()).hexdigest(),
            )
        )
        total_rows = sum(len(members) for _, members in stratum_groups)
        targets = {name: total_rows * fraction for name, fraction in fractions.items()}
        assigned_counts = {name: 0 for name in fractions}
        tie_order = _stable_tie_order(seed, stratum)

        for normalized, members in stratum_groups:
            size = len(members)
            incremental_cost = {
                name: (assigned_counts[name] + size - targets[name]) ** 2
                - (assigned_counts[name] - targets[name]) ** 2
                for name in fractions
            }
            split = min(tie_order, key=lambda name: (incremental_cost[name], tie_order.index(name)))
            assigned_counts[split] += size
            assigned_records[split].extend(members)
            membership.append(
                SplitMembership(
                    group_id=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                    split=split,
                    row_count=size,
                    main_intent=stratum[0],
                    sub_intent=stratum[1],
                )
            )

    def stable_records(name: str) -> tuple[IntentRecord, ...]:
        return tuple(
            sorted(
                assigned_records[name],
                key=lambda record: (record.group_id, record.stable_id, record.source_row or 0),
            )
        )

    result = DatasetSplits(
        train=stable_records("train"),
        validation=stable_records("validation"),
        test=stable_records("test"),
        membership=tuple(sorted(membership, key=lambda item: (item.split, item.group_id))),
    )
    assert_no_normalized_text_overlap(result)
    return result


def assert_no_normalized_text_overlap(splits: DatasetSplits) -> None:
    """Raise if a normalized utterance appears in more than one split."""

    names = tuple(splits.as_dict())
    groups = {name: {record.normalized_text for record in splits.as_dict()[name]} for name in names}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = groups[left] & groups[right]
            if overlap:
                raise DatasetValidationError(
                    f"Normalized-text leakage detected between {left} and {right}: "
                    f"{len(overlap)} group(s)."
                )


def write_split_membership(path: str | Path, memberships: Iterable[SplitMembership]) -> None:
    """Write stable group-level membership without copying utterance text."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("group_id", "split", "row_count", "main_intent", "sub_intent"),
        )
        writer.writeheader()
        for item in sorted(memberships, key=lambda value: (value.split, value.group_id)):
            writer.writerow(
                {
                    "group_id": item.group_id,
                    "split": item.split,
                    "row_count": item.row_count,
                    "main_intent": item.main_intent,
                    "sub_intent": item.sub_intent,
                }
            )
