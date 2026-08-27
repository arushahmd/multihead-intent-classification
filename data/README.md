# Data

The package expects a UTF-8 CSV with exactly these required semantic fields:

| Column | Meaning |
| --- | --- |
| `text` | Conversational utterance to classify |
| `main_intent` | Broad intent category |
| `sub_intent` | Fine-grained intent category owned by the main intent |

[`sample/intents.csv`](sample/intents.csv) contains 102 fictional examples—three per sub-intent—for schema demonstrations, CLI smoke checks, and unit tests. It contains no menu identifiers, merchant names, customer data, or production records.

> Public sample data is illustrative and is not the historical training dataset used for the reported archival experiments.

The loader rejects missing columns and blank values. It normalizes Unicode, casing, and whitespace for duplicate analysis, rejects contradictory labels attached to the same normalized utterance, and keeps identical normalized utterances in one split group.

Full training corpora are intentionally not distributed. A future dataset should be documented with its origin, usage rights, construction method, taxonomy version, SHA-256 digest, and stable split membership before its metrics are published.
