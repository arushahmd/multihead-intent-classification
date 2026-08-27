# Historical results

This directory keeps only compact aggregate evidence from one preserved historical run. It contains no model weights, raw evaluation data, prediction dump, ground-truth dump, or utterance-level failure export.

## Evidence classes

### Historical internal split

The four internal values in [`historical_summary.csv`](historical_summary.csv) are **Historical experiment metrics — strongly evidenced but not end-to-end reproducible from the currently preserved artifacts.** The preserved experiment registry and summary agree on the 2,705/580/580 train/validation/test row counts, seed 42, configuration, and metrics. The exact historical training CSV, row membership, checkpoint, and internal predictions were not recovered, so these values cannot be reproduced by this repository.

### Historical common-set evaluation audit

The three common-set values were recomputed from preserved prediction rows. The raw prediction file is intentionally not distributed; its digest is recorded in [`common_eval_metrics.json`](historical/run_002/common_eval_metrics.json). The common set is not an independent blind external test: it overlaps historical training-source data, has a six-main/37-sub taxonomy rather than the run's five-main/34-sub taxonomy, includes changed concepts, and was later used for model selection.

The classification reports, confusion matrices, and aggregate failure summary are retained to support inspection of that mismatch. They must not be read as a clean generalization benchmark or as current model performance.

## Provenance checks

- Preserved internal leaderboard SHA-256: `e13cf5229b95b06238304a7e358ea28fe077aec757b0a48404cdff1d7804316f`
- Preserved best-run summary SHA-256: `9e7c623d235c55db0340088761ed248b6e5843ed9730087df6a214ebc4c502d8`
- Preserved common-set metrics SHA-256: `c4eb4b595f8079b883349ed09f0ef758d5ac9575a4091359689229d06c62cea2`
- Preserved common-set predictions SHA-256 (not distributed): `88976e1ff9ab1a378fc2bc6bb37c19edb9d4f8b0a274990cbdff7d9525cbde7d`
- Preserved training-loss figure SHA-256: `7b5da27d3c01e56be9e63782e228292425a16d8200368e2fafe5751537f4a145`

These hashes identify evidence in the separately preserved historical archive; they are not claims that the missing checkpoint can be reconstructed.
