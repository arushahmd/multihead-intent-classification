# Historical error analysis

This note summarizes aggregate evidence only. It contains no copied utterances and makes no causal claims that the available artifacts cannot support.

## Main intent versus sub-intent

On the 16,050-row historical common-set audit, main-intent accuracy was `0.9543302180685358`, while sub-intent accuracy was `0.6632398753894081`. Joint exact match was lower still at `0.6456697819314642` because an example counts as correct only when **both** independently predicted heads are correct.

The gap is consistent with two observable conditions: fine-grained sub-intents are harder to separate in the recorded outputs, and the common-set taxonomy does not match the model's taxonomy. It is not sufficient evidence to estimate performance on new traffic.

## Aggregate category patterns

The common-set main-intent report shows high F1 for the dominant `cart` (`0.9913877222010097`, support 11,782) and `menu` (`0.9771673819742489`, support 2,908) categories. `order` was weaker (`0.861244019138756`, support 643). The weighted main F1 (`0.954390267656145`) is much higher than macro F1 (`0.5328024248502196`), reflecting imbalance and labels that cannot map cleanly across the two taxonomies.

At the sub-intent level:

- `replace_item` recall was `0.3108522378908645` over 3,262 rows; the aggregate failure table shows frequent confusion with `modify_item`.
- `remove_item` recall was `0.6885944155229532` over 2,113 rows.
- `review_order` recall was `0.2807017543859649` over 57 rows, often confused with `show_cart` in the aggregate counts.
- The common-set label `ask_price` had support 1,227 but zero recall, while the model taxonomy exposed `ask_item_price`. This is direct evidence of taxonomy mismatch, not evidence that the underlying price-query concept was wholly unlearned.

## Interpretation limits

The common set overlaps historical training-source material, changes the label inventory from five main/34 sub labels to six main/37 sub labels, and was later involved in model selection. The exact historical checkpoint is missing. Accordingly, this analysis is useful for understanding recorded failure modes and evaluation design, but not for claiming blind external generalization.

Future runs should freeze the hierarchy before splitting, group duplicate normalized texts, select models on validation metrics only, and reserve one untouched final test set for report-only evaluation.
