"""
Transform a batch of text records into tokenized features.
Doesn't assume the texts are de-duplicated.
"""

from collections import defaultdict
from typing import Iterable

from datasets import Dataset
from sentence_transformers.data_collator import SentenceTransformerDataCollator


def group_positives_by_anchor(
    dataset: Iterable[dict[str, str]],
) -> dict[str, dict[str, None]]:
    anchor_to_positives = defaultdict(dict)
    # Using a dict to de-duplicate. Not using a set so that positives and negatives are
    # in a deterministic (insertion) order.
    for record in dataset:
        anchor_to_positives[record["anchor"]][record["positive"]] = None
    return anchor_to_positives


class MPNRLDataCollator(SentenceTransformerDataCollator):
    def __init__(self, dataset: Dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: SentenceTransformerDataCollator actually assumes that first is anchor,
        # next is positive, rest are negative, regardless of the actual column names.
        # Here, we assume they're labeled "anchor", "positive", "negative".
        self.anchor_to_positives = group_positives_by_anchor(dataset)

    def __call__(self, features: list[dict[str, str]]):
        # Using a dict for deterministic (insertion) order.
        anchors = list({record["anchor"]: None for record in features})
        positives = list({record["positive"]: None for record in features})
        negatives = list({record["negative"]: None for record in features})

        positive_idxs = [
            [
                j
                for j, positive in enumerate(positives)
                if positive in self.anchor_to_positives[anchor]
            ]
            for anchor in anchors
        ]
        # positive_idxs[i] is a list of all indices of positives which are positives for
        # anchors[i]. It's structured like this (instead of, e.g., a flat list of (i, j)
        # pairs) b/c it may be useful to batch by anchors later.

        # TODO: this is gettin hacky
        batch = (
            super().__call__([{"anchor": anchor} for anchor in anchors])
            | super().__call__([{"positive": positive} for positive in positives])
            | super().__call__([{"negative": negative} for negative in negatives])
        )
        batch["label"] = positive_idxs
        return batch

    def maybe_warn_about_column_order(self, *args, **kwargs):
        # TODO: this is temporarily overriden and suppressed.
        pass
