"""
Transform a batch of text records into tokenized features.
Doesn't assume the texts are de-duplicated.
"""

from collections import defaultdict
from typing import Iterable

from sentence_transformers.data_collator import SentenceTransformerDataCollator
import torch


def group_positives_by_anchor(
    dataset: Iterable[dict[str, str]], anchor_name: str, positive_name: str
) -> dict[str, set[str]]:
    anchor_to_positives = defaultdict(set)
    for record in dataset:
        anchor_to_positives[record[anchor_name]].add(record[positive_name])
    return anchor_to_positives


class GroupingDataCollator(SentenceTransformerDataCollator):
    """
    A data collator that groups positives by the anchor.

    As assumed in SentenceTransformers, for every record, the first entry is the anchor,
    the second is the positive, and the rest are negatives.

    Currently, it only makes sense to use this collator with
    `MultiplePositivesNegativesRankingLoss`.
    """

    def __init__(self, dataset: Iterable[dict[str, str]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        first_record = next(iter(dataset))
        self.column_names = list(first_record.keys())
        self.anchor_name, self.positive_name = self.column_names[:2]
        self.anchor_to_positives = group_positives_by_anchor(
            dataset, self.anchor_name, self.positive_name
        )

    def __call__(
        self, features: list[dict[str, str]]
    ) -> dict[str, torch.Tensor | list[list[int]]]:
        # Sentence features
        tokenize_and_name_encodings = super().__call__
        batch = {
            encoding_name: encoding_value
            # e.g., "anchor_input_ids": torch.Tensor([[...], ..., [...]])
            for column_name in self.column_names
            for encoding_name, encoding_value in (
                tokenize_and_name_encodings(
                    [
                        {column_name: text}
                        for text in {record[column_name]: None for record in features}
                        # Using a dict for deterministic (insertion) order.
                    ]
                ).items()
            )
        }
        # NOTE: SentenceTransformerTrainer.collect_features will group this data by the
        # prefix of each key.

        # Labels
        anchors = {record[self.anchor_name]: None for record in features}
        positives = {record[self.positive_name]: None for record in features}
        # These are in the same (insertion) order as above.
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
        batch["label"] = positive_idxs
        return batch

    def maybe_warn_about_column_order(self, *args, **kwargs):
        # TODO: this is temporarily overriden and suppressed.
        pass
