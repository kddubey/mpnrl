"""
Sigmoid-style loss, i.e., assumes the classifications are independent.
"""

from typing import Any, Iterable, Sequence

from sentence_transformers import SentenceTransformer, util
import torch


class MultiplePositivesNegativesRankingLoss(torch.nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 20.0,
        similarity_fct=util.cos_sim,
        bias: float = -10.0,
    ) -> None:
        super(MultiplePositivesNegativesRankingLoss, self).__init__()
        self.model = model
        self.scale = torch.nn.Parameter(torch.tensor(scale, device=model.device))
        self.similarity_fct = similarity_fct
        self.bias = torch.nn.Parameter(torch.tensor(bias, device=model.device))
        # TODO: the learning rate for scale and bias should probably be higher. See
        # https://github.com/UKPLab/sentence-transformers/blob/679ab5d38e4cf9cd73d4dcf1cda25ba2ef1ad837/sentence_transformers/trainer.py#L1206-L1219
        self.bce_with_logits_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        sentence_features: Iterable[dict[str, torch.Tensor]],
        labels: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        # Compute the embeddings and distribute them to anchor and candidates (positive
        # and optionally negatives)
        embeddings = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]
        anchors = embeddings[0]  # (batch_size, embedding_dim)
        candidates = torch.cat(
            embeddings[1:]
        )  # (num_positives + num_negatives), embedding_dim)

        # For every anchor, we compute the similarity to all other candidates (positives
        # and negatives), also from other anchors. This gives us a lot of in-batch
        # negatives.
        scores: torch.Tensor = (
            self.similarity_fct(anchors, candidates) * self.scale
        ) + self.bias
        # (batch_size, batch_size * (1 + num_negatives))

        positive_pairs = labels
        labels = torch.zeros_like(scores)
        for i, positive_indices in enumerate(positive_pairs):
            for j in positive_indices:
                labels[i, j] = 1.0

        return self.bce_with_logits_loss(scores, labels)

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale.item(),
            "similarity_fct": self.similarity_fct.__name__,
            "bias": self.bias.item(),
        }

    @property
    def citation(self) -> str:
        return """
@inproceedings{zhai2023sigmoid,
    title={Sigmoid loss for language image pre-training},
    author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={11975--11986},
    year={2023}
}
"""
