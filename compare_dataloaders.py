"""
Plot the observed batch size for each iteration for no-duplicates sampling and then for
MPNRL (plain sampling with grouping in the collator).

Saves 2 plots: dataloader_mnrl.png and dataloader_mpnrl.png
"""

import os
from typing import Any, Optional

from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.training_args import BatchSamplers
from tap import tapify
from tqdm.auto import tqdm

from mpnrl.data_collator import GroupingDataCollator
from mpnrl.losses import MultiplePositivesNegativesRankingLoss

sns.set_theme(style="darkgrid")


def dry_run(trainer: SentenceTransformerTrainer) -> list[dict[str, Any]]:
    num_batches = len(trainer.train_dataset) // trainer.args.train_batch_size
    data = []
    # IIUC, the trainer doesn't fully exhaust the dataloader iterator; it stops when
    # num_batches is hit. Let's replicate this behavior.
    dataloader = zip(
        range(num_batches),
        map(trainer.collect_features, trainer.get_train_dataloader()),
    )
    for _, (sentence_features, labels) in tqdm(dataloader, total=num_batches):
        batch_data = {}
        features_anchor, features_candidates = (
            sentence_features[0],
            sentence_features[1:],
        )
        batch_data["num_anchors"] = features_anchor["input_ids"].shape[0]
        batch_data["num_candidates"] = sum(
            features["input_ids"].shape[0] for features in features_candidates
        )
        if labels is not None:
            batch_data["num_positives_per_anchor"] = [
                len(indices) for indices in labels
            ]
        data.append(batch_data)
    return data


def plot(
    batch_data: list[dict[str, Any]],
    batch_size: int,
    title: str = "",
    num_batch_ticks: int = 5,
) -> plt.Axes:
    df = (
        pl.DataFrame(batch_data)
        .with_columns(pl.Series("batch", range(1, len(batch_data) + 1)))
        .rename({"num_anchors": "anchors", "num_candidates": "candidates"})
    )
    on = ["anchors", "candidates"]
    if "num_positives_per_anchor" in batch_data[0]:
        df = df.with_columns(
            pl.col("num_positives_per_anchor")
            .map_elements(sum, returns_scalar=True, return_dtype=int)
            .alias("positives")
        )
        on.append("positives")

    melted_df = df.unpivot(
        index=["batch"],
        on=on,
        variable_name="Type",
        value_name="Value",
    )
    ax = sns.lineplot(data=melted_df, x="batch", y="Value", hue="Type")
    ax.set_xticks(range(1, len(df) + 1, round(len(df) / num_batch_ticks)))
    ax.set_ylim(0, max(ax.get_ylim()[1], batch_size * 1.1))
    ax.axhline(
        y=batch_size, color="gray", linestyle="dotted", label="inputted batch size"
    )
    ax.set_ylabel("# observations")
    ax.set_title(title)
    ax.legend()
    return ax


def main(
    dataset_name: str,
    batch_size: int,
    dataset_size_train: Optional[int] = None,
    dataset_config: Optional[str] = None,
    dataset_split_train: Optional[str] = "train",
    seed: Optional[int] = None,
):
    """
    Parameters
    ----------
    dataset_name : str
        Name of a HF dataset.
    batch_size : int
        Desired batch size.
    dataset_size_train : Optional[int], optional
        Number of training records to subsample, by default keep all
    dataset_config : Optional[str], optional
        Config/subset name in HF, by default None
    dataset_split_train : Optional[str], optional
        Training split name in HF., by default "train"
    seed : Optional[int], optional
        If given, the training data will be shuffled before subsampling
        `dataset_size_train` records, by default no shuffling (take the first)
    """
    # if there are already plots saved here, raise an error
    if os.path.exists("dataloader_mnrl.png") or os.path.exists("dataloader_mpnrl.png"):
        raise FileExistsError(
            "Please move the existing plots at `dataloader_mnrl.png` and/or "
            "`dataloader_mpnrl.png` before running this script."
        )

    model = SentenceTransformer("distilroberta-base")
    # Just need a dummy model. hardcoding it.

    train_dataset = load_dataset(
        dataset_name, name=dataset_config, split=dataset_split_train
    )
    # Subsample
    dataset_size_train = (
        len(train_dataset)
        if dataset_size_train is None
        else min(dataset_size_train, len(train_dataset))
    )
    if seed is not None:
        generator = np.random.default_rng(seed=seed)
        indices = generator.choice(
            dataset_size_train, size=dataset_size_train, replace=False
        )
    else:
        indices = range(dataset_size_train)

    train_dataset = train_dataset.select(indices)

    output_dir = "_output"  # won't get created b/c we never train

    print("\nDry running the MNRL dataloader")
    dummy_trainer_mnrl = SentenceTransformerTrainer(
        model=model,
        args=SentenceTransformerTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=batch_size,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            seed=42,
        ),
        train_dataset=train_dataset,
        loss=losses.MultipleNegativesRankingLoss(model),
        data_collator=None,
    )
    batch_data_mnrl = dry_run(dummy_trainer_mnrl)

    print("\nDry running the MPNRL dataloader")
    dummy_trainer_mpnrl = SentenceTransformerTrainer(
        model=model,
        args=SentenceTransformerTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=batch_size,
            batch_sampler=BatchSamplers.BATCH_SAMPLER,
            seed=42,
        ),
        train_dataset=train_dataset,
        loss=MultiplePositivesNegativesRankingLoss(model),
        data_collator=GroupingDataCollator(train_dataset, tokenize_fn=model.tokenize),
    )
    batch_data_mpnrl = dry_run(dummy_trainer_mpnrl)

    def sum_key(data: list[dict[str, Any]], key: str) -> int:
        return sum(batch_data[key] for batch_data in data)

    print("\nMNRL")
    print(f'total #    anchors: {sum_key(batch_data_mnrl, "num_anchors")}')
    print(f'total # candidates: {sum_key(batch_data_mnrl, "num_candidates")}')
    # If skewed, it's prolly the case that the dataloader for MNRL didn't process all of
    # the inputted training data b/c it stops after `num_batches` steps.
    print("\nMPNRL")
    print(f'total #    anchors: {sum_key(batch_data_mpnrl, "num_anchors")}')
    print(f'total # candidates: {sum_key(batch_data_mpnrl, "num_candidates")}')

    print()

    ax_mnrl = plot(batch_data_mnrl, batch_size, title=f"{dataset_name} - MNRL")
    print("Saving MNRL plot to dataloader_mnrl.png")
    ax_mnrl.get_figure().savefig("dataloader_mnrl.png")
    plt.close(ax_mnrl.get_figure())

    ax_mpnrl = plot(batch_data_mpnrl, batch_size, title=f"{dataset_name} - MPNRL")
    print("Saving MPNRL plot to dataloader_mpnrl.png")
    ax_mpnrl.get_figure().savefig("dataloader_mpnrl.png")
    plt.close(ax_mpnrl.get_figure())


if __name__ == "__main__":
    tapify(main, description=__doc__)
