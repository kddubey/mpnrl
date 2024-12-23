"""
Script to run MNRL or MPNRL training on an inputted dataset.
"""

from contextlib import contextmanager
from datetime import datetime
from functools import partial
import json
import os
import pickle
from typing import Any, Callable, Iterable, Literal, Optional, get_args

from datasets import Dataset, load_dataset
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    SentenceEvaluator,
    SimilarityFunction,
)
from sentence_transformers.training_args import BatchSamplers
from tap import tapify
import torch
from transformers.trainer import TrainOutput
import wandb


class Experiment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    # Pydantic stuff: extra attributes are not allowed, and the object is immutable

    # TODO: would be cool to compose pydantic models into one and create the
    # argparser

    run_name: str = Field(
        default="",
        description=(
            "Name of the run, in case it helps you remember what changed. If supplied, "
            "this name gets appended to the run ID string: run-{timestamp}-{run_name}"
        ),
    )

    loss: Literal["mpnrl", "mnrl"] = Field(
        description="MPNRL is sigmoid-based. MNRL is softmax-based."
    )

    # Trainer
    per_device_train_batch_size: int = Field(
        description=(
            "Maximum number of anchors to sample per batch (may not be unique for "
            "MPNRL, and may not be reached for MNRL)."
        )
    )
    per_device_eval_batch_size: int = Field(
        default=32, description="Number of pairs to sample per batch during evaluation."
    )
    num_train_epochs: int = Field(default=1)
    seed: int = Field(default=42)

    # Dataset
    dataset_name: str = Field(description="Name of a HF dataset.")
    dataset_config: Optional[str] = Field(default=None)
    dataset_split_train: Optional[str] = Field(
        default="train", description="Training split name in HF."
    )
    dataset_split_val: Optional[str] = Field(
        default="dev",
        description="Validation/development split name in HF.",
    )
    dataset_size_train: Optional[int] = Field(
        default=None,
        description=(
            "Number of training observations to subsample. Will select the first N."
        ),
    )
    dataset_size_val: Optional[int] = Field(
        default=None,
        description=(
            "If dataset_split_val is not provided, then this number of validation "
            "observations will be randomly subsampled from the training set."
        ),
    )

    # Model
    model_name: str = Field(
        default="distilroberta-base",
        description="Name of HF model or local path to a SentenceTransformer.",
    )

    # Wandb
    wandb_project: str = Field(
        default="mpnrl",
        description="Name of the Wandb project to log training and eval info to.",
    )

    @model_validator(mode="after")
    def check_splits_are_different(self):
        if self.dataset_split_train == self.dataset_split_val:
            raise ValueError("The training and val splits must be different.")
        return self

    @model_validator(mode="after")
    def check_val_dataset(self):
        if (self.dataset_split_val is None) == (self.dataset_size_val is None):
            raise TypeError(
                "Exactly one of dataset_split_val or dataset_size_val must be provided."
            )
        return self


def _train_val_datasets(experiment: Experiment) -> tuple[Dataset, Dataset]:
    load_dataset_ = partial(
        load_dataset, path=experiment.dataset_name, name=experiment.dataset_config
    )
    train_dataset = load_dataset_(split=experiment.dataset_split_train)

    # Load validation data. Split it off train if it's not explicitly provided
    if experiment.dataset_split_val is not None:
        val_dataset = load_dataset_(split=experiment.dataset_split_val)
    else:
        dataset_dict = train_dataset.train_test_split(
            test_size=experiment.dataset_size_val, seed=experiment.seed
        )
        train_dataset: Dataset = dataset_dict["train"]
        val_dataset: Dataset = dataset_dict["test"]

    if experiment.dataset_size_train is not None:
        train_dataset = train_dataset.select(range(experiment.dataset_size_train))

    return train_dataset, val_dataset


def _bf16():
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if bf16:
        print("Using mixed precision in bf16")
    else:
        print("Not using mixed precision")
        # B/c I hardcode fp16 = False
    return bf16


def _trainer_args(
    experiment: Experiment,
    model: SentenceTransformer,
    train_dataset: Iterable[dict[str, str]],
):
    match experiment.loss:
        case "mnrl":
            custom_args = dict(
                batch_sampler=BatchSamplers.NO_DUPLICATES,
            )
            trainer_args = dict(
                loss=losses.MultipleNegativesRankingLoss(model),
                data_collator=None,
            )
        case "mpnrl":
            from mpnrl.collator import MPNRLDataCollator
            from mpnrl.loss import MultiplePositivesNegativesRankingLoss

            custom_args = dict(
                batch_sampler=BatchSamplers.BATCH_SAMPLER,
            )
            trainer_args = dict(
                loss=MultiplePositivesNegativesRankingLoss(model),
                data_collator=MPNRLDataCollator(
                    train_dataset, tokenize_fn=model.tokenize
                ),
            )
        case _:
            raise ValueError(
                f"{experiment.loss} not supported. Input one of: "
                f'{" ".join(get_args(Experiment.model_fields["loss"].annotation))}'
            )
    trainer_args["train_dataset"] = train_dataset
    return custom_args, trainer_args


def _stsb_evaluator(split: str):
    stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split=split)
    return EmbeddingSimilarityEvaluator(
        sentences1=stsb_eval_dataset["sentence1"],
        sentences2=stsb_eval_dataset["sentence2"],
        scores=stsb_eval_dataset["score"],
        main_similarity=SimilarityFunction.COSINE,
        show_progress_bar=True,
        write_csv=False,
    )


# Values are callables so that data loading only happens when needed.
dataset_name_to_val_evaluator_creator: dict[str, Callable[[], SentenceEvaluator]] = {
    "sentence-transformers/all-nli": partial(_stsb_evaluator, split="validation")
}
dataset_name_to_test_evaluator_creator: dict[str, Callable[[], SentenceEvaluator]] = {
    "sentence-transformers/all-nli": partial(_stsb_evaluator, split="test")
}


def _create_trainer(
    experiment: Experiment,
    model: SentenceTransformer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    run_id: str,
    results_dir: str,
):
    output_dir = os.path.join(results_dir, "trainer")
    common_training_args = dict(
        # Required arg:
        output_dir=output_dir,
        # Optional args:
        num_train_epochs=experiment.num_train_epochs,
        per_device_train_batch_size=experiment.per_device_train_batch_size,
        per_device_eval_batch_size=experiment.per_device_eval_batch_size,
        warmup_ratio=0.1,
        fp16=False,
        bf16=_bf16(),
        eval_strategy="steps",
        eval_steps=100,
        seed=experiment.seed,
        save_strategy="no",
        use_mps_device=False,
        # Wandb
        report_to="wandb",
        logging_steps=100,
    )
    custom_args, trainer_args = _trainer_args(experiment, model, train_dataset)
    # For Wandb
    custom_args["run_name"] = run_id

    val_evaluator_creator = dataset_name_to_val_evaluator_creator.get(
        experiment.dataset_name, lambda: None
    )
    val_evaluator = val_evaluator_creator()

    trainer = SentenceTransformerTrainer(
        model=model,
        args=SentenceTransformerTrainingArguments(
            **common_training_args, **custom_args
        ),
        **trainer_args,
        eval_dataset=val_dataset,  # evaluate val loss on iid data
        evaluator=val_evaluator,  # evaluate val accuracy on some downstream task
    )
    return trainer


def _dump_dict_to_json(d: dict, filepath: str) -> None:
    with open(filepath, "w") as json_file:
        json.dump(d, json_file, indent=4)


class _CUDAMemoryStats(BaseModel):
    peak_memory_allocated_gb: int | None = None
    peak_memory_reserved_gb: int | None = None
    snapshot: dict[str, Any] | None = None


@contextmanager
def _track_cuda_memory():
    # TODO: maybe use the torch profiler
    try:
        cuda_memory_stats = _CUDAMemoryStats()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            # Tell CUDA to start recording memory allocations
            torch.cuda.memory._record_memory_history(enabled="all")

        yield cuda_memory_stats
    finally:
        if torch.cuda.is_available():
            bytes_per_gb = 1024**3
            cuda_memory_stats.peak_memory_allocated_gb = (
                torch.cuda.max_memory_allocated() / bytes_per_gb
            )
            cuda_memory_stats.peak_memory_reserved_gb = (
                torch.cuda.max_memory_reserved() / bytes_per_gb
            )
            cuda_memory_stats.snapshot = torch.cuda.memory._snapshot()
            torch.cuda.memory._record_memory_history(enabled=None)


def _set_up_run(experiment: Experiment):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = (
        f"run-{current_time}{f'-{experiment.run_name}' if experiment.run_name else ''}"
    )
    results_dir = os.path.join(
        run_id,
        "dataset_reports",
        f"{experiment.loss}",
        f"batch_size_{experiment.per_device_train_batch_size}",
    )
    # Upload experiment settings
    os.makedirs(results_dir)
    _dump_dict_to_json(
        experiment.model_dump(), os.path.join(results_dir, "experiment.json")
    )

    print()
    print(f"{run_id=}")
    print(f"Results will be saved to {results_dir}")
    print()

    return run_id, results_dir


def _save_results(
    results_dir: str,
    result_cuda_memory: _CUDAMemoryStats,
    result_train: TrainOutput,
    result_test: dict[str, float],
):
    if result_cuda_memory.snapshot is not None:  # experiment was run on a CUDA machine
        with open(os.path.join(results_dir, "cuda_snapshot.pkl"), "wb") as f:
            pickle.dump(result_cuda_memory.snapshot, f)

    metrics = {
        "train_output": result_train._asdict(),
        "cuda_peak_memory_gb": {
            "allocated": result_cuda_memory.peak_memory_allocated_gb,
            "reserved": result_cuda_memory.peak_memory_reserved_gb,
        },
        "test_output": result_test,
    }
    _dump_dict_to_json(metrics, os.path.join(results_dir, "metrics.json"))


def run(experiment: Experiment):
    run_id, results_dir = _set_up_run(experiment)

    print("\n*********************** Loading train, val data ***********************\n")
    train_dataset, val_dataset = _train_val_datasets(experiment)

    print("\n****************************** Training ******************************\n")
    model = SentenceTransformer(experiment.model_name)
    trainer = _create_trainer(
        experiment, model, train_dataset, val_dataset, run_id, results_dir
    )
    with _track_cuda_memory() as result_cuda_memory:
        result_train: TrainOutput = trainer.train()

    print("\n********************* Evaluating on the test set *********************\n")
    test_evaluator = dataset_name_to_test_evaluator_creator[experiment.dataset_name]()
    result_test = test_evaluator(
        model, output_path=os.path.join(results_dir, "test_evaluator")
    )
    print(result_test)

    _save_results(results_dir, result_cuda_memory, result_train, result_test)
    if not os.listdir(trainer.args.output_dir):
        os.rmdir(trainer.args.output_dir)

    return run_id


if __name__ == "__main__":
    experiment = tapify(Experiment)
    print(experiment)

    os.environ["WANDB_PROJECT"] = experiment.wandb_project
    os.environ["WANDB_LOG_MODEL"] = "false"

    wandb.login(key=os.environ["WANDB_API_KEY"])

    run(experiment)
