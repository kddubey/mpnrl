"""
Script to run MNRL or MPNRL training on an inputted dataset.
"""

from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from functools import partial
import json
from math import ceil
import os
import pickle
from typing import Any, Callable, Iterable, Literal, Optional, get_args

from datasets import Dataset, load_dataset
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sentence_transformers import (
    evaluation,
    losses,
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
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
    num_evals_per_epoch: int = Field(
        default=0,
        description=(
            "Number of times to compute the loss on validation data (and, if "
            "applicable, run the validation evaluator)."
        ),
    )
    seed: int = Field(default=42)
    disable_cuda_memory_snapshot: bool = Field(default=True)

    # Dataset
    dataset_name: str = Field(description="Name of a HF dataset.")
    dataset_config: Optional[str] = Field(
        default=None, description="Config/subset name in HF."
    )
    dataset_split_train: Optional[str] = Field(
        default="train", description="Training split name in HF."
    )
    dataset_split_val: Optional[str] = Field(
        default=None,
        description=(
            "Validation/development split name in HF. If not provided, the training "
            "split will be split into training and validation splits."
        ),
    )
    dataset_size_train: Optional[int] = Field(
        default=None,
        description=(
            "Number of training records to subsample. Will select the first N."
        ),
    )
    dataset_size_val: int = Field(
        default=1_000, description="Number of validation records."
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
    num_steps_to_log: int = Field(
        default=50, description="Number of times to log training info."
    )

    @model_validator(mode="after")
    def check_splits_are_different(self):
        if self.dataset_split_train == self.dataset_split_val:
            raise ValueError("The training and val splits must be different.")
        return self


def _train_val_datasets(experiment: Experiment) -> tuple[Dataset, Dataset]:
    load_dataset_ = partial(
        load_dataset, path=experiment.dataset_name, name=experiment.dataset_config
    )

    train_dataset = load_dataset_(split=experiment.dataset_split_train)
    if experiment.dataset_size_train is not None:
        train_dataset = train_dataset.select(range(experiment.dataset_size_train))

    # Load validation data. Split it off train if it's not explicitly provided
    if experiment.dataset_split_val is not None:
        val_dataset = load_dataset_(split=experiment.dataset_split_val)
        if experiment.dataset_size_val is not None:
            val_dataset = val_dataset.select(range(experiment.dataset_size_val))
    else:
        dataset_dict = train_dataset.train_test_split(
            test_size=experiment.dataset_size_val, seed=experiment.seed
        )
        train_dataset: Dataset = dataset_dict["train"]
        val_dataset: Dataset = dataset_dict["test"]

    return train_dataset, val_dataset


def _bf16():
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if bf16:
        print("\nUsing mixed precision in bf16\n")
    else:
        print("\nNot using mixed precision\n")
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
            from mpnrl.collator import GroupingDataCollator
            from mpnrl.loss import MultiplePositivesNegativesRankingLoss

            custom_args = dict(
                batch_sampler=BatchSamplers.BATCH_SAMPLER,
            )
            trainer_args = dict(
                loss=MultiplePositivesNegativesRankingLoss(model),
                data_collator=GroupingDataCollator(
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


def _stsb_evaluator(split: str, experiment: Experiment):
    stsb_eval_dataset = load_dataset("sentence-transformers/stsb", split=split)
    if split == "validation":
        stsb_eval_dataset = stsb_eval_dataset.select(range(experiment.dataset_size_val))
    return evaluation.EmbeddingSimilarityEvaluator(
        sentences1=stsb_eval_dataset["sentence1"],
        sentences2=stsb_eval_dataset["sentence2"],
        scores=stsb_eval_dataset["score"],
        main_similarity=evaluation.SimilarityFunction.COSINE,
        show_progress_bar=True,
        write_csv=False,
    )


def _ir_evaluator_from_mteb(
    dataset_name: str, per_device_eval_batch_size: int
) -> evaluation.InformationRetrievalEvaluator:
    queries = load_dataset("mteb/AILA_casedocs", "queries", split="queries")
    corpus = load_dataset("mteb/AILA_casedocs", "corpus", split="corpus")
    pair_labels = load_dataset("mteb/AILA_casedocs", "default", split="test")
    query_to_relevant_docs = defaultdict(set)
    for pair in pair_labels:
        query_to_relevant_docs[pair["query-id"]].add(pair["corpus-id"])
    return evaluation.InformationRetrievalEvaluator(
        queries={q["_id"]: q["text"] for q in queries},
        corpus={c["_id"]: c["text"] for c in corpus},
        relevant_docs=query_to_relevant_docs,
        corpus_chunk_size=per_device_eval_batch_size,
        show_progress_bar=True,
        write_csv=False,
    )


def _legal_evaluator(experiment: Experiment):
    # https://huggingface.co/bwang0911/jev2-legal#evaluation
    legal_dataset_names = [
        "mteb/AILA_casedocs",  # 50 queries, 186 documents
        "mteb/AILA_statutes",
        "mteb/legalbench_consumer_contracts_qa",
    ]
    return evaluation.SequentialEvaluator(
        [
            _ir_evaluator_from_mteb(dataset_name, experiment.per_device_eval_batch_size)
            for dataset_name in legal_dataset_names
        ]
    )


def _sql_questions_evaluator(experiment: Experiment):
    dataset = load_dataset("aladar/sql-questions", split="test")
    return evaluation.TripletEvaluator(
        anchors=dataset["query"],
        positives=dataset["positive"],
        negatives=dataset["negative"],
        show_progress_bar=True,
        write_csv=False,
    )


# Values are callables so that data loading only happens when needed.
_EvaluatorCreator = Callable[[Any, Experiment], evaluation.SentenceEvaluator]
dataset_name_to_val_evaluator_creator: dict[str, _EvaluatorCreator] = {
    "sentence-transformers/all-nli": partial(_stsb_evaluator, "validation"),
}
dataset_name_to_test_evaluator_creator: dict[str, _EvaluatorCreator] = {
    "sentence-transformers/all-nli": partial(_stsb_evaluator, "test"),
    "sentence-transformers/coliee": _legal_evaluator,
    "aladar/sql-questions": _sql_questions_evaluator,
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

    num_steps = len(train_dataset) // experiment.per_device_train_batch_size
    logging_steps = ceil(num_steps / experiment.num_steps_to_log)
    if experiment.num_evals_per_epoch <= 0:
        eval_args = dict(eval_strategy="no")
    else:
        eval_args = dict(
            eval_strategy="steps",
            eval_steps=ceil(num_steps / experiment.num_evals_per_epoch),
        )

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
        **eval_args,
        seed=experiment.seed,
        save_strategy="no",
        use_mps_device=False,
        # Wandb
        report_to="wandb",
        logging_steps=logging_steps,
    )
    custom_args, trainer_args = _trainer_args(experiment, model, train_dataset)
    # For Wandb
    custom_args["run_name"] = run_id

    val_evaluator_creator = dataset_name_to_val_evaluator_creator.get(
        experiment.dataset_name, lambda experiment: None
    )
    val_evaluator = val_evaluator_creator(experiment)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=SentenceTransformerTrainingArguments(
            **common_training_args, **custom_args
        ),
        **trainer_args,
        eval_dataset=val_dataset,  # evaluate val loss on iid data
        evaluator=val_evaluator,  # evaluate val accuracy on some downstream task(s)
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
def _track_cuda_memory(disable: bool = False):
    # TODO: maybe use the torch profiler
    enable = (not disable) and torch.cuda.is_available()
    try:
        cuda_memory_stats = _CUDAMemoryStats()

        if enable:
            torch.cuda.reset_peak_memory_stats()
            # Tell CUDA to start recording memory allocations
            torch.cuda.memory._record_memory_history(enabled="all")

        yield cuda_memory_stats
    finally:
        if enable:
            bytes_per_gb = 1024**3
            cuda_memory_stats.peak_memory_allocated_gb = (
                torch.cuda.max_memory_allocated() / bytes_per_gb
            )
            cuda_memory_stats.peak_memory_reserved_gb = (
                torch.cuda.max_memory_reserved() / bytes_per_gb
            )
            cuda_memory_stats.snapshot = torch.cuda.memory._snapshot()
            # Tell CUDA to stop recording memory allocations
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
    metrics = {
        "train_output": result_train._asdict(),
        "cuda_peak_memory_gb": {
            "allocated": result_cuda_memory.peak_memory_allocated_gb,
            "reserved": result_cuda_memory.peak_memory_reserved_gb,
        },
        "test_output": result_test,
    }
    _dump_dict_to_json(metrics, os.path.join(results_dir, "metrics.json"))

    if result_cuda_memory.snapshot is None:  # experiment was run on non-CUDA
        return
    print("\n************************ Saving CUDA snapshot ************************\n")
    with open(os.path.join(results_dir, "cuda_snapshot.pkl"), "wb") as f:
        pickle.dump(result_cuda_memory.snapshot, f)


def run(experiment: Experiment):
    run_id, results_dir = _set_up_run(experiment)

    print(
        "\n********************** Loading train, val, test data *********************\n"
    )
    train_dataset, val_dataset = _train_val_datasets(experiment)
    test_evaluator_creator = dataset_name_to_test_evaluator_creator[
        experiment.dataset_name
    ]
    test_evaluator = test_evaluator_creator(experiment)
    # Load the test evaluator b/c it'd be annoying to have that fail after training

    print("\n****************************** Training ******************************\n")
    model = SentenceTransformer(experiment.model_name)
    trainer = _create_trainer(
        experiment, model, train_dataset, val_dataset, run_id, results_dir
    )
    with _track_cuda_memory(
        disable=experiment.disable_cuda_memory_snapshot
    ) as result_cuda_memory:
        result_train: TrainOutput = trainer.train()

    print("\n********************* Evaluating on the test set *********************\n")
    result_test = test_evaluator(
        model, output_path=os.path.join(results_dir, "test_eval")
    )
    print(result_test)

    _save_results(results_dir, result_cuda_memory, result_train, result_test)

    if not os.listdir(trainer.args.output_dir):
        os.rmdir(trainer.args.output_dir)

    return run_id


if __name__ == "__main__":
    experiment = tapify(Experiment, description=__doc__)
    print(experiment)

    os.environ["WANDB_PROJECT"] = experiment.wandb_project
    os.environ["WANDB_LOG_MODEL"] = "false"

    wandb.login(key=os.environ["WANDB_API_KEY"])

    run(experiment)
