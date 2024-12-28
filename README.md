# Multiple Positives and Negatives Ranking Loss

On AllNLI, training w/ MPNRL has higher training throughput and better memory
utilization than training w/ MNRL. They're on par in terms of task performance, but I
need to run more experiments.


## Why

No-duplicates sampling causes batch sizes to decay if there's high skewnewss in the
number of positives per anchors.

<details>
<summary>Plot for AllNLI, no-duplicates sampling</summary>

![](./images/dataloader/mnrl.png)

Reproduce by running:

```bash
python compare_dataloaders.py \
    --dataset_name "sentence-transformers/all-nli" \
    --dataset_config "triplet" \
    --dataset_split "train" \
    --batch_size 128 \
    --dataset_size_train 10000
```

</details>

Here are CUDA memory snapshots across time for MNRL + AllNLI (first 10k triplets,
inputted batch size of 200):

![](./images/memory_snapshots/mnrl.png)

The drops in memory are caused by drops in the batch size. There is a long tail of
under-utilization. Peak usage is determined by the first few batches, which is a small
portion of time.

It's simpler to use a loss which seamlessley handles multiple positives. As a result,
training throughput is higher, and GPU utilization (in terms of % memory and % time) is
more stable. Data loading itself is also 15x faster, as there's no de-duplication.

<details>
<summary>Plot for All-NLI, plain sampling with grouping in the collator</summary>

![](./images/dataloader/mpnrl.png)

</details>

Here are CUDA memory snapshots across time for MPNRL:

![](./images/memory_snapshots/mpnrl.png)

Here's a comparison of time-based GPU utilization:

<img src="./images/gpu_utilization_time.png" alt="drawing" width="400"/>

The small experiment in [`./demos/train_allnli.ipynb`](./demos/train_allnli.ipynb)
demonstrates that task/statistical performance is on par with MNRL.

In an experiment on the first 100k triplets in AllNLI and an inputted batch size of 200,
MNRL took ~33 minutes while MPNRL took ~20 minutes. Statistical performance was similar.


## Setup

```
python -m pip install git+https://github.com/kddubey/mpnrl.git
```

To run [`./run.py`](./run.py), clone the repo and then:

```
python -m pip install ".[demos]"
```

NOTE: this isn't meant to be a stable Python package. There are many TODOs.


## Usage

Make sure to **not** use the no-duplicates sampler for MPNRL.

```python
from sentence_transformers.sampler import BatchSamplers
from sentence_transformers import SentenceTransformerTrainer

import mpnrl

model = ...
train_dataset = ...
# Iterable of these records:
#   {"anchor": ..., "positive": ...}
# Can also have negatives:
#   {"anchor": ..., "positive": ..., "negative": ...}

trainer = SentenceTransformerTrainer(
    model=...,
    train_dataset=...,
    args=SentenceTransformerTrainingArguments(
        ...
        batch_sampler=BatchSamplers.BATCH_SAMPLER,
    ),
    loss=mpnrl.loss.MultiplePositivesNegativesRankingLoss(model),
    data_collator=mpnrl.collator.MPNRLDataCollator(
        train_dataset, tokenize_fn=model.tokenize
    ),
)

trainer.train()
```

There's a small demo in [`./demos/train_allnli.ipynb`](./demos/train_allnli.ipynb).


## TODOs

- [ ] `mpnrl.collator` `TODO`s.
- [ ] `mpnrl.loss` `TODO`s.
- [ ] Measure how long it takes for MNRL vs MPRNL to get to a good model
(pearson/spearman correlation on validation data).
- [ ] Repeat for a few datasets and study how the level of data duplication affects
these outcomes.
