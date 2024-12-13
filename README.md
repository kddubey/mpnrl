# Multiple Positives and Negatives Ranking Loss


## Setup

```
python -m pip install git+https://github.com/kddubey/mpnrl.git
```

NOTE: this isn't meant to be a stable Python package. There are many TODOs in the code.


## Why

No-duplicates sampling hurts training throughput if there are many duplicates. See the
numbers and plots for MNRL + AllNLI in
[`./demos/compare_dataloaders.ipynb`](./demos/compare_dataloaders.ipynb).

Here are CUDA memory snapshots across time for MNRL + AllNLI (first 10k triplets,
inputted batch size of 200):

![](./images/memory_snapshots/mnrl.png)

The drops in memory are caused by drops in the batch size.  There is a long tail of
under-utilization. Peak usage is determined by the first few batches, which is a small
portion of time.

It's simpler to use a loss which seamlessley handles multiple positives. As a result,
training throughput is higher, and GPU utilization (in terms of % memory and % time) is
more stable.

Here are CUDA memory snapshots across time for MPNRL:

![](./images/memory_snapshots/mpnrl.png)

Here's a comparison of time-based GPU utilization:

![](./images/gpu_utilization_time.png)

The small experiment in [`./demos/train_allnli.ipynb`](./demos/train_allnli.ipynb)
demonstrates that task/statistical performance is on par with MNRL.


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
```

There's a small demo in [`./demos/train_allnli.ipynb`](./demos/train_allnli.ipynb).


## TODOs

- [ ] `mpnrl.collator` `TODO`s
- [ ] `mpnrl.loss` `TODO`s
- [ ] Measure how long it takes for MNRL vs MPRNL to get to a good model
(pearson/spearman correlation on validation data).
- [ ] Repeat for a few datasets and study how the level of data duplication affects
these outcomes.
