# Prior Data Generation & Dataloaders

This directory contains the synthetic-prior data generation pipeline and the
dataloaders used to feed synthetic tabular datasets into MotherTree training.

There are two prior families:

- **TabICL v1** (also used by MotherNet) — a bag of independent generative
  priors: MLP, Gaussian Process (GP), and (optionally) a tree-structured
  causal model (TreeSCM). Each dataset is a single batch saved as a `.pkl`
  file.
- **TabICL v2** — a graph-based structural causal model (`GraphPrior`) driven
  by a JSON config. Each dataset is saved as a `.pt` file.

---

## 1. Data generation

Both pipelines are driven by `prior/generate.py`. The two convenience scripts
wrap it with sensible defaults.

### 1.1 TabICL v1 / MotherNet — `generate_tabiclv1.sh`

```bash
bash prior/generate_tabiclv1.sh
```

which runs:

```bash
python prior/generate.py \
    --save_dir /Users/code/mt_tree/dataset \
    --dataset_type mothernet \
    --padding False \
    --device cpu --n_jobs 1 \
    --batch_size 256 \
    --max_seq_len 1024 \
    --max_features 100 \
    --prior_type prior_bag \
    --num_batches 1
```

Notes:

- `--dataset_type` can be `tabiclv1` or `mothernet`. Both use the **TabICL v1**
  generators; the value only selects the MLP-prior configuration
  (`mothernet_mlp_config()` vs `tabiclv1_mlp_config()` in
  `data_prior_config.py`).
- `--prior_type` selects which priors are mixed (see table below).

| `prior_type`        | Composition                                                          |
|---------------------|----------------------------------------------------------------------|
| `prior_bag`         | MLP (0.961) + GP (0.039)                                             |
| `prior_bag_plus_tree` | MLP (0.661) + GP (0.039) + TreeSCM (0.30)                          |
| `tree_only`         | TreeSCM (1.0) only                                                   |

**Output format** — one file per batch:

```
{save_dir}/{batch_idx}.pkl
```

Each `.pkl` contains a dict with keys `x`, `y`, `single_eval_pos`, `n_samples`,
`info`.

### 1.2 TabICL v2 — `generate_tabiclv2.sh`

```bash
bash prior/generate_tabiclv2.sh
```

which runs:

```bash
cd mt_tree
python prior/generate.py \
    --dataset_type tabiclv2 \
    --save_dir /Users/code/mt_tree/dataset \
    --dataset_config_path tabiclv2/graph_lib/config/stage1.json \
    --num_batches 10 \
    --n_jobs 2
```

Notes:

- `--dataset_config_path` is **required** for `tabiclv2`; it points to the JSON
  file consumed by `PriorConfig` (e.g. `tabiclv2/graph_lib/config/stage1.json`).
- Generation uses the graph-based `GraphPrior` (structural causal model).
- Use `--resume_from` to continue a previous run without regenerating earlier
  batches.

**Output format** — one file per batch:

```
{save_dir}/batch_{batch_idx:06d}.pt
```

Each `.pt` contains `X`, `y`, `d`, `seq_lens`, `train_sizes`, and `batch_size`.

### 1.3 Common `generate.py` arguments

| Argument            | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `--dataset_type`    | `tabiclv1`, `mothernet`, or `tabiclv2`                                      |
| `--prior_type`      | `prior_bag`, `prior_bag_plus_tree`, `tree_only` (v1 only)                   |
| `--save_dir`        | Directory where generated batches are written                               |
| `--num_batches`     | Total number of batches to generate                                         |
| `--resume_from`     | Batch index to resume from                                                  |
| `--batch_size`      | Number of datasets per batch                                                |
| `--min_features` / `--max_features` | Feature-count range per dataset                          |
| `--min_seq_len` / `--max_seq_len`   | Sequence-length range per dataset                     |
| `--max_classes`     | Maximum number of classes (binary tasks are sampled with `--binary_class_p`)|
| `--min_train_size` / `--max_train_size` | Train split position (int) or ratio (float in (0,1)) |
| `--log_seq_len`     | Sample sequence length log-uniformly                                        |
| `--n_jobs`          | Parallel workers (`joblib`); `1` = single-threaded                          |
| `--device`          | `cpu` or `cuda`                                                             |
| `--dataset_config_path` | JSON config for `tabiclv2` only                                          |

---

## 2. Dataloaders

There are two ways to consume data: generate **on the fly** (no disk I/O) or
**read pre-generated batches from disk**.

### 2.1 TabICL v1 — on-the-fly

`prior/tabiclv1/dataloader.py` exposes `get_dataloader`, which builds a
`PriorDataLoader` that samples synthetic datasets live:

```python
from prior.tabiclv1.dataloader import get_dataloader

dataloader = get_dataloader(prior_config, dataloader_config, device)
for batch in dataloader:
    ...
```

`generate_one_batch(batch_idx, config)` is the offline counterpart used by
`generate.py` to write `.pkl` batches to disk.

### 2.2 TabICL v2 — on-the-fly(Extremely slow, never used in practice)

`get_tabiclv2_dataset(dataset_config_path)` returns a `(dataset, dataset_config)`
pair where `dataset` is a `PriorDataset` (an infinite `IterableDataset`):

```python
from prior.tabiclv2.dataloader import get_tabiclv2_dataset

dataset, dataset_config = get_tabiclv2_dataset(
    dataset_config_path="/Users/code/mt_tree/prior/tabiclv2/graph_lib/config/stage1.json"
)
X, y, d, seq_lens, train_sizes = dataset.get_batch()
```

### 2.3 TabICL v2 — disk-based

`TabICLv2DiskDataLoader` iterates over the `batch_XXXXXX.pt` files produced by
`generate_tabiclv2.sh`. It supports `shuffle`, `max_batches`, and DDP
(file-strided sharding across ranks):

```python
from prior.tabiclv2.dataloader import TabICLv2DiskDataLoader

dataloader = TabICLv2DiskDataLoader(
    data_dir="/Users/code/mt_tree/dataset",
    batch_size=4,
    shuffle=True,
)
for X, y, d, seq_lens, train_sizes in dataloader:
    ...
```

---

## 3. Run example

A complete end-to-end example lives in `prior/dataloader.py`:

```python
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prior.tabiclv2.dataloader import get_tabiclv2_dataset, TabICLv2DiskDataLoader

# (a) On-the-fly dataset from a JSON config
dataset, dataset_config = get_tabiclv2_dataset(
    dataset_config_path="/Users/code/mt_tree/prior/tabiclv2/graph_lib/config/stage1.json"
)
print(dataset.get_batch())

# (b) Disk dataloader over pre-generated batches
dataloader = TabICLv2DiskDataLoader(
    data_dir="/Users/code/mt_tree/dataset", batch_size=4, shuffle=True
)

for i, (X, y, d, seq_lens, train_sizes) in enumerate(dataloader):
    for tensor_X, tensor_y in zip(X, y):
        print(tensor_X.shape, tensor_y.shape)
    print(type(X), type(y), d, seq_lens, train_sizes)
    break
```

Run it from the repository root so that `prior` is importable:

```bash
python /Users/code/mt_tree/prior/dataloader.py
```

---

## 4. Notes

- **Import path.** Scripts import the package with `from prior.xxx import ...`,
  so the repository root (`mt_tree/`, the parent of `prior/`) must be on
  `sys.path`. Entry scripts do this at the top, e.g.:

  ```python
  sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
  ```

- **Determinism.** Each batch is generated with seed
  `(base_seed + batch_idx * 10007) % 2**31`, so a given batch index is
  reproducible across runs with the same `--np_seed`.
- **Resuming.** Pass `--resume_from N` to skip batches `< N`; existing files are
  left untouched.
- **DDP.** `TabICLv2DiskDataLoader` shards files by rank stride
  (`rank 0 → batch_000000, batch_000004, ...`), so each process reads disjoint
  batches.
