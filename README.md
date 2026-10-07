# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

On Linux and Windows, this project pins **PyTorch 2.7.1 with CUDA 12.8**
through the official PyTorch wheel index. This build supports both the RTX
5070 Ti (`sm_120`) and RTX 3090 (`sm_86`). The original PyTorch 2.6 / CUDA 12.4
build does not support the 5070 Ti. See the
[PyTorch 2.7 release notes](https://pytorch.org/blog/pytorch-2-7/).
PyTorch wheels include their CUDA runtime dependencies; this project does not
need a separate CUDA Toolkit installation.

### Verify locally, then run on a 3090 server

Install the locked environment with Python 3.12:

```sh
uv sync --locked --python 3.12
```

For a quick local code check, use CPU. This requires no dataset downloads and
runs a tiny Transformer using synthetic token IDs stored as `uint16`:

```sh
uv run --locked verify_runtime.py --device cpu
```

The check executes data loading, forward and backward passes, gradient
clipping, AdamW updates, checkpoint restoration, and a resumed training step.
It fails on non-finite losses or gradients, unchanged weights, or a resumed
model that differs from uninterrupted training. It uses an in-memory checkpoint.

To verify the 5070 Ti or the 3090, explicitly select CUDA:

```sh
nvidia-smi
uv run --locked verify_runtime.py --device cuda
```

The CUDA check also resumes a GPU checkpoint on CPU. An explicit CUDA request
fails if CUDA is unavailable; it never silently falls back to CPU.
`--device auto` is available if automatic selection is wanted. Successful CPU execution
checks the code path; successful CUDA execution additionally checks the actual
GPU and driver environment. Neither check measures full training performance.

For a straightforward CUDA 12.8 setup, use an NVIDIA driver **570.26 or newer
on Linux** (the driver shipped with CUDA 12.8 GA), or a more recent driver.
Older drivers can have restrictions under CUDA minor-version compatibility.
See [NVIDIA's CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html).

When copying to the server, include the source files, `pyproject.toml`,
`uv.lock`, and `.python-version`, and recreate `.venv` on the server. For
example, package the code without the local environment or large datasets,
then copy the archive (replace the SSH destination and directory):

```sh
tar --exclude=.venv --exclude=.git --exclude=data --exclude=__pycache__ \
    --exclude=.pytest_cache -czf /tmp/cs336-basics-code.tar.gz .
scp /tmp/cs336-basics-code.tar.gz user@server:/path/to/project/
```

In that directory on the server:

```sh
tar -xzf cs336-basics-code.tar.gz
uv sync --locked --python 3.12
uv run --locked verify_runtime.py --device cuda
uv run --locked pytest
```

Copy or download the required datasets separately before full training. The
tokenizer and BPE training scripts run on CPU; they do not require CUDA.

### Run unit tests


```sh
uv run pytest
```

### Train BPE on TinyStories

Run both BPE implementations sequentially with progress bars, timing, and peak
RAM statistics:

```sh
uv run train_tinystories_bpe.py
```

The default target vocabulary size is 10,000. Results are written under
`artifacts/tinystories_bpe/{train_bpe,train_bpe_v2}/`, with `vocab.json`,
`merges.txt`, and `metadata.json` for each implementation. The metadata includes
the RAM baseline, peak, and increase in bytes. For a quick run or a custom
destination, use for example:

```sh
uv run train_tinystories_bpe.py --vocab-size 1000 --output-dir artifacts/bpe_1k
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
