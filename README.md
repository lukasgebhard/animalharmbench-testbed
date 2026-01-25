# Setup

## Prerequisites

- An x86_64 machine running Linux and supporting CUDA 12.8
- [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-1-download-archive?target_os=Linux&target_arch=x86_64)
- [git-lfs](https://git-lfs.com)
- [uv](https://docs.astral.sh/uv/)
- A logged-in Huggingface account with read access to sentientfutures/ahb

## Installation

(1) Set up the repo:

```sh
git clone git@github.com:lukasgebhard/animalharmbench-testbed.git
cd animalharmbench-testbed
```

(2) Set up the project's Python environment:

```sh    
uv venv --python 3.13
source .venv/bin/activate
uv sync --all-extras
```

# Usage

Create `.env` file.

```sh
uvx --from huggingface_hub hf auth login
python -m pipeline
```

# Stats

Using default settings:

SFT-Dataset-Generation on 2xH100: about 5h