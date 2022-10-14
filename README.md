# MEMIT: Mass-Editing Memory in a Transformer

Editing thousands of facts into a transformer memory at once.

<!-- [![Colab MEMIT Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kmeng01/memit/blob/main/notebooks/memit.ipynb) -->

## Table of Contents

- [Installation](#installation)
- [MEMIT Algorithm Demo](#memit-algorithm-demo)
- [Running the Full Evaluation Suite](#running-the-full-evaluation-suite)
- [Generating Scaling Curves](#generating-scaling-curves)
- [How to Cite](#how-to-cite)

## Installation

We recommend `conda` for managing Python, CUDA, and PyTorch; `pip` is for everything else. To get started, simply install `conda` and run:
```bash
CONDA_HOME=$CONDA_HOME ./scripts/setup_conda.sh
```

`$CONDA_HOME` should be the path to your `conda` installation, e.g., `~/miniconda3`.

## MEMIT Algorithm Demo

[`notebooks/memit.ipynb`](notebooks/memit.ipynb) demonstrates MEMIT. The API is simple; simply specify a *requested rewrite* of the following form:

```python
request = [
    {
        "prompt": "{} plays the sport of",
        "subject": "LeBron James",
        "target_new": {
            "str": "football"
        }
    },
    {
        "prompt": "{} plays the sport of",
        "subject": "Michael Jordan",
        "target_new": {
            "str": "baseball"
        }
    },
]
```

Other similar example(s) are included in the notebook.

## Running the Full Evaluation Suite

[`experiments/evaluate.py`](experiments/evaluate.py) can be used to evaluate any method in [`baselines/`](baselines/).

For example:
```
python3 -m experiments.evaluate \
    --alg_name=MEMIT \
    --model_name=EleutherAI/gpt-j-6B \
    --hparams_fname=EleutherAI_gpt-j-6B.json \
    --num_edits=10000 \
    --use_cache
```
Results from each run are stored at `results/<method_name>/run_<run_id>` in a specific format:
```bash
results/
|__ MEMIT/
    |__ run_<run_id>/
        |__ params.json
        |__ case_0.json
        |__ case_1.json
        |__ ...
        |__ case_10000.json
```

To summarize the results, you can use [`experiments/summarize.py`](experiments/summarize.py):
```bash
python3 -m experiments.summarize --dir_name=MEMIT --runs=run_<run1>,run_<run2>
```

Running `python3 -m experiments.evaluate -h` or `python3 -m experiments.summarize -h` provides details about command-line flags.

## How to Cite

```bibtex
@article{meng2022memit,
  title={Mass Editing Memory in a Transformer},
  author={Kevin Meng and Sen Sharma, Arnab and Alex Andonian and Yonatan Belinkov and David Bau},
  journal={arXiv preprint arXiv:2210.07229},
  year={2022}
}
```
