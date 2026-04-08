---
title: SmartGrow AI
sdk: docker
app_port: 7860
tags:
  - openenv
---

# SmartGrow AI

SmartGrow AI is a lightweight plant-care simulator with a tabular reinforcement-learning agent, a CLI training loop, and a Streamlit dashboard for visualizing plant growth over time.

## Project Layout

- `env/`: simulation state, weather, reward, and environment adapter code
- `agent/`: tabular DQN-style agent, policy, replay buffer, and baseline agent
- `training/`: training config, training loop, and evaluation
- `tasks/`: scenario and benchmark task definitions
- `ui/`: terminal report helpers and Streamlit dashboard
- `config/`: runtime defaults loaded by the CLI and dashboard
- `config/openenv.yaml`: explicit OpenEnv spec artifact
- `tests/`: regression tests for env, reward, config loading, and adapter behavior

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Training From CLI

```powershell
.\.venv\Scripts\python main.py
```

You can override runtime parameters:

```powershell
.\.venv\Scripts\python main.py --scenario hot_dry --episodes 120 --days 40 --seed 11
```

## Visualize In Streamlit

```powershell
.\.venv\Scripts\python -m streamlit run ui\streamlit_app.py
```

The dashboard lets you train an agent, step the simulation day by day, and watch growth, health, water, nutrient, and reward trends update live.

## Baseline Benchmark

```powershell
$env:HF_TOKEN="hf_xxx"
$env:HF_MODEL="openai/gpt-oss-20b"
.\.venv\Scripts\python baseline_inference.py
```

The baseline runner uses the OpenAI Python client against Hugging Face's OpenAI-compatible router. It reads credentials from `HF_TOKEN`, uses fixed task seeds by default, runs all benchmark tasks, and prints a reproducible per-task score plus `aggregate_score`.

## Tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

## Docker

Build the image:

```powershell
docker build -t smartgrow-ai .
```

Run the Hugging Face Space container locally:

```powershell
docker run --rm -p 7860:7860 smartgrow-ai
```

Then open `http://localhost:7860`.

Run the baseline benchmark inside the container:

```powershell
docker run --rm -e HF_TOKEN=hf_xxx -e HF_MODEL=openai/gpt-oss-20b smartgrow-ai python baseline_inference.py
```

## Hugging Face Spaces

Use a Docker Space and point it at this repository. The repository metadata at the top of this `README.md` already declares:

- `sdk: docker`
- `app_port: 7860`
- `tags: [openenv]`

Set these Space secrets:

- `HF_TOKEN`: token used by the OpenAI-compatible baseline evaluator
- `HF_MODEL`: optional model override for `baseline_inference.py`
