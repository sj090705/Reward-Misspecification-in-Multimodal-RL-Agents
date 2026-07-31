# GridWorld Reward Misspecification Experiment

This directory contains an LLM-based agent exploring a 5×5 GridWorld environment to study reward misspecification in reinforcement learning.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```

3. Add your OpenRouter API key to `.env`:
   ```
   OPENROUTER_API_KEY=sk-or-v1-your-key-here
   ```

## Running

Open `experiment.ipynb` in Jupyter and run the cells to start experiments.

The notebook uses the `new_agent.py` module to interact with the environment via LLM calls and the `openenv_env.py` environment.

## Environment

- **Environment**: `RewardMisspecGridWorldEnv` in `openenv_env.py`
- **Agent**: LLM-based agent in `new_agent.py`
- **Reward Modes**: true, proxy, misleading, delayed
