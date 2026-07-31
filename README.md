# Reward Misspecification in Multimodal RL Agents — Research README

Short summary
-------------

This repository contains code and notebooks for experiments studying reward misspecification in multimodal RL agents. The project explores how LLM-driven agents learn and exploit reward functions in a small GridWorld-style environment. Primary artifacts are interactive notebooks, an environment implementation, LLM-agent code, and CSV logs of experiment outputs.

Key goals
---------
- Compare LLM-guided agent behaviours under different reward definitions (true, proxy, misleading, delayed).
- Measure reward misalignment (RMM) and other diagnostics across episodes and seeds.
- Provide reproducible scripts/notebooks to run experiments and collect logs.

Repository layout (important paths)
----------------------------------
- `GridWorld Object Manipulation/` — notebook-driven experiments with a simpler env and agent.
- `GridWorld_-Reward-Misspecification--main/` — alternative experiment folder containing scripts, environment implementation, and logs.
- `GSA/` — additional environment/server code and supporting utilities (see `GSA/README.md`).
- Top-level CSV outputs: `experiment_log*.csv` (per-model/per-condition results).

Requirements
------------
- Python 3.10+ recommended
- Create a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r "GridWorld Object Manipulation/requirements.txt"
# or for script-based experiments
pip install -r "GridWorld_-Reward-Misspecification--main/requirements.txt"
```

Environment variables (secrets)
------------------------------
This project uses the OpenRouter API (or other LLM providers). Do NOT commit real keys.

- Copy an example `.env` where needed:

```bash
cp GridWorld\ Object\ Manipulation/.env.example GridWorld\ Object\ Manipulation/.env
cp GridWorld_-Reward-Misspecification--main/.env.example GridWorld_-Reward-Misspecification--main/.env
```

- Edit `.env` and set:

```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
MODEL_NAME=google/gemma-4-31b-it  # optional override
```

Running experiments
-------------------

Notebook-driven (recommended for exploration):

- Open `GridWorld Object Manipulation/experiment.ipynb` in Jupyter and run cells. The notebook reads `.env` and will raise if `OPENROUTER_API_KEY` is missing.

Script-driven (batch runs / reproducible):

- If a `run_experiment.py` exists in a folder, run from that folder:

```bash
cd GridWorld_-Reward-Misspecification--main
python run_experiment.py --help
# Example: run the default experiment set
python run_experiment.py --episodes 10 --max-steps 30
```

- To run the full comparison set (if `run_four_experiments.py` exists):

```bash
python run_four_experiments.py
```

Outputs
-------
- CSV logs: `experiment_log.csv`, `experiment_log_gemma_*.csv`, `experiment_log_qwen_*.csv` — these contain per-episode metrics (reward, true_reward, RMM, steps, actions, reflections).
- Notebooks may also produce per-episode JSON or summary outputs depending on the folder.

Reproducibility notes
---------------------
- Set random seeds where available (notebooks and scripts may accept seeds). Use consistent `requirements.txt` and Python version.
- Experiments use an external LLM API — results will vary over time, across model versions, and with API rate limits.

Security & privacy
------------------
- Never store real API keys in the repo or in output logs. Use `.env` files and add them to `.gitignore`.

How to extend
-------------
- Add new reward modes to `openenv_env.py` (or `env.py`) and update `REWARD_MODES` in the agent code.
- Swap models by setting `MODEL_NAME` in `.env` or by passing model arguments to scripts.

References and citation
----------------------
If you use this code in published work, please cite the repository and include an appropriate description of the LLM models and versions used at time of experiment.

Contact
-------
Project maintainer: sjain090705@gmail.com

---
Generated README for the research project. If you want this moved to the repository root `README.md` or merged into an existing README, tell me and I will update it.
