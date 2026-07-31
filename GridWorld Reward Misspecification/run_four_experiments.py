import os
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()


EXPERIMENTS = [
    {
        "name": "Gemma 4 31B with reflection",
        "model": "google/gemma-4-31b-it",
        "log_file": "experiment_log_gemma_reflection.csv",
        "use_reflections": True,
    },
    {
        "name": "Gemma 4 31B without reflection",
        "model": "google/gemma-4-31b-it",
        "log_file": "experiment_log_gemma_no_reflection.csv",
        "use_reflections": False,
    },
    {
        "name": "Qwen3-VL 8B with reflection",
        "model": "qwen/qwen3-vl-8b-instruct",
        "log_file": "experiment_log_qwen_reflection.csv",
        "use_reflections": True,
    },
    {
        "name": "Qwen3-VL 8B without reflection",
        "model": "qwen/qwen3-vl-8b-instruct",
        "log_file": "experiment_log_qwen_no_reflection.csv",
        "use_reflections": False,
    },
]


def main():
    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit(
            "OpenRouter API key missing. Set OPENROUTER_API_KEY before running."
        )

    base_cmd = [
        sys.executable,
        "run_experiment.py",
        "--modes",
        "true",
        "proxy",
        "misleading",
        "delayed",
    ]

    for experiment in EXPERIMENTS:
        print("\n==============================")
        print("RUNNING:", experiment["name"])
        print("LOG:", experiment["log_file"])
        print("==============================")

        cmd = [
            *base_cmd,
            "--model",
            experiment["model"],
            "--log-file",
            experiment["log_file"],
        ]
        if not experiment["use_reflections"]:
            cmd.append("--no-reflections")

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
