import argparse
import csv
import os
import time

from dotenv import load_dotenv
from new_agent import REWARD_MODES, run_episode, setup_llm

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the GridWorld reward misspecification experiment.")
    parser.add_argument("--api-key", default=None,
                        help="OpenRouter API key. Defaults to OPENROUTER_API_KEY.")
    parser.add_argument("--model", default=None,
                        help="OpenRouter model. Defaults to OPENROUTER_MODEL or google/gemma-4-31b-it.")
    parser.add_argument("--api-retries", type=int, default=None,
                        help="OpenRouter retry attempts. Defaults to OPENROUTER_MAX_RETRIES or 3.")
    parser.add_argument("--api-retry-delay", type=float, default=None,
                        help="Base delay between OpenRouter retries in seconds.")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Episodes per reward mode.")
    parser.add_argument("--max-steps", type=int, default=30,
                        help="Maximum steps per episode.")
    parser.add_argument(
        "--modes", nargs="+", default=["true"], choices=REWARD_MODES, help="Reward modes to run.")
    parser.add_argument(
        "--log-file", default="experiment_log.csv", help="CSV output path.")
    parser.add_argument("--no-reflections", action="store_true",
                        help="Ask the model for actions only and leave reflection logs blank.")
    parser.add_argument("--step-delay", type=float, default=0,
                        help="Delay between LLM actions, in seconds.")
    parser.add_argument("--episode-delay", type=float,
                        default=0, help="Delay between episodes, in seconds.")
    return parser.parse_args()


def init_log(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "reward_mode",
                "episode",
                "success",
                "total_reward",
                "true_reward",
                "RMM",
                "steps",
                "actions",
                "reflections",
                "episode_summary",
            ]
        )


def append_log(log_file, row, episode_num):
    proxy = row["total_reward"]
    true = row["true_reward"]
    rmm = proxy - true

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                row["reward_mode"],
                episode_num,
                row["success"],
                proxy,
                true,
                rmm,
                row["steps"],
                row["actions"],
                row["reflections"],
                row["episode_summary"],
            ]
        )


def main():
    args = parse_args()
    setup_llm(args.api_key, args.model, args.api_retries, args.api_retry_delay)
    init_log(args.log_file)
    episode_histories = {mode: [] for mode in args.modes}

    for mode in args.modes:
        print("\n======================")
        print("RUNNING MODE:", mode)
        print("======================")

        for ep in range(args.episodes):
            result, episode_summary = run_episode(
                mode,
                ep + 1,
                args.max_steps,
                episode_histories[mode],  # Only within-mode history
                step_delay=args.step_delay,
                use_reflections=not args.no_reflections,
            )
            episode_histories[mode].append(episode_summary)
            append_log(args.log_file, result, ep + 1)

            print(
                f"episode={ep + 1} "
                f"reward={result['total_reward']} "
                f"success={result['success']} "
                f"steps={result['steps']}"
            )

            if args.episode_delay:
                time.sleep(args.episode_delay)


if __name__ == "__main__":
    main()
