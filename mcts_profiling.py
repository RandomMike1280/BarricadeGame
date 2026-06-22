"""
Profile the shared Barricade MCTS implementation with cProfile.

Examples:
    python mcts_profiling.py --simulations 64 --positions 4
    python mcts_profiling.py --checkpoint checkpoints/latest.pt --profile-out mcts.prof
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import random

import torch

from mcts_benchmark import (
    add_common_args,
    load_model,
    make_positions,
    print_summary,
    resolve_device,
    run_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile shared Barricade MCTS.")
    add_common_args(parser)
    parser.set_defaults(
        positions=4,
        searches_per_position=1,
        warmup_searches=0,
        simulations=64,
    )
    parser.add_argument("--profile-out", type=str, default="mcts_profile.prof")
    parser.add_argument("--stats-limit", type=int, default=40)
    parser.add_argument("--sort", type=str, default="cumtime")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    model, model_info = load_model(args, device)
    network_config = model_info["network_config"]
    positions = make_positions(
        count=args.positions,
        warmup_plies=args.warmup_plies,
        seed=args.seed,
    )

    profiler = cProfile.Profile()
    profiler.enable()
    summary = run_benchmark(
        model,
        positions,
        args,
        device=device,
        history_length=int(network_config["history_length"]),
        rng=random.Random(args.seed),
    )
    profiler.disable()

    print_summary(summary)
    if args.profile_out:
        profiler.dump_stats(args.profile_out)
        print(f"profile_out={args.profile_out}")

    print()
    pstats.Stats(profiler).strip_dirs().sort_stats(args.sort).print_stats(
        max(1, int(args.stats_limit))
    )


if __name__ == "__main__":
    main()
