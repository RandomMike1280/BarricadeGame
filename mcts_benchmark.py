"""
Benchmark the shared Barricade MCTS implementation.

Examples:
    python mcts_benchmark.py --simulations 64 --positions 8
    python mcts_benchmark.py --checkpoint checkpoints/latest.pt --simulations 256
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import time
from typing import Any, Dict, Optional, Sequence

import torch
from torch import nn

from barricade_env import BarricadeEnv, BarricadeState
from mcts import MCTS, MCTSConfig
from network import build_network, infer_policy_head_type_from_state_dict

print(torch.get_num_threads())
print(torch.get_num_interop_threads())

NETWORK_DEFAULTS = {
    "history_length": 0,
    "conv_channels": 128,
    "residual_channels": None,
    "num_conv_layers": 1,
    "num_residual_layers": 10,
    "value_hidden_size": 256,
    "policy_head_type": "factored",
}


@dataclass(frozen=True)
class BenchmarkSummary:
    searches: int
    positions: int
    requested_simulations: int
    completed_simulations: int
    neural_batches: int
    evaluated_leaves: int
    collisions: int
    elapsed_seconds: float
    durations: Sequence[float]
    device: str
    history_length: int
    checkpoint: Optional[str]

    @property
    def searches_per_second(self) -> float:
        return self.searches / max(self.elapsed_seconds, 1.0e-12)

    @property
    def simulations_per_second(self) -> float:
        return self.completed_simulations / max(self.elapsed_seconds, 1.0e-12)

    @property
    def average_batch_size(self) -> float:
        return self.evaluated_leaves / max(1, self.neural_batches)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--positions", type=int, default=64)
    parser.add_argument("--searches-per-position", type=int, default=1)
    parser.add_argument("--warmup-plies", type=int, default=12)
    parser.add_argument("--warmup-searches", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--action-temperature", type=float, default=0.0)
    parser.add_argument("--policy-target-temperature", type=float, default=None)
    parser.add_argument("--add-root-noise", action="store_true")
    parser.add_argument("--history-length", type=int, default=None)
    parser.add_argument("--conv-channels", type=int, default=None)
    parser.add_argument("--residual-channels", type=int, default=None)
    parser.add_argument("--num-conv-layers", type=int, default=None)
    parser.add_argument("--num-residual-layers", type=int, default=None)
    parser.add_argument("--value-hidden-size", type=int, default=None)
    parser.add_argument(
        "--policy-head-type",
        choices=("factored", "flat"),
        default=None,
    )


def resolve_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, Dict[str, Any]]:
    payload: Any = None
    checkpoint_config: Dict[str, Any] = {}
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None

    if checkpoint_path is not None:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        raw_config = payload.get("network_config", {}) if isinstance(payload, dict) else {}
        if isinstance(raw_config, dict):
            checkpoint_config = raw_config
        state_dict = payload.get("model_state", payload) if isinstance(payload, dict) else payload
        if isinstance(state_dict, dict) and "policy_head_type" not in checkpoint_config:
            checkpoint_config = dict(checkpoint_config)
            checkpoint_config["policy_head_type"] = infer_policy_head_type_from_state_dict(
                state_dict
            )

    network_config = _resolved_network_config(args, checkpoint_config)
    model = build_network(**network_config).to(device)

    if payload is not None:
        state_dict = payload.get("model_state", payload) if isinstance(payload, dict) else payload
        try:
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint is not compatible with network.py's 9x9 AlphaZeroNetwork."
            ) from exc

    model.eval()
    return model, {
        "network_config": network_config,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
    }


def make_positions(
    *,
    count: int,
    warmup_plies: int,
    seed: int,
) -> list[BarricadeState]:
    if count <= 0:
        raise ValueError("--positions must be positive.")
    if warmup_plies < 0:
        raise ValueError("--warmup-plies must be non-negative.")

    rng = random.Random(seed)
    positions: list[BarricadeState] = []
    for _ in range(count):
        env = BarricadeEnv(
            max_steps=max(1, warmup_plies + 1),
            invalid_action_mode="raise",
        )
        env.reset(seed=rng.randrange(2**31))
        plies = rng.randint(0, warmup_plies)
        for _ in range(plies):
            legal_actions = env.legal_actions()
            if not legal_actions:
                break
            _, _, terminated, truncated, _ = env.step(rng.choice(legal_actions))
            if terminated or truncated:
                break
        positions.append(env.state.copy())
    return positions


def build_mcts_config(
    args: argparse.Namespace,
    *,
    device: torch.device,
    history_length: int,
) -> MCTSConfig:
    return MCTSConfig(
        num_simulations=max(1, int(args.simulations)),
        batch_size=max(1, int(args.batch_size)),
        action_temperature=float(args.action_temperature),
        policy_target_temperature=args.policy_target_temperature,
        history_length=history_length,
        device=str(device),
        add_root_noise=bool(args.add_root_noise),
    )


def run_benchmark(
    model: nn.Module,
    positions: Sequence[BarricadeState],
    args: argparse.Namespace,
    *,
    device: torch.device,
    history_length: int,
    rng: Optional[random.Random] = None,
) -> BenchmarkSummary:
    if args.searches_per_position <= 0:
        raise ValueError("--searches-per-position must be positive.")

    rng = rng or random.Random(args.seed)
    mcts = MCTS(
        model,
        build_mcts_config(args, device=device, history_length=history_length),
        device=device,
        rng=rng,
    )

    for index in range(max(0, int(args.warmup_searches))):
        mcts.search(positions[index % len(positions)])

    _synchronize(device)
    started = time.perf_counter()
    durations: list[float] = []
    completed_simulations = 0
    neural_batches = 0
    evaluated_leaves = 0
    collisions = 0

    for state in positions:
        for _ in range(args.searches_per_position):
            search_started = time.perf_counter()
            result = mcts.search(state)
            _synchronize(device)
            durations.append(time.perf_counter() - search_started)
            diagnostics = result.diagnostics
            completed_simulations += int(diagnostics.get("completed_simulations", 0))
            neural_batches += int(diagnostics.get("neural_batches", 0))
            evaluated_leaves += int(diagnostics.get("evaluated_leaves", 0))
            collisions += int(diagnostics.get("collisions", 0))

    _synchronize(device)
    elapsed = time.perf_counter() - started
    return BenchmarkSummary(
        searches=len(durations),
        positions=len(positions),
        requested_simulations=int(args.simulations),
        completed_simulations=completed_simulations,
        neural_batches=neural_batches,
        evaluated_leaves=evaluated_leaves,
        collisions=collisions,
        elapsed_seconds=elapsed,
        durations=durations,
        device=str(device),
        history_length=history_length,
        checkpoint=args.checkpoint,
    )


def print_summary(summary: BenchmarkSummary) -> None:
    mean_ms = _mean(summary.durations) * 1000.0
    p50_ms = _percentile(summary.durations, 0.50) * 1000.0
    p95_ms = _percentile(summary.durations, 0.95) * 1000.0
    print("mcts_benchmark")
    print(
        f"device={summary.device} checkpoint={summary.checkpoint or 'random-init'} "
        f"history_length={summary.history_length}"
    )
    print(
        f"positions={summary.positions} searches={summary.searches} "
        f"requested_simulations={summary.requested_simulations} "
        f"completed_simulations={summary.completed_simulations}"
    )
    print(
        f"elapsed={summary.elapsed_seconds:.3f}s "
        f"searches_per_second={summary.searches_per_second:.2f} "
        f"simulations_per_second={summary.simulations_per_second:.2f}"
    )
    print(
        f"latency_ms mean={mean_ms:.2f} p50={p50_ms:.2f} p95={p95_ms:.2f} "
        f"neural_batches={summary.neural_batches} "
        f"avg_batch={summary.average_batch_size:.2f} "
        f"collisions={summary.collisions}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark shared Barricade MCTS.")
    add_common_args(parser)
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
    summary = run_benchmark(
        model,
        positions,
        args,
        device=device,
        history_length=int(network_config["history_length"]),
        rng=random.Random(args.seed),
    )
    print_summary(summary)


def _resolved_network_config(
    args: argparse.Namespace,
    checkpoint_config: Dict[str, Any],
) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    for key, default in NETWORK_DEFAULTS.items():
        arg_value = getattr(args, key)
        config[key] = arg_value if arg_value is not None else checkpoint_config.get(key, default)
    return config


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / len(values)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


if __name__ == "__main__":
    main()
