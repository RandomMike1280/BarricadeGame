"""
Plot MCTS speed across MCTS neural-network batch sizes and simulation counts.

Edit the lists/config values below directly, then run:

    python bench_plot.py
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any, Dict, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-barricade-bench"))

import matplotlib.pyplot as plt
import torch
from torch import nn

from barricade_env import BarricadeEnv, BarricadeState
from mcts import MCTS, MCTSConfig
from network import build_network


# Edit these lists directly for the benchmark grid you want.
NETWORK_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SIMULATION_COUNTS = [16, 32, 64, 128, 256, 512, 1024]

# Edit these settings directly too; no command-line args are used.
REPEATS = 3
POSITIONS = 1
SEARCHES_PER_POSITION = 1
WARMUP_PLIES = 12
WARMUP_SEARCHES = 1
SEED = 1
DEVICE: Optional[str] = None  # None => "cuda" when available, otherwise "cpu".
CHECKPOINT: Optional[str] = None

ACTION_TEMPERATURE = 0.0
POLICY_TARGET_TEMPERATURE: Optional[float] = None
ADD_ROOT_NOISE = False

NETWORK_CONFIG: Dict[str, Any] = {
    "history_length": 0,
    "conv_channels": 128,
    "residual_channels": None,
    "num_conv_layers": 1,
    "num_residual_layers": 10,
    "value_hidden_size": 256,
}

OUTPUT_DIR = Path("media")
PLOT_PATH = OUTPUT_DIR / "mcts_batch_sim_benchmark.png"


@dataclass(frozen=True)
class BenchResult:
    simulations: int
    batch_size: int
    repeat: int
    searches: int
    completed_simulations: int
    neural_batches: int
    evaluated_leaves: int
    collisions: int
    elapsed_seconds: float
    mean_latency_ms: float

    @property
    def searches_per_second(self) -> float:
        return self.searches / max(self.elapsed_seconds, 1.0e-12)

    @property
    def simulations_per_second(self) -> float:
        return self.completed_simulations / max(self.elapsed_seconds, 1.0e-12)

    @property
    def average_batch_size(self) -> float:
        return self.evaluated_leaves / max(1, self.neural_batches)


def main() -> None:
    seed_everything(SEED)
    device = resolve_device(DEVICE)
    model, network_config, checkpoint = load_model(device)
    history_length = int(network_config["history_length"])

    positions = make_positions(
        count=POSITIONS,
        warmup_plies=WARMUP_PLIES,
        seed=SEED,
    )

    results: list[BenchResult] = []
    total_runs = len(SIMULATION_COUNTS) * len(NETWORK_BATCH_SIZES) * REPEATS
    run_index = 0

    print(
        f"device={device} checkpoint={checkpoint or 'random-init'} "
        f"positions={POSITIONS} searches_per_position={SEARCHES_PER_POSITION} "
        f"repeats={REPEATS}"
    )

    for simulations in SIMULATION_COUNTS:
        for batch_size in NETWORK_BATCH_SIZES:
            for repeat in range(REPEATS):
                run_index += 1
                result = run_once(
                    model=model,
                    positions=positions,
                    simulations=simulations,
                    batch_size=batch_size,
                    repeat=repeat,
                    device=device,
                    history_length=history_length,
                    seed=SEED + repeat,
                )
                results.append(result)
                print_result(result, run_index=run_index, total_runs=total_runs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_results(results, PLOT_PATH)
    print(f"wrote {PLOT_PATH}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device: torch.device) -> tuple[nn.Module, Dict[str, Any], Optional[str]]:
    checkpoint_path = Path(CHECKPOINT) if CHECKPOINT else None
    checkpoint_config: Dict[str, Any] = {}
    payload: Any = None

    if checkpoint_path is not None:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        raw_config = payload.get("network_config", {}) if isinstance(payload, dict) else {}
        if isinstance(raw_config, dict):
            checkpoint_config = raw_config

    network_config = {
        key: checkpoint_config.get(key, value)
        for key, value in NETWORK_CONFIG.items()
    }
    model = build_network(**network_config).to(device)

    if payload is not None:
        state_dict = payload.get("model_state", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model, network_config, str(checkpoint_path) if checkpoint_path else None


def make_positions(*, count: int, warmup_plies: int, seed: int) -> list[BarricadeState]:
    if count <= 0:
        raise ValueError("POSITIONS must be positive.")
    if warmup_plies < 0:
        raise ValueError("WARMUP_PLIES must be non-negative.")

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


def run_once(
    *,
    model: nn.Module,
    positions: Sequence[BarricadeState],
    simulations: int,
    batch_size: int,
    repeat: int,
    device: torch.device,
    history_length: int,
    seed: int,
) -> BenchResult:
    mcts = MCTS(
        model,
        MCTSConfig(
            num_simulations=max(1, int(simulations)),
            batch_size=max(1, int(batch_size)),
            action_temperature=ACTION_TEMPERATURE,
            policy_target_temperature=POLICY_TARGET_TEMPERATURE,
            history_length=history_length,
            device=str(device),
            add_root_noise=ADD_ROOT_NOISE,
        ),
        device=device,
        rng=random.Random(seed),
    )

    for index in range(max(0, int(WARMUP_SEARCHES))):
        mcts.search(positions[index % len(positions)])

    synchronize(device)
    started = time.perf_counter()
    durations: list[float] = []
    completed_simulations = 0
    neural_batches = 0
    evaluated_leaves = 0
    collisions = 0

    for state in positions:
        for _ in range(SEARCHES_PER_POSITION):
            search_started = time.perf_counter()
            result = mcts.search(state)
            synchronize(device)
            durations.append(time.perf_counter() - search_started)

            diagnostics = result.diagnostics
            completed_simulations += int(diagnostics.get("completed_simulations", 0))
            neural_batches += int(diagnostics.get("neural_batches", 0))
            evaluated_leaves += int(diagnostics.get("evaluated_leaves", 0))
            collisions += int(diagnostics.get("collisions", 0))

    synchronize(device)
    elapsed = time.perf_counter() - started
    return BenchResult(
        simulations=simulations,
        batch_size=batch_size,
        repeat=repeat,
        searches=len(durations),
        completed_simulations=completed_simulations,
        neural_batches=neural_batches,
        evaluated_leaves=evaluated_leaves,
        collisions=collisions,
        elapsed_seconds=elapsed,
        mean_latency_ms=mean(durations) * 1000.0,
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def mean(values: Sequence[float]) -> float:
    return float(sum(values)) / max(1, len(values))


def grouped_results(results: Sequence[BenchResult]) -> dict[tuple[int, int], list[BenchResult]]:
    grouped: dict[tuple[int, int], list[BenchResult]] = {}
    for result in results:
        grouped.setdefault((result.simulations, result.batch_size), []).append(result)
    return grouped


def average_metric(group: Sequence[BenchResult], metric: str) -> float:
    return statistics.mean(getattr(result, metric) for result in group)


def print_result(result: BenchResult, *, run_index: int, total_runs: int) -> None:
    print(
        f"[{run_index:>3}/{total_runs}] "
        f"sims={result.simulations:<4} batch={result.batch_size:<4} "
        f"repeat={result.repeat + 1:<2} "
        f"sim/s={result.simulations_per_second:>9.1f} "
        f"latency={result.mean_latency_ms:>8.2f}ms "
        f"avg_batch={result.average_batch_size:>5.2f} "
        f"collisions={result.collisions}"
    )


def plot_results(results: Sequence[BenchResult], path: Path) -> None:
    grouped = grouped_results(results)

    plt.figure(figsize=(10, 6))
    for simulations in SIMULATION_COUNTS:
        y_values = [
            average_metric(grouped[(simulations, batch_size)], "simulations_per_second")
            for batch_size in NETWORK_BATCH_SIZES
        ]
        plt.plot(
            NETWORK_BATCH_SIZES,
            y_values,
            marker="o",
            linewidth=2.0,
            label=f"{simulations} simulations",
        )

    plt.xscale("log", base=2)
    plt.xticks(NETWORK_BATCH_SIZES, [str(batch_size) for batch_size in NETWORK_BATCH_SIZES])
    plt.xlabel("MCTS neural-network batch size")
    plt.ylabel("Completed simulations per second")
    plt.title("MCTS Speed vs Batch Size and Simulation Count")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend(title="Search budget")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
