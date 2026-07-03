"""
Plot MCTS speed across MCTS neural-network batch sizes and simulation counts.

Edit the lists/config values below directly, then run:

    python bench_plot.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any, Dict, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-barricade-bench"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from barricade_env import BarricadeEnv, BarricadeState
from mcts import MCTS, MCTSConfig
from network import build_network, infer_policy_head_type_from_state_dict


# Edit these lists directly for the benchmark grid you want.
NETWORK_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SIMULATION_COUNTS = [16, 32, 64, 128, 256, 512, 1024]

# Edit these settings directly too; no command-line args are used.
REPEATS = 3
POSITIONS = 10
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
    "policy_head_type": "factored",
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
        state_dict = payload.get("model_state", payload) if isinstance(payload, dict) else payload
        if isinstance(state_dict, dict) and "policy_head_type" not in checkpoint_config:
            checkpoint_config = dict(checkpoint_config)
            checkpoint_config["policy_head_type"] = infer_policy_head_type_from_state_dict(
                state_dict
            )

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
    colors = plt.get_cmap("tab10").colors

    figure, axis = plt.subplots(figsize=(11.5, 6.4), facecolor="white")
    axis.set_facecolor("white")

    for index, simulations in enumerate(SIMULATION_COUNTS):
        x_values = [float(batch_size) for batch_size in NETWORK_BATCH_SIZES]
        y_values = [
            average_metric(grouped[(simulations, batch_size)], "simulations_per_second")
            for batch_size in NETWORK_BATCH_SIZES
        ]
        smooth_x, smooth_y = smooth_curve(x_values, y_values)
        color = colors[index % len(colors)]

        axis.plot(
            smooth_x,
            smooth_y,
            color=color,
            linewidth=2.4,
            alpha=0.95,
            solid_capstyle="round",
            label=f"{simulations} simulations",
        )
        axis.plot(
            x_values,
            y_values,
            linestyle="none",
            marker="o",
            markersize=6.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.9,
            color=color,
        )

    axis.set_xscale("log", base=2)
    axis.set_xticks(NETWORK_BATCH_SIZES)
    axis.set_xticklabels([str(batch_size) for batch_size in NETWORK_BATCH_SIZES])
    axis.set_xlabel("MCTS neural-network batch size", fontsize=11)
    axis.set_ylabel("Completed simulations per second", fontsize=11)
    axis.set_title("MCTS Speed vs Batch Size and Simulation Count", fontsize=15, pad=12)
    axis.grid(True, which="major", color="#e5e7eb", linestyle="--", linewidth=0.85)
    axis.grid(True, which="minor", color="#f1f5f9", linestyle=":", linewidth=0.55)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#d1d5db")
    axis.spines["bottom"].set_color("#d1d5db")
    axis.tick_params(colors="#111827", labelsize=10)
    axis.margins(x=0.03, y=0.08)
    axis.legend(
        title="Search budget",
        frameon=True,
        facecolor="white",
        edgecolor="#d1d5db",
        fontsize=9.5,
        title_fontsize=10.5,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def smooth_curve(
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    points_per_segment: int = 28,
) -> tuple[list[float], list[float]]:
    if len(x_values) < 4:
        return list(x_values), list(y_values)

    log_x = [log2(value) for value in x_values]
    smooth_log_x: list[float] = []
    smooth_y: list[float] = []

    for index in range(len(log_x) - 1):
        x0 = log_x[max(0, index - 1)]
        x1 = log_x[index]
        x2 = log_x[index + 1]
        x3 = log_x[min(len(log_x) - 1, index + 2)]
        y0 = y_values[max(0, index - 1)]
        y1 = y_values[index]
        y2 = y_values[index + 1]
        y3 = y_values[min(len(y_values) - 1, index + 2)]

        for step in range(points_per_segment):
            if index > 0 or step > 0:
                t = step / points_per_segment
                smooth_log_x.append(catmull_rom(x0, x1, x2, x3, t))
                smooth_y.append(catmull_rom(y0, y1, y2, y3, t))

    smooth_log_x.append(log_x[-1])
    smooth_y.append(y_values[-1])
    return [2.0**value for value in smooth_log_x], smooth_y


def catmull_rom(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t
    )


def log2(value: float) -> float:
    return math.log2(value)


if __name__ == "__main__":
    main()
