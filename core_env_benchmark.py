"""
Benchmark core Barricade environment and rule-engine functions.

Examples:
    python core_env_benchmark.py
    python core_env_benchmark.py --positions 128 --iterations 2000
    python core_env_benchmark.py --only legal_actions_uncached,apply_move,env_step
    python core_env_benchmark.py --csv-out core_env_benchmark.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import random
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from barricade_env import (
    BarricadeEnv,
    BarricadeState,
    Move,
    Player,
    WallOrientation,
)


BenchmarkRunner = Callable[[int], int]


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    runner: BenchmarkRunner
    iterations: int
    note: str = ""


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    iterations: int
    best_seconds: float
    median_seconds: float
    checksum: int
    note: str

    @property
    def best_us_per_call(self) -> float:
        return self.best_seconds * 1_000_000.0 / max(1, self.iterations)

    @property
    def median_us_per_call(self) -> float:
        return self.median_seconds * 1_000_000.0 / max(1, self.iterations)

    @property
    def best_calls_per_second(self) -> float:
        return self.iterations / max(self.best_seconds, 1.0e-12)


def make_positions(
    *,
    count: int,
    warmup_plies: int,
    seed: int,
) -> List[BarricadeState]:
    if count <= 0:
        raise ValueError("--positions must be positive.")
    if warmup_plies < 0:
        raise ValueError("--warmup-plies must be non-negative.")

    rng = random.Random(seed)
    positions: List[BarricadeState] = []

    for _ in range(count):
        state = BarricadeState()
        plies = rng.randint(0, warmup_plies)
        for _ in range(plies):
            legal_action_moves = state.legal_action_moves()
            if not legal_action_moves:
                break
            _, move = rng.choice(legal_action_moves)
            state = state.apply_move(move)
            if state.winner is not None:
                break
        positions.append(state.copy())

    return positions


def sample_action_moves(
    positions: Sequence[BarricadeState],
    *,
    seed: int,
) -> List[Tuple[BarricadeState, int, Move]]:
    rng = random.Random(seed)
    samples: List[Tuple[BarricadeState, int, Move]] = []
    for state in positions:
        legal_action_moves = state.legal_action_moves()
        if not legal_action_moves:
            continue
        action, move = rng.choice(legal_action_moves)
        samples.append((state, action, move))
    if not samples:
        raise ValueError("No legal moves were available in the generated positions.")
    return samples


def clear_rule_caches(state: BarricadeState) -> BarricadeState:
    """Return a copy with per-state memoized rule results cleared."""
    copied = state.copy()
    copied._path_cache = {}
    copied._route_cache = {}
    copied._valid_moves_cache_key = None
    copied._valid_moves_cache = None
    copied._valid_action_moves_cache = None
    return copied


def build_cases(
    *,
    positions: Sequence[BarricadeState],
    samples: Sequence[Tuple[BarricadeState, int, Move]],
    iterations: int,
    step_iterations: int,
    seed: int,
) -> Dict[str, BenchmarkCase]:
    if not positions:
        raise ValueError("positions cannot be empty.")
    if not samples:
        raise ValueError("samples cannot be empty.")

    position_count = len(positions)
    sample_count = len(samples)

    for state in positions:
        state.legal_action_moves()
        state.action_mask()
        state.shortest_path_length(Player.RED)
        state.shortest_path_length(Player.BLUE)

    def pawn_moves(count: int) -> int:
        checksum = 0
        for index in range(count):
            checksum += len(positions[index % position_count].get_pawn_moves())
        return checksum

    def wall_moves_uncached(count: int) -> int:
        checksum = 0
        for index in range(count):
            state = positions[index % position_count].copy()
            checksum += len(state.get_valid_wall_moves())
        return checksum

    def legal_moves_cached(count: int) -> int:
        checksum = 0
        for index in range(count):
            checksum += len(positions[index % position_count].legal_moves())
        return checksum

    def legal_action_moves_cached(count: int) -> int:
        checksum = 0
        for index in range(count):
            checksum += len(positions[index % position_count].legal_action_moves())
        return checksum

    def legal_actions_uncached(count: int) -> int:
        checksum = 0
        for index in range(count):
            state = positions[index % position_count].copy()
            checksum += len(state.legal_actions())
        return checksum

    def action_mask_uncached(count: int) -> int:
        checksum = 0
        for index in range(count):
            state = positions[index % position_count].copy()
            checksum += sum(state.action_mask())
        return checksum

    def shortest_path_cached(count: int) -> int:
        checksum = 0
        for index in range(count):
            state = positions[index % position_count]
            red_distance = state.shortest_path_length(Player.RED)
            blue_distance = state.shortest_path_length(Player.BLUE)
            checksum += (red_distance or 0) + (blue_distance or 0)
        return checksum

    def shortest_path_uncached(count: int) -> int:
        checksum = 0
        for index in range(count):
            state = clear_rule_caches(positions[index % position_count])
            red_distance = state.shortest_path_length(Player.RED)
            blue_distance = state.shortest_path_length(Player.BLUE)
            checksum += (red_distance or 0) + (blue_distance or 0)
        return checksum

    def greedy_route_uncached(count: int) -> int:
        checksum = 0
        for index in range(count):
            state = clear_rule_caches(positions[index % position_count])
            red_route = state.greedy_path_cells(Player.RED)
            blue_route = state.greedy_path_cells(Player.BLUE)
            checksum += len(red_route or ()) + len(blue_route or ())
        return checksum

    def wall_shape_available(count: int) -> int:
        checksum = 0
        for index in range(count):
            state = positions[index % position_count]
            row = index % state.wall_board_size
            col = (index // state.wall_board_size) % state.wall_board_size
            orientation = (
                WallOrientation.HORIZONTAL
                if index % 2 == 0
                else WallOrientation.VERTICAL
            )
            checksum += int(state.is_wall_shape_available(orientation, row, col))
        return checksum

    def valid_wall_placement(count: int) -> int:
        checksum = 0
        for index in range(count):
            state = positions[index % position_count]
            row = index % state.wall_board_size
            col = (index // state.wall_board_size) % state.wall_board_size
            orientation = (
                WallOrientation.HORIZONTAL
                if index % 2 == 0
                else WallOrientation.VERTICAL
            )
            checksum += int(state.is_valid_wall_placement(orientation, row, col))
        return checksum

    def apply_move(count: int) -> int:
        checksum = 0
        for index in range(count):
            state, _, move = samples[index % sample_count]
            next_state = state.apply_move(move)
            checksum += next_state.current_player.value + len(next_state.walls)
        return checksum

    def apply_action_no_validate(count: int) -> int:
        checksum = 0
        for index in range(count):
            state, action, _ = samples[index % sample_count]
            next_state = state.apply_action(action, validate=False)
            checksum += next_state.current_player.value + len(next_state.walls)
        return checksum

    def apply_action_validate(count: int) -> int:
        checksum = 0
        for index in range(count):
            state, action, _ = samples[index % sample_count]
            next_state = state.apply_action(action, validate=True)
            checksum += next_state.current_player.value + len(next_state.walls)
        return checksum

    def env_observation(count: int) -> int:
        checksum = 0
        env = BarricadeEnv(invalid_action_mode="raise")
        for index in range(count):
            env.state = positions[index % position_count].copy()
            obs = env._get_observation()
            mask = obs["action_mask"]
            checksum += int(mask.sum() if hasattr(mask, "sum") else sum(mask))
        return checksum

    def env_legal_action_mask(count: int) -> int:
        checksum = 0
        env = BarricadeEnv(invalid_action_mode="raise")
        for index in range(count):
            env.state = positions[index % position_count].copy()
            mask = env.legal_action_mask()
            checksum += int(mask.sum() if hasattr(mask, "sum") else sum(mask))
        return checksum

    def env_step(count: int) -> int:
        checksum = 0
        rng = random.Random(seed)
        env = BarricadeEnv(invalid_action_mode="raise", max_steps=500)
        env.reset(seed=seed)
        for _ in range(count):
            legal_actions = env.legal_actions()
            if not legal_actions:
                env.reset(seed=rng.randrange(2**31))
                legal_actions = env.legal_actions()
            action = rng.choice(legal_actions)
            _, reward, terminated, truncated, info = env.step(action)
            checksum += int(action) + int(reward * 1000.0) + int(info["steps"])
            if terminated or truncated:
                env.reset(seed=rng.randrange(2**31))
        return checksum

    return {
        "pawn_moves": BenchmarkCase(
            "pawn_moves",
            pawn_moves,
            iterations,
            "BarricadeState.get_pawn_moves()",
        ),
        "wall_moves_uncached": BenchmarkCase(
            "wall_moves_uncached",
            wall_moves_uncached,
            iterations,
            "BarricadeState.get_valid_wall_moves() on copied states",
        ),
        "legal_moves_cached": BenchmarkCase(
            "legal_moves_cached",
            legal_moves_cached,
            iterations,
            "BarricadeState.legal_moves() after cache warmup",
        ),
        "legal_action_moves_cached": BenchmarkCase(
            "legal_action_moves_cached",
            legal_action_moves_cached,
            iterations,
            "BarricadeState.legal_action_moves() after cache warmup",
        ),
        "legal_actions_uncached": BenchmarkCase(
            "legal_actions_uncached",
            legal_actions_uncached,
            iterations,
            "BarricadeState.legal_actions() on copied states",
        ),
        "action_mask_uncached": BenchmarkCase(
            "action_mask_uncached",
            action_mask_uncached,
            iterations,
            "BarricadeState.action_mask() on copied states",
        ),
        "shortest_path_cached": BenchmarkCase(
            "shortest_path_cached",
            shortest_path_cached,
            iterations,
            "Two shortest_path_length() calls after cache warmup",
        ),
        "shortest_path_uncached": BenchmarkCase(
            "shortest_path_uncached",
            shortest_path_uncached,
            iterations,
            "Two shortest_path_length() calls with per-state path caches cleared",
        ),
        "greedy_route_uncached": BenchmarkCase(
            "greedy_route_uncached",
            greedy_route_uncached,
            iterations,
            "Two greedy_path_cells() calls with per-state route caches cleared",
        ),
        "wall_shape_available": BenchmarkCase(
            "wall_shape_available",
            wall_shape_available,
            iterations,
            "BarricadeState.is_wall_shape_available()",
        ),
        "valid_wall_placement": BenchmarkCase(
            "valid_wall_placement",
            valid_wall_placement,
            iterations,
            "BarricadeState.is_valid_wall_placement()",
        ),
        "apply_move": BenchmarkCase(
            "apply_move",
            apply_move,
            iterations,
            "BarricadeState.apply_move(); pure play-move cost",
        ),
        "apply_action_no_validate": BenchmarkCase(
            "apply_action_no_validate",
            apply_action_no_validate,
            iterations,
            "BarricadeState.apply_action(validate=False)",
        ),
        "apply_action_validate": BenchmarkCase(
            "apply_action_validate",
            apply_action_validate,
            iterations,
            "BarricadeState.apply_action(validate=True)",
        ),
        "env_observation": BenchmarkCase(
            "env_observation",
            env_observation,
            iterations,
            "BarricadeEnv._get_observation() with rotating states",
        ),
        "env_legal_action_mask": BenchmarkCase(
            "env_legal_action_mask",
            env_legal_action_mask,
            iterations,
            "BarricadeEnv.legal_action_mask() with rotating states",
        ),
        "env_step": BenchmarkCase(
            "env_step",
            env_step,
            step_iterations,
            "BarricadeEnv.step() with random legal actions",
        ),
    }


def run_case(case: BenchmarkCase, *, repeats: int) -> BenchmarkResult:
    if case.iterations <= 0:
        raise ValueError(f"{case.name} iterations must be positive.")
    durations: List[float] = []
    checksum = 0

    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        checksum = case.runner(case.iterations)
        durations.append(time.perf_counter() - started)

    durations.sort()
    median_seconds = durations[len(durations) // 2]
    return BenchmarkResult(
        name=case.name,
        iterations=case.iterations,
        best_seconds=durations[0],
        median_seconds=median_seconds,
        checksum=checksum,
        note=case.note,
    )


def print_results(results: Sequence[BenchmarkResult]) -> None:
    name_width = max([len("benchmark")] + [len(result.name) for result in results])
    print("core_env_benchmark")
    print(
        f"{'benchmark':<{name_width}}  {'iters':>8}  {'best_us':>10}  "
        f"{'median_us':>10}  {'best_ops/s':>12}  {'checksum':>10}  note"
    )
    print("-" * (name_width + 75))
    for result in results:
        print(
            f"{result.name:<{name_width}}  "
            f"{result.iterations:>8}  "
            f"{result.best_us_per_call:>10.2f}  "
            f"{result.median_us_per_call:>10.2f}  "
            f"{result.best_calls_per_second:>12.1f}  "
            f"{result.checksum:>10}  "
            f"{result.note}"
        )


def write_csv(path: str, results: Sequence[BenchmarkResult]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "benchmark",
                "iterations",
                "best_seconds",
                "median_seconds",
                "best_us_per_call",
                "median_us_per_call",
                "best_calls_per_second",
                "checksum",
                "note",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "benchmark": result.name,
                    "iterations": result.iterations,
                    "best_seconds": result.best_seconds,
                    "median_seconds": result.median_seconds,
                    "best_us_per_call": result.best_us_per_call,
                    "median_us_per_call": result.median_us_per_call,
                    "best_calls_per_second": result.best_calls_per_second,
                    "checksum": result.checksum,
                    "note": result.note,
                }
            )


def parse_only(value: Optional[str]) -> Optional[set[str]]:
    if value is None:
        return None
    names = {item.strip() for item in value.split(",") if item.strip()}
    return names or None


def selected_cases(
    cases: Dict[str, BenchmarkCase],
    requested: Optional[Iterable[str]],
) -> List[BenchmarkCase]:
    if requested is None:
        return list(cases.values())

    missing = [name for name in requested if name not in cases]
    if missing:
        available = ", ".join(cases)
        raise ValueError(
            f"Unknown benchmark name(s): {', '.join(missing)}. "
            f"Available benchmarks: {available}"
        )
    return [case for name, case in cases.items() if name in requested]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark core functions in barricade_env.py."
    )
    parser.add_argument("--positions", type=int, default=64)
    parser.add_argument("--warmup-plies", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--step-iterations", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated benchmark names to run.",
    )
    parser.add_argument("--csv-out", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    step_iterations = (
        int(args.step_iterations)
        if args.step_iterations is not None
        else int(args.iterations)
    )

    positions = make_positions(
        count=int(args.positions),
        warmup_plies=int(args.warmup_plies),
        seed=int(args.seed),
    )
    samples = sample_action_moves(positions, seed=int(args.seed) + 1)
    cases = build_cases(
        positions=positions,
        samples=samples,
        iterations=int(args.iterations),
        step_iterations=step_iterations,
        seed=int(args.seed),
    )
    requested = parse_only(args.only)
    cases_to_run = selected_cases(cases, requested)
    results = [run_case(case, repeats=int(args.repeats)) for case in cases_to_run]

    print(
        f"positions={len(positions)} warmup_plies={args.warmup_plies} "
        f"iterations={args.iterations} repeats={args.repeats} seed={args.seed}"
    )
    print_results(results)

    if args.csv_out:
        write_csv(args.csv_out, results)
        print(f"csv_out={args.csv_out}")


if __name__ == "__main__":
    main()
