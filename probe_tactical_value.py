"""Probe value-head behavior on empty-board near-terminal race positions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import torch

from barricade_env import BarricadeState, Player
from mini_bench import (
    MODEL_HISTORY_LENGTH_ATTR,
    evaluate_value_head_blue_pov,
    load_model,
    raw_value_head,
)


Scenario = Tuple[str, Tuple[int, int], Tuple[int, int], Player, float]


def scenarios() -> Iterable[Scenario]:
    return (
        (
            "RED wins next move",
            (7, 1),
            (8, 0),
            Player.RED,
            1.0,
        ),
        (
            "RED two moves from win",
            (6, 1),
            (8, 0),
            Player.RED,
            1.0,
        ),
        (
            "BLUE wins next move; RED to move",
            (0, 1),
            (1, 0),
            Player.RED,
            -1.0,
        ),
        (
            "BLUE two moves from win; RED to move",
            (0, 1),
            (2, 0),
            Player.RED,
            -1.0,
        ),
        (
            "BLUE wins next move",
            (0, 1),
            (1, 0),
            Player.BLUE,
            1.0,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect value-head calibration on tactical race states."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/latest.pt"),
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_model(args.checkpoint, board_size=9, device=device)
    history_length = int(getattr(model, MODEL_HISTORY_LENGTH_ATTR, 0))
    print(
        f"checkpoint={args.checkpoint} device={device} "
        f"history_length={history_length} "
        f"policy_head={getattr(model, 'policy_head_type', '?')}"
    )
    print(
        f"{'scenario':<36} {'target':>7} {'raw_stm':>8} "
        f"{'abs_err':>8} {'blue_pov':>9} {'red_d':>5} {'blue_d':>6}"
    )
    abs_errors = []
    for label, red_start, blue_start, current, target in scenarios():
        state = BarricadeState(
            red_start=red_start,
            blue_start=blue_start,
            red_walls=10,
            blue_walls=10,
            starting_player=current,
            board_size=9,
        )
        raw = raw_value_head(model, state, board_size=9, device=device)
        blue = evaluate_value_head_blue_pov(
            model, state, board_size=9, device=device
        )
        abs_error = abs(raw - target)
        abs_errors.append(abs_error)
        print(
            f"{label:<36} {target:>+7.2f} {raw:>+8.3f} "
            f"{abs_error:>8.3f} {blue:>+9.3f} {state.shortest_path_length(Player.RED):>5} "
            f"{state.shortest_path_length(Player.BLUE):>6}"
        )
    print(f"mean_abs_error={sum(abs_errors) / max(1, len(abs_errors)):.3f}")


if __name__ == "__main__":
    main()
