"""
Benchmark two 7x7 Barricade checkpoints head-to-head.

Examples:
    python benchmark_models.py checkpoints/a.pt checkpoints/b.pt --games 32 --simulations 64
    python benchmark_models.py checkpoint_copies/best_1506.pt checkpoints/7x7_mcts_aux.pt --games 8 --simulations 0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import time
from typing import Dict, Optional

import torch
from torch import nn

from barricade_env import apply_selected_action
from mcts import MCTSConfig
from train_7x7_tabula_rasa import (
    AlphaZeroNet,
    BOARD_SIZE,
    DEFAULT_MAX_STEPS,
    DEFAULT_WALLS_PER_PLAYER,
    GameState,
    Player,
    adjudicated_winner,
    build_mcts,
    select_model_action,
)


@dataclass(frozen=True)
class LoadedModel:
    name: str
    path: Path
    model: AlphaZeroNet
    hidden_channels: int
    residual_blocks: int


@dataclass(frozen=True)
class GameResult:
    index: int
    a_player: Player
    starting_player: Player
    start_col: int
    winner: Optional[Player]
    truncated: bool
    steps: int

    @property
    def a_won(self) -> bool:
        return self.winner is not None and self.winner == self.a_player

    @property
    def b_won(self) -> bool:
        return self.winner is not None and self.winner != self.a_player


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play two 7x7 Barricade checkpoints against each other."
    )
    parser.add_argument("checkpoint_a", type=Path)
    parser.add_argument("checkpoint_b", type=Path)
    parser.add_argument("--name-a", type=str, default=None)
    parser.add_argument("--name-b", type=str, default=None)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--walls", type=int, default=DEFAULT_WALLS_PER_PLAYER)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=None,
        help="Fallback model width if it cannot be inferred from a checkpoint.",
    )
    parser.add_argument(
        "--residual-blocks",
        type=int,
        default=None,
        help="Fallback residual block count if it cannot be inferred from a checkpoint.",
    )
    parser.add_argument("--no-adjudicate-step-limit", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.games <= 0:
        raise ValueError("--games must be positive.")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    if args.walls < 0:
        raise ValueError("--walls must be non-negative.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_a = load_model(
        args.checkpoint_a,
        device=device,
        name=args.name_a,
        fallback_hidden_channels=args.hidden_channels,
        fallback_residual_blocks=args.residual_blocks,
    )
    model_b = load_model(
        args.checkpoint_b,
        device=device,
        name=args.name_b,
        fallback_hidden_channels=args.hidden_channels,
        fallback_residual_blocks=args.residual_blocks,
    )

    rng = random.Random(args.seed)
    started = time.perf_counter()
    results: list[GameResult] = []
    for game_index in range(args.games):
        result = play_game(
            game_index,
            model_a.model,
            model_b.model,
            rng=rng,
            device=device,
            simulations=max(0, int(args.simulations)),
            batch_size=max(1, int(args.batch_size)),
            walls=int(args.walls),
            max_steps=int(args.max_steps),
            adjudicate_step_limit=not args.no_adjudicate_step_limit,
        )
        results.append(result)
        if not args.quiet:
            print(format_game_result(result, model_a.name, model_b.name))

    elapsed = time.perf_counter() - started
    print_summary(
        results,
        model_a=model_a,
        model_b=model_b,
        device=device,
        simulations=max(0, int(args.simulations)),
        batch_size=max(1, int(args.batch_size)),
        max_steps=int(args.max_steps),
        walls=int(args.walls),
        elapsed=elapsed,
    )


def load_model(
    path: Path,
    *,
    device: torch.device,
    name: Optional[str],
    fallback_hidden_channels: Optional[int],
    fallback_residual_blocks: Optional[int],
) -> LoadedModel:
    payload = torch.load(path, map_location=device, weights_only=False)
    state_dict = _state_dict_from_payload(payload)
    hidden_channels, residual_blocks = _infer_model_config(
        payload,
        state_dict,
        fallback_hidden_channels=fallback_hidden_channels,
        fallback_residual_blocks=fallback_residual_blocks,
    )
    model = AlphaZeroNet(
        hidden_channels=hidden_channels,
        residual_blocks=residual_blocks,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return LoadedModel(
        name=name or path.stem,
        path=path,
        model=model,
        hidden_channels=hidden_channels,
        residual_blocks=residual_blocks,
    )


def _state_dict_from_payload(payload: object) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model_state" in payload:
        state_dict = payload["model_state"]
    else:
        state_dict = payload
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint must be a state_dict or contain 'model_state'.")
    return state_dict


def _infer_model_config(
    payload: object,
    state_dict: Dict[str, torch.Tensor],
    *,
    fallback_hidden_channels: Optional[int],
    fallback_residual_blocks: Optional[int],
) -> tuple[int, int]:
    args = payload.get("args", {}) if isinstance(payload, dict) else {}
    if not isinstance(args, dict):
        args = {}

    hidden_channels = args.get("hidden_channels", fallback_hidden_channels)
    if hidden_channels is None:
        stem_weight = state_dict.get("stem.0.weight")
        if stem_weight is not None:
            hidden_channels = int(stem_weight.shape[0])
    if hidden_channels is None:
        hidden_channels = 64

    residual_blocks = args.get("residual_blocks", fallback_residual_blocks)
    if residual_blocks is None:
        block_indices = []
        for key in state_dict:
            if not key.startswith("tower."):
                continue
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_indices.append(int(parts[1]))
        residual_blocks = max(block_indices) + 1 if block_indices else 4

    return int(hidden_channels), int(residual_blocks)


def play_game(
    game_index: int,
    model_a: AlphaZeroNet,
    model_b: AlphaZeroNet,
    *,
    rng: random.Random,
    device: torch.device,
    simulations: int,
    batch_size: int,
    walls: int,
    max_steps: int,
    adjudicate_step_limit: bool,
) -> GameResult:
    a_player = Player.RED if game_index % 2 == 0 else Player.BLUE
    starting_player = Player.RED if (game_index // 2) % 2 == 0 else Player.BLUE
    models = {
        a_player: model_a,
        a_player.opposite(): model_b,
    }
    start_col = rng.randrange(BOARD_SIZE)
    state = standard_state(
        walls=walls,
        starting_player=starting_player,
        start_col=start_col,
    )
    steps = 0
    truncated = False

    for ply in range(max_steps):
        if state.winner is not None:
            break

        legal_actions = state.legal_actions()
        if not legal_actions:
            state.winner = state.current_player.opposite()
            break

        action = choose_action(
            models[state.current_player],
            state,
            device=device,
            rng=rng,
            simulations=simulations,
            batch_size=batch_size,
        )
        state = apply_selected_action(state, action, legal_actions)
        steps = ply + 1
    else:
        truncated = state.winner is None

    if truncated:
        winner = adjudicated_winner(state, enabled=adjudicate_step_limit)
        if winner is not None:
            state.winner = winner
            truncated = False

    return GameResult(
        index=game_index,
        a_player=a_player,
        starting_player=starting_player,
        start_col=start_col,
        winner=state.winner,
        truncated=truncated,
        steps=steps,
    )


def standard_state(*, walls: int, starting_player: Player, start_col: int) -> GameState:
    return GameState(
        red_start=(0, int(start_col)),
        blue_start=(BOARD_SIZE - 1, int(start_col)),
        red_walls=walls,
        blue_walls=walls,
        starting_player=starting_player,
        board_size=BOARD_SIZE,
    )


def choose_action(
    model: AlphaZeroNet,
    state: GameState,
    *,
    device: torch.device,
    rng: random.Random,
    simulations: int,
    batch_size: int,
) -> int:
    if simulations <= 0:
        return select_model_action(model, state, device=device, rng=rng)

    result = build_mcts(
        model,
        device=device,
        rng=rng,
        config=MCTSConfig(
            num_simulations=simulations,
            batch_size=batch_size,
            cpuct_init=1.5,
            policy_target_temperature=1.0,
            action_temperature=0.0,
            add_root_noise=False,
            lead_weight=0.02,
            lead_scale=5.0,
        ),
    ).search(state)
    action = int(result.action)
    legal_actions = state.legal_actions()
    if action in legal_actions:
        return action
    return select_model_action(model, state, device=device, rng=rng)


def format_game_result(result: GameResult, name_a: str, name_b: str) -> str:
    red_name = name_a if result.a_player == Player.RED else name_b
    blue_name = name_b if result.a_player == Player.RED else name_a
    if result.winner is None:
        winner = "draw"
    else:
        winner_name = red_name if result.winner == Player.RED else blue_name
        winner = f"{winner_name}/{result.winner.name}"

    return (
        f"game={result.index + 1} red={red_name} blue={blue_name} "
        f"start={result.starting_player.name} col={result.start_col} winner={winner} "
        f"steps={result.steps} truncated={result.truncated}"
    )


def print_summary(
    results: list[GameResult],
    *,
    model_a: LoadedModel,
    model_b: LoadedModel,
    device: torch.device,
    simulations: int,
    batch_size: int,
    max_steps: int,
    walls: int,
    elapsed: float,
) -> None:
    total = len(results)
    a_wins = sum(1 for result in results if result.a_won)
    b_wins = sum(1 for result in results if result.b_won)
    draws = total - a_wins - b_wins
    a_red_games = sum(1 for result in results if result.a_player == Player.RED)
    a_blue_games = total - a_red_games
    a_red_wins = sum(
        1 for result in results if result.a_won and result.a_player == Player.RED
    )
    a_blue_wins = sum(
        1 for result in results if result.a_won and result.a_player == Player.BLUE
    )
    first_player_wins = sum(
        1 for result in results
        if result.winner is not None and result.winner == result.starting_player
    )
    avg_steps = sum(result.steps for result in results) / max(1, total)

    print("benchmark_models")
    print(
        f"device={device} games={total} simulations={simulations} "
        f"batch_size={batch_size} walls={walls} max_steps={max_steps}"
    )
    print(
        f"model_a={model_a.name} path={model_a.path} "
        f"hidden={model_a.hidden_channels} residual_blocks={model_a.residual_blocks}"
    )
    print(
        f"model_b={model_b.name} path={model_b.path} "
        f"hidden={model_b.hidden_channels} residual_blocks={model_b.residual_blocks}"
    )
    print(
        f"score {model_a.name}={a_wins} {model_b.name}={b_wins} draws={draws} "
        f"{model_a.name}_win_rate={a_wins / max(1, total):.3f}"
    )
    print(
        f"{model_a.name}_as_red={a_red_wins}/{a_red_games} "
        f"{model_a.name}_as_blue={a_blue_wins}/{a_blue_games} "
        f"first_player_wins={first_player_wins}/{total}"
    )
    print(
        f"avg_steps={avg_steps:.1f} elapsed={elapsed:.3f}s "
        f"games_per_second={total / max(elapsed, 1.0e-12):.2f}"
    )


if __name__ == "__main__":
    main()
