"""
Benchmark two 9x9 Barricade checkpoints from train.py head-to-head.

Examples:
    python benchmark_models.py checkpoints/a.pt checkpoints/b.pt --games 32 --simulations 64
    python benchmark_models.py checkpoints/latest.pt checkpoints/model_iter_000010.pt --games 8 --simulations 0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import time
from typing import Any, Dict, Optional, Sequence

import torch
from torch import Tensor, nn

from barricade_env import (
    BOARD_SIZE,
    DEFAULT_WALLS_PER_PLAYER,
    BarricadeEnv,
    BarricadeState,
    Player,
)
from canonical import canonical_action
from mcts import MCTS, MCTSConfig
from network import encode_state_stack, infer_policy_head_type_from_state_dict
from train import NetworkConfig, _load_model_state, build_model, policy_action_for_mcts


DEFAULT_MAX_STEPS = 500


@dataclass(frozen=True)
class LoadedModel:
    name: str
    path: Path
    model: nn.Module
    network_config: NetworkConfig


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
        description="Play two 9x9 train.py Barricade checkpoints against each other."
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
        "--history-length",
        type=int,
        default=None,
        help="Override checkpoint history length when loading plain state_dict checkpoints.",
    )
    parser.add_argument(
        "--conv-channels",
        type=int,
        default=None,
        help="Override checkpoint conv channel count when loading plain state_dict checkpoints.",
    )
    parser.add_argument(
        "--residual-channels",
        type=int,
        default=None,
        help="Override checkpoint residual channel count when loading plain state_dict checkpoints.",
    )
    parser.add_argument(
        "--num-conv-layers",
        type=int,
        default=None,
        help="Override checkpoint conv layer count when loading plain state_dict checkpoints.",
    )
    parser.add_argument(
        "--num-residual-layers",
        type=int,
        default=None,
        help="Override checkpoint residual layer count when loading plain state_dict checkpoints.",
    )
    parser.add_argument(
        "--value-hidden-size",
        type=int,
        default=None,
        help="Override checkpoint value hidden size when loading plain state_dict checkpoints.",
    )
    parser.add_argument(
        "--policy-head-type",
        choices=("factored", "flat"),
        default=None,
        help="Override checkpoint policy head type when loading plain state_dict checkpoints.",
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
    model_a = load_model(args.checkpoint_a, device=device, name=args.name_a, args=args)
    model_b = load_model(args.checkpoint_b, device=device, name=args.name_b, args=args)

    rng = random.Random(args.seed)
    started = time.perf_counter()
    results: list[GameResult] = []
    for game_index in range(args.games):
        result = play_game(
            game_index,
            model_a,
            model_b,
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
    args: argparse.Namespace,
) -> LoadedModel:
    payload = torch.load(path, map_location=device, weights_only=False)
    state_dict = state_dict_from_payload(payload)
    network_config = network_config_from_payload(payload, args)
    model = build_model(network_config).to(device)
    _load_model_state(model, state_dict)
    model.eval()
    return LoadedModel(
        name=name or path.stem,
        path=path,
        model=model,
        network_config=network_config,
    )


def state_dict_from_payload(payload: object) -> Dict[str, Tensor]:
    if isinstance(payload, dict) and "model_state" in payload:
        state_dict = payload["model_state"]
    else:
        state_dict = payload
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint must be a state_dict or contain 'model_state'.")
    return state_dict


def network_config_from_payload(
    payload: object,
    args: argparse.Namespace,
) -> NetworkConfig:
    raw_config: Dict[str, Any] = {}
    if isinstance(payload, dict) and isinstance(payload.get("network_config"), dict):
        raw_config = dict(payload["network_config"])
    state_dict = state_dict_from_payload(payload)
    if "policy_head_type" not in raw_config:
        raw_config["policy_head_type"] = infer_policy_head_type_from_state_dict(
            state_dict
        )

    defaults = NetworkConfig()
    values = {
        "history_length": raw_config.get("history_length", defaults.history_length),
        "conv_channels": raw_config.get("conv_channels", defaults.conv_channels),
        "residual_channels": raw_config.get("residual_channels", defaults.residual_channels),
        "num_conv_layers": raw_config.get("num_conv_layers", defaults.num_conv_layers),
        "num_residual_layers": raw_config.get(
            "num_residual_layers", defaults.num_residual_layers
        ),
        "value_hidden_size": raw_config.get("value_hidden_size", defaults.value_hidden_size),
        "policy_head_type": raw_config.get(
            "policy_head_type", defaults.policy_head_type
        ),
    }

    for key in values:
        override = getattr(args, key)
        if override is not None:
            values[key] = override

    return NetworkConfig(**values)


def play_game(
    game_index: int,
    model_a: LoadedModel,
    model_b: LoadedModel,
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
    players = {
        a_player: model_a,
        a_player.opposite(): model_b,
    }
    start_col = rng.randrange(BOARD_SIZE)
    env = BarricadeEnv(max_steps=max_steps, invalid_action_mode="raise")
    _, info = env.reset(
        options=standard_options(
            walls=walls,
            starting_player=starting_player,
            start_col=start_col,
        )
    )
    history: list[BarricadeState] = []
    roots = {Player.RED: None, Player.BLUE: None}
    terminated = False
    truncated = False

    while not terminated and not truncated:
        current = env.state.current_player
        player_model = players[current]
        state_before = env.state.copy()
        history_before = history_window(
            history,
            player_model.network_config.history_length,
        )
        action, root = choose_action(
            player_model.model,
            state_before,
            history=history_before,
            root=roots[current],
            device=device,
            rng=rng,
            simulations=simulations,
            batch_size=batch_size,
            history_length=player_model.network_config.history_length,
        )
        history.append(state_before)
        _, _, terminated, truncated, info = env.step(action)
        roots[current] = root
        roots[current.opposite()] = None

    winner = player_from_name(info.get("winner"))
    if truncated:
        adjudicated = adjudicated_winner(env.state, enabled=adjudicate_step_limit)
        if adjudicated is not None:
            winner = adjudicated
            truncated = False

    return GameResult(
        index=game_index,
        a_player=a_player,
        starting_player=starting_player,
        start_col=start_col,
        winner=winner,
        truncated=truncated,
        steps=int(info.get("steps") or max_steps),
    )


def standard_options(*, walls: int, starting_player: Player, start_col: int) -> Dict[str, Any]:
    return {
        "red_start": (0, int(start_col)),
        "blue_start": (BOARD_SIZE - 1, int(start_col)),
        "red_walls": walls,
        "blue_walls": walls,
        "starting_player": starting_player,
    }


def choose_action(
    model: nn.Module,
    state: BarricadeState,
    *,
    history: Sequence[BarricadeState],
    root: object,
    device: torch.device,
    rng: random.Random,
    simulations: int,
    batch_size: int,
    history_length: int,
) -> tuple[int, object]:
    if simulations <= 0:
        return select_model_action(
            model,
            state,
            history=history,
            device=device,
            history_length=history_length,
        ), None

    mcts = MCTS(
        model,
        MCTSConfig(
            num_simulations=simulations,
            batch_size=batch_size,
            cpuct_init=1.5,
            policy_target_temperature=1.0,
            action_temperature=0.0,
            history_length=history_length,
            device=str(device),
            add_root_noise=False,
            lead_weight=0.02,
            lead_scale=5.0,
        ),
        device=device,
        rng=rng,
        policy_action_transform=policy_action_for_mcts,
    )
    result = mcts.search(state, history=history, root=root)
    action = int(result.action)
    legal_actions = state.legal_actions()
    if action in legal_actions:
        return action, mcts.advance_root(result.root, action)
    return (
        select_model_action(
            model,
            state,
            history=history,
            device=device,
            history_length=history_length,
        ),
        None,
    )


@torch.inference_mode()
def select_model_action(
    model: nn.Module,
    state: BarricadeState,
    *,
    history: Sequence[BarricadeState],
    device: torch.device,
    history_length: int,
) -> int:
    legal_actions = state.legal_actions()
    if not legal_actions:
        return -1

    model.eval()
    encoded = encode_state_stack(
        state,
        history,
        history_length=history_length,
    ).unsqueeze(0).to(device)
    logits, _, _ = model.inference(encoded)
    logits = torch.nan_to_num(logits.squeeze(0), nan=0.0, posinf=30.0, neginf=-30.0)
    canonical_legal = [
        canonical_action(action, state.current_player)
        for action in legal_actions
    ]
    legal_tensor = torch.as_tensor(canonical_legal, dtype=torch.long, device=device)
    legal_logits = logits.index_select(0, legal_tensor).clamp(-30.0, 30.0)
    return int(legal_actions[int(torch.argmax(legal_logits).item())])


def history_window(
    history: Sequence[BarricadeState],
    history_length: int,
) -> tuple[BarricadeState, ...]:
    if history_length <= 0:
        return ()
    return tuple(history[-history_length:])


def player_from_name(name: object) -> Optional[Player]:
    if not isinstance(name, str):
        return None
    try:
        return Player[name]
    except KeyError:
        return None


def adjudicated_winner(state: BarricadeState, *, enabled: bool) -> Optional[Player]:
    if not enabled:
        return None

    red_distance = state.shortest_path_length(Player.RED)
    blue_distance = state.shortest_path_length(Player.BLUE)
    if red_distance is not None and blue_distance is not None:
        if red_distance < blue_distance:
            return Player.RED
        if blue_distance < red_distance:
            return Player.BLUE

    red_walls = state.walls_left[Player.RED]
    blue_walls = state.walls_left[Player.BLUE]
    if red_walls > blue_walls:
        return Player.RED
    if blue_walls > red_walls:
        return Player.BLUE

    red_row = state.pawns[Player.RED][0]
    blue_progress = BOARD_SIZE - 1 - state.pawns[Player.BLUE][0]
    if red_row > blue_progress:
        return Player.RED
    if blue_progress > red_row:
        return Player.BLUE

    return None


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
        f"network_config={model_a.network_config}"
    )
    print(
        f"model_b={model_b.name} path={model_b.path} "
        f"network_config={model_b.network_config}"
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
