"""
Mini Benchmark Script for Barricade Game.
Loads up a model (either 7x7 or 9x9) and lets it play N games
with random starting columns against a simple Alpha-Beta search opponent.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from barricade_env import (
    BarricadeState,
    Player,
    WallOrientation,
    MoveDirection,
    apply_selected_action,
)
from mcts import MCTS, MCTSConfig


MODEL_HISTORY_LENGTH_ATTR = "_mini_bench_history_length"
MODEL_VALUE_MULTIPLIER_ATTR = "_mini_bench_value_multiplier"
MODEL_POLICY_HEAD_TYPE_ATTR = "_mini_bench_policy_head_type"


class ValueSignCorrectedModel(nn.Module):
    def __init__(self, model: nn.Module, value_multiplier: float):
        super().__init__()
        self.model = model
        self.value_multiplier = float(value_multiplier)

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        if not isinstance(outputs, tuple) or len(outputs) < 2:
            return outputs
        outputs_list = list(outputs)
        outputs_list[1] = outputs_list[1] * self.value_multiplier
        return tuple(outputs_list)

    def inference(self, *args, **kwargs):
        logits, value, lead = self.model.inference(*args, **kwargs)
        return logits, value * self.value_multiplier, lead


def model_for_mcts(model: nn.Module) -> nn.Module:
    value_multiplier = float(getattr(model, MODEL_VALUE_MULTIPLIER_ATTR, 1.0))
    if value_multiplier == 1.0:
        return model
    return ValueSignCorrectedModel(model, value_multiplier)


@torch.inference_mode()
def raw_value_head(
    model: nn.Module,
    state: BarricadeState,
    *,
    board_size: int,
    device: torch.device,
    history: Sequence[BarricadeState] = (),
) -> float:
    model.eval()
    if board_size == 7:
        from train_7x7_tabula_rasa import encode_state

        state_planes = encode_state(state, device).unsqueeze(0)
    else:
        from network import encode_state_stack

        history_length = int(getattr(model, MODEL_HISTORY_LENGTH_ATTR, 0))
        state_planes = encode_state_stack(
            state,
            history[-history_length:] if history_length > 0 else (),
            history_length=history_length,
        ).unsqueeze(0).to(device)

    _, value, _ = model.inference(state_planes)
    return float(
        torch.nan_to_num(
            value.reshape(-1)[0],
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        .clamp(-1.0, 1.0)
        .item()
    )


def probe_value_perspective(
    model: nn.Module,
    *,
    board_size: int,
    device: torch.device,
) -> float:
    walls = 5 if board_size == 7 else 10
    columns = sorted({0, board_size // 2, board_size - 1})
    probes: List[BarricadeState] = []
    for col in columns:
        other_col = (col + 1) % board_size
        probes.append(
            BarricadeState(
                red_start=(board_size - 2, col),
                blue_start=(board_size - 1, other_col),
                red_walls=walls,
                blue_walls=walls,
                starting_player=Player.RED,
                board_size=board_size,
            )
        )
        probes.append(
            BarricadeState(
                red_start=(0, other_col),
                blue_start=(1, col),
                red_walls=walls,
                blue_walls=walls,
                starting_player=Player.BLUE,
                board_size=board_size,
            )
        )

    mean_raw_value = sum(
        raw_value_head(
            model,
            state,
            board_size=board_size,
            device=device,
        )
        for state in probes
    ) / max(1, len(probes))
    return mean_raw_value


def infer_value_perspective(
    model: nn.Module,
    *,
    board_size: int,
    device: torch.device,
) -> Tuple[str, float]:
    """Auto-detect the raw value-head sign convention.

    ``probe_value_perspective`` evaluates positions where the side-to-move is
    clearly winning, so a side-to-move POV head returns a positive mean and an
    inverted ("opponent") head returns a negative mean. Returns the detected
    perspective ("side-to-move" or "opponent") and the probe score.

    Note: the training target in train.py is side-to-move POV by construction
    (+1 iff side_to_move == winner), so "side-to-move" is the expected result.
    The probe magnitude can be small for under-trained value heads, which is why
    the CLI defaults to "side-to-move" rather than this auto-detection.
    """
    score = probe_value_perspective(model, board_size=board_size, device=device)
    perspective = "side-to-move" if score >= 0.0 else "opponent"
    return perspective, score


class AlphaBetaAI:
    def __init__(self, player: Player, max_depth: int = 3):
        self.player = player
        self.max_depth = max_depth

    def get_best_move(self, state: BarricadeState) -> int:
        legal_action_moves = state.legal_action_moves()
        if not legal_action_moves:
            return -1

        best_score = -float("inf")
        best_action = legal_action_moves[0][0]
        alpha = -float("inf")
        beta = float("inf")

        for action, _ in legal_action_moves:
            next_state = state.apply_action(action, validate=False)
            score = self._minimax(next_state, self.max_depth - 1, alpha, beta, False)
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, score)

        return best_action

    def _minimax(
        self,
        state: BarricadeState,
        depth: int,
        alpha: float,
        beta: float,
        is_maximizing: bool,
    ) -> float:
        if state.winner is not None:
            if state.winner == self.player:
                return 1000.0 + depth
            else:
                return -1000.0 - depth

        if depth == 0:
            return self._evaluate(state)

        legal_action_moves = state.legal_action_moves()
        if not legal_action_moves:
            return -1000.0 - depth

        if is_maximizing:
            max_eval = -float("inf")
            for action, _ in legal_action_moves:
                next_state = state.apply_action(action, validate=False)
                eval_score = self._minimax(next_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for action, _ in legal_action_moves:
                next_state = state.apply_action(action, validate=False)
                eval_score = self._minimax(next_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def _evaluate(self, state: BarricadeState) -> float:
        dist_me = state.shortest_path_length(self.player)
        dist_opp = state.shortest_path_length(self.player.opposite())

        max_dist = state.board_size * state.board_size
        if dist_me is None:
            dist_me = max_dist
        if dist_opp is None:
            dist_opp = max_dist

        score = (dist_opp - dist_me) * 10.0
        score += (state.walls_left[self.player] - state.walls_left[self.player.opposite()]) * 2.0
        return score


@torch.inference_mode()
def select_model_action_9x9(
    model: nn.Module,
    state: BarricadeState,
    *,
    device: torch.device,
    rng: random.Random,
    history: Sequence[BarricadeState] = (),
) -> int:
    legal_actions = state.legal_actions()
    if not legal_actions:
        return -1
    model.eval()
    from network import encode_state_stack

    history_length = int(getattr(model, MODEL_HISTORY_LENGTH_ATTR, 0))
    state_planes = encode_state_stack(
        state,
        history[-history_length:] if history_length > 0 else (),
        history_length=history_length,
    ).unsqueeze(0).to(device)
    logits, _, _ = model.inference(state_planes)
    logits = torch.nan_to_num(
        logits.squeeze(0),
        nan=0.0,
        posinf=30.0,
        neginf=-30.0,
    ).clamp(-30.0, 30.0)

    from canonical import canonical_action

    canonical_legal = [canonical_action(action, state.current_player) for action in legal_actions]
    legal_tensor = torch.as_tensor(canonical_legal, dtype=torch.long, device=logits.device)
    legal_logits = logits.index_select(0, legal_tensor)
    return int(legal_actions[int(torch.argmax(legal_logits).item())])


def build_mcts_9x9(
    model: nn.Module,
    *,
    device: torch.device,
    rng: random.Random,
    config: MCTSConfig,
) -> MCTS:
    from canonical import canonical_action

    return MCTS(
        model,
        config=config,
        device=device,
        rng=rng,
        policy_action_transform=lambda action, state: canonical_action(
            action,
            state.current_player,
        ),
    )


def choose_model_action(
    model: nn.Module,
    state: BarricadeState,
    *,
    board_size: int,
    device: torch.device,
    rng: random.Random,
    simulations: int,
    batch_size: int,
    history: Sequence[BarricadeState] = (),
) -> int:
    if board_size == 7:
        from train_7x7_tabula_rasa import build_mcts, select_model_action
        if simulations <= 0:
            return select_model_action(model, state, device=device, rng=rng)
        
        config = MCTSConfig(
            num_simulations=simulations,
            batch_size=batch_size,
            cpuct_init=1.5,
            policy_target_temperature=1.0,
            action_temperature=0.0,
            add_root_noise=False,
            lead_weight=0.02,
            lead_scale=5.0,
        )
        result = build_mcts(
            model_for_mcts(model),
            device=device,
            rng=rng,
            config=config,
        ).search(state)
        action = int(result.action)
        if action in state.legal_actions():
            return action
        return select_model_action(model, state, device=device, rng=rng)
    else:
        if simulations <= 0:
            return select_model_action_9x9(
                model,
                state,
                device=device,
                rng=rng,
                history=history,
            )
        
        history_length = int(getattr(model, MODEL_HISTORY_LENGTH_ATTR, 0))
        config = MCTSConfig(
            num_simulations=simulations,
            batch_size=batch_size,
            cpuct_init=1.5,
            policy_target_temperature=1.0,
            action_temperature=0.0,
            history_length=history_length,
            add_root_noise=False,
            lead_weight=0.02,
            lead_scale=5.0,
        )
        result = build_mcts_9x9(
            model_for_mcts(model),
            device=device,
            rng=rng,
            config=config,
        ).search(
            state,
            history=history[-history_length:] if history_length > 0 else (),
        )
        action = int(result.action)
        if action in state.legal_actions():
            return action
        return select_model_action_9x9(
            model,
            state,
            device=device,
            rng=rng,
            history=history,
        )


@torch.inference_mode()
def evaluate_value_head_blue_pov(
    model: nn.Module,
    state: BarricadeState,
    *,
    board_size: int,
    device: torch.device,
    history: Sequence[BarricadeState] = (),
) -> float:
    """Return the value head from a fixed Blue perspective.

    ``MODEL_VALUE_MULTIPLIER_ATTR`` converts the checkpoint's raw scalar into
    side-to-move POV before this function converts it into Blue POV.
    """
    if getattr(state, "is_draw", False):
        return 0.0
    if state.winner is not None:
        return 1.0 if state.winner == Player.BLUE else -1.0

    side_to_move_value = raw_value_head(
        model,
        state,
        board_size=board_size,
        device=device,
        history=history,
    )
    value_multiplier = float(getattr(model, MODEL_VALUE_MULTIPLIER_ATTR, 1.0))
    side_to_move_value *= value_multiplier
    return side_to_move_value if state.current_player == Player.BLUE else -side_to_move_value


def format_move(move_type: str, details: Tuple) -> str:
    if move_type == "move":
        direction = details[0]
        name = direction.name if hasattr(direction, "name") else str(direction)
        return f"move {name}"
    elif move_type == "wall":
        orientation, row, col = details
        name = orientation.name if hasattr(orientation, "name") else str(orientation)
        return f"wall {name} r={row} c={col}"
    return f"{move_type} {details}"


def adjudicated_winner(state: BarricadeState) -> Optional[Player]:
    red_distance = state.shortest_path_length(Player.RED)
    blue_distance = state.shortest_path_length(Player.BLUE)
    if red_distance is None or blue_distance is None:
        return None
    if red_distance < blue_distance:
        return Player.RED
    if blue_distance < red_distance:
        return Player.BLUE
    return None


def play_single_game(
    game_idx: int,
    model: nn.Module,
    *,
    board_size: int,
    walls: int,
    simulations: int,
    batch_size: int,
    max_steps: int,
    ab_depth: int,
    device: torch.device,
    rng: random.Random,
    starting_player: Player = Player.RED,
    verbose: bool = True,
    adjudicate_step_limit: bool = False,
) -> Tuple[Optional[Player], int, bool]:
    # Alternate roles for model: game 0: model is RED, game 1: model is BLUE
    model_role = Player.RED if (game_idx // 2) % 2 == 0 else Player.BLUE
    ab_role = model_role.opposite()

    ab_agent = AlphaBetaAI(ab_role, max_depth=ab_depth)

    # Random starting column
    start_col = rng.randrange(board_size)
    start_col2 = rng.randrange(board_size)
    state = BarricadeState(
        red_start=(0, start_col),
        blue_start=(board_size - 1, start_col2),
        red_walls=walls,
        blue_walls=walls,
        starting_player=starting_player,
        board_size=board_size,
    )

    history: List[BarricadeState] = []

    if verbose:
        print(f"\n--- Game {game_idx + 1} ---")
        print(f"Model Role: {model_role.name} | AB Opponent Role: {ab_role.name}")
        print(f"Starting Player: {starting_player.name} | Random Start Col: {start_col}")
        print(f"Initial Walls: {walls} | Board Size: {board_size}x{board_size}")
        initial_value = evaluate_value_head_blue_pov(
            model,
            state,
            board_size=board_size,
            device=device,
            history=history,
        )
        print(
            f"Value Head (Blue POV): {initial_value:+.3f} "
            "(+1 BLUE winning / RED losing, -1 RED winning / BLUE losing)"
        )

    steps = 0
    truncated = False

    for ply in range(max_steps):
        if state.winner is not None or getattr(state, "is_draw", False):
            break

        legal_actions = state.legal_actions()
        if not legal_actions:
            if not getattr(state, "is_draw", False):
                state.winner = state.current_player.opposite()
            break

        current_player = state.current_player
        is_model_turn = current_player == model_role
        state_before = state.copy()
        value_blue_pov = (
            evaluate_value_head_blue_pov(
                model,
                state,
                board_size=board_size,
                device=device,
                history=history,
            )
            if verbose
            else 0.0
        )

        if is_model_turn:
            action = choose_model_action(
                model,
                state,
                board_size=board_size,
                device=device,
                rng=rng,
                simulations=simulations,
                batch_size=batch_size,
                history=history,
            )
        else:
            action = ab_agent.get_best_move(state)

        move_decoded = state.decode_action(action)
        state = apply_selected_action(state, action, legal_actions)
        history.append(state_before)
        steps = ply + 1

        if verbose:
            move_str = format_move(move_decoded[0], move_decoded[1:])
            pre_red_pos = state_before.pawns[Player.RED]
            pre_blue_pos = state_before.pawns[Player.BLUE]
            pre_walls = (
                state_before.walls_left[Player.RED],
                state_before.walls_left[Player.BLUE],
            )
            red_pos = state.pawns[Player.RED]
            blue_pos = state.pawns[Player.BLUE]
            post_walls = (
                state.walls_left[Player.RED],
                state.walls_left[Player.BLUE],
            )
            print(
                f"{steps:03d} {current_player.name:<4} valueB_pre={value_blue_pov:+.3f} "
                f"{move_str:<25} R={pre_red_pos}->{red_pos} "
                f"B={pre_blue_pos}->{blue_pos} walls={pre_walls}->{post_walls}"
            )
    else:
        truncated = state.winner is None

    if truncated and adjudicate_step_limit:
        winner = adjudicated_winner(state)
        if winner is not None:
            state.winner = winner
            truncated = False

    if verbose:
        winner_name = state.winner.name if state.winner is not None else "DRAW/TRUNCATED"
        actor = "Model" if state.winner == model_role else "AB Opponent"
        if state.winner is None:
            actor = "None"
        draw_reason = getattr(state, "draw_reason", None)
        suffix = f" [{draw_reason}]" if draw_reason else ""
        print(f"Winner: {winner_name} ({actor}) in {steps} steps{suffix}")

    return state.winner, steps, truncated


def _infer_policy_head_type(state_dict: Dict[str, torch.Tensor]) -> str:
    from network import infer_policy_head_type_from_state_dict

    return infer_policy_head_type_from_state_dict(state_dict)


def _infer_model_config(
    state_dict: Dict[str, torch.Tensor],
    fallback_hidden_channels: int,
    fallback_residual_blocks: int,
) -> Tuple[int, int, int, int, int, int, str]:
    hidden_channels = fallback_hidden_channels
    stem_weight_keys = ["stem.0.weight", "conv_tower.0.conv.weight"]
    input_planes = 9
    for key in stem_weight_keys:
        weight = state_dict.get(key)
        if weight is not None:
            hidden_channels = int(weight.shape[0])
            input_planes = int(weight.shape[1])
            break

    block_indices = []
    for key in state_dict:
        if not (key.startswith("tower.") or key.startswith("residual_tower.")):
            continue
        parts = key.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            block_indices.append(int(parts[1]))
    residual_blocks = max(block_indices) + 1 if block_indices else fallback_residual_blocks

    residual_channels = hidden_channels
    residual_weight = state_dict.get("residual_tower.0.conv1.weight")
    if residual_weight is not None:
        residual_channels = int(residual_weight.shape[0])

    value_hidden_size = hidden_channels
    value_weight = state_dict.get("value_head.fc1.weight")
    if value_weight is not None:
        value_hidden_size = int(value_weight.shape[0])

    num_conv_layers = 1
    conv_layer_indices = []
    for key in state_dict:
        if not key.startswith("conv_tower."):
            continue
        parts = key.split(".")
        if len(parts) > 2 and parts[1].isdigit():
            conv_layer_indices.append(int(parts[1]))
    if conv_layer_indices:
        num_conv_layers = max(conv_layer_indices) + 1
    policy_head_type = _infer_policy_head_type(state_dict)

    return (
        hidden_channels,
        residual_blocks,
        input_planes,
        residual_channels,
        value_hidden_size,
        num_conv_layers,
        policy_head_type,
    )


def _history_length_from_input_planes(input_planes: int) -> int:
    if input_planes % 9 != 0:
        raise RuntimeError(
            f"Cannot infer history length from {input_planes} input planes."
        )
    return input_planes // 9 - 1


def load_model(
    path: Path,
    *,
    board_size: int,
    device: torch.device,
) -> nn.Module:
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model_state" in payload:
        state_dict = payload["model_state"]
    else:
        state_dict = payload

    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint must be a state_dict or contain 'model_state'.")

    if board_size == 7:
        from train_7x7_tabula_rasa import AlphaZeroNet
        hidden_channels, residual_blocks, *_ = _infer_model_config(state_dict, 64, 4)
        model = AlphaZeroNet(
            hidden_channels=hidden_channels,
            residual_blocks=residual_blocks,
        ).to(device)
        setattr(model, MODEL_HISTORY_LENGTH_ATTR, 0)
    else:
        from network import build_network

        (
            hidden_channels,
            residual_blocks,
            input_planes,
            residual_channels,
            value_hidden_size,
            num_conv_layers,
            policy_head_type,
        ) = _infer_model_config(state_dict, 128, 10)
        history_length = _history_length_from_input_planes(input_planes)

        network_config = {}
        if isinstance(payload, dict) and isinstance(payload.get("network_config"), dict):
            network_config = dict(payload["network_config"])
            history_length = int(network_config.get("history_length", history_length))
            hidden_channels = int(network_config.get("conv_channels", hidden_channels))
            residual_channels = network_config.get("residual_channels", residual_channels)
            if residual_channels is None:
                residual_channels = hidden_channels
            else:
                residual_channels = int(residual_channels)
            num_conv_layers = int(network_config.get("num_conv_layers", num_conv_layers))
            residual_blocks = int(
                network_config.get("num_residual_layers", residual_blocks)
            )
            value_hidden_size = int(
                network_config.get("value_hidden_size", value_hidden_size)
            )
            policy_head_type = str(
                network_config.get("policy_head_type", policy_head_type)
            )

        model = build_network(
            history_length=history_length,
            conv_channels=hidden_channels,
            residual_channels=residual_channels,
            num_conv_layers=num_conv_layers,
            num_residual_layers=residual_blocks,
            value_hidden_size=value_hidden_size,
            policy_head_type=policy_head_type,
        ).to(device)
        setattr(model, MODEL_HISTORY_LENGTH_ATTR, history_length)
        setattr(model, MODEL_POLICY_HEAD_TYPE_ATTR, policy_head_type)

    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing = ("lead_head.", "future_map_head.", "score_head.")
    disallowed_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(allowed_missing)
    ]
    if incompatible.unexpected_keys or disallowed_missing:
        raise RuntimeError(
            "Checkpoint architecture mismatch while loading benchmark model: "
            f"missing={disallowed_missing[:8]} "
            f"unexpected={list(incompatible.unexpected_keys)[:8]}"
        )
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a trained Barricade model against a simple Alpha-Beta opponent."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/model_1606.pt"),
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--games",
        "-n",
        type=int,
        default=16,
        help="Number of games to play.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=128,
        help="Number of MCTS simulations per move for the model (0 to use policy network directly).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="MCTS batch size.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Alpha-Beta search depth.",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=7,
        help="Board size (7 or 9).",
    )
    parser.add_argument(
        "--walls",
        type=int,
        default=None,
        help="Walls per player. Defaults to 5 for 7x7, 10 for 9x9.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=96,
        help="Maximum step limit per game.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device (cpu/cuda).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="If set, suppresses printing move-by-move logs.",
    )
    parser.add_argument(
        "--adjudicate-step-limit",
        action="store_true",
        help="Award max-step games by shortest path instead of counting them as draws.",
    )
    parser.add_argument(
        "--starting-player",
        choices=("red", "blue"),
        default="red",
        help="Fixed starting player for every game.",
    )
    parser.add_argument(
        "--value-perspective",
        choices=("auto", "side-to-move", "opponent"),
        default="side-to-move",
        help=(
            "Raw value-head convention. train.py trains value as side-to-move "
            "POV (+1 == player-to-move winning), so this is the default. Use "
            "'opponent' if -1 means the current player is winning, or 'auto' to "
            "probe near-winning states (can be unreliable for weak value heads)."
        ),
    )
    args = parser.parse_args()

    # Seed configuration
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device configuration
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Board walls configuration
    if args.walls is None:
        walls = 5 if args.board_size == 7 else 10
    else:
        walls = args.walls

    print(f"Loading checkpoint from: {args.checkpoint} on device: {device}")
    model = load_model(args.checkpoint, board_size=args.board_size, device=device)
    value_perspective = args.value_perspective
    value_probe_score: Optional[float] = None
    if value_perspective == "auto":
        value_perspective, value_probe_score = infer_value_perspective(
            model,
            board_size=args.board_size,
            device=device,
        )
    value_multiplier = 1.0 if value_perspective == "side-to-move" else -1.0
    setattr(model, MODEL_VALUE_MULTIPLIER_ATTR, value_multiplier)

    print(f"\nStarting benchmark: {args.games} games, board size: {args.board_size}x{args.board_size}")
    print(f"Model: MCTS simulations = {args.simulations}")
    print(f"Alpha-Beta Opponent: search depth = {args.depth}")
    starting_player = Player[args.starting_player.upper()]
    print(f"Starting player: {starting_player.name}")
    probe_suffix = (
        f" (auto probe mean raw={value_probe_score:+.3f})"
        if value_probe_score is not None
        else ""
    )
    print(
        f"Value perspective: {value_perspective} "
        f"(MCTS multiplier {value_multiplier:+.0f}){probe_suffix}"
    )

    rng = random.Random(args.seed)
    model_wins = 0
    ab_wins = 0
    draws = 0
    total_steps = 0

    started_at = time.time()
    for game_idx in range(args.games):
        winner, steps, truncated = play_single_game(
            game_idx=game_idx,
            model=model,
            board_size=args.board_size,
            walls=walls,
            simulations=args.simulations,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            ab_depth=args.depth,
            device=device,
            rng=rng,
            starting_player=starting_player,
            verbose=not args.quiet,
            adjudicate_step_limit=args.adjudicate_step_limit,
        )
        total_steps += steps

        # Track results based on role assignment
        # Role formula in play_single_game:
        # model_role = Player.RED if (game_idx // 2) % 2 == 0 else Player.BLUE
        model_role = Player.RED if (game_idx // 2) % 2 == 0 else Player.BLUE
        if winner == model_role:
            model_wins += 1
        elif winner == model_role.opposite():
            ab_wins += 1
        else:
            draws += 1

    elapsed = time.time() - started_at
    avg_steps = total_steps / max(1, args.games)
    win_rate = model_wins / max(1, args.games)

    print("\n" + "=" * 40)
    print(" BENCHMARK RESULTS SUMMARY")
    print("=" * 40)
    print(f"Games played: {args.games}")
    print(f"Model wins:   {model_wins} ({model_wins / args.games * 100:.1f}%)")
    print(f"AB wins:      {ab_wins} ({ab_wins / args.games * 100:.1f}%)")
    print(f"Draws:        {draws} ({draws / args.games * 100:.1f}%)")
    print(f"Model Win Rate: {win_rate:.3f}")
    print(f"Average steps per game: {avg_steps:.1f}")
    print(f"Total time elapsed: {elapsed:.2f}s ({elapsed / args.games:.2f}s per game)")
    print("=" * 40)


if __name__ == "__main__":
    main()
