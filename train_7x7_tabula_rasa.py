"""
Tabula-rasa 7x7 Barricade AlphaZero-style training test.

This standalone file trains from random weights using self-play MCTS only:
there is no teacher, minimax player, scripted expert, or heuristic policy
target. The policy target is the MCTS visit distribution. Auxiliary targets are
generated from the played games and current rules:

    value          terminal winner from the side-to-move perspective
    lead           exact current race lead from the side-to-move perspective
    score          remaining plies to win, masked to eventual winner samples
    future_traverse future pawn cells visited by side-to-move and opponent

The script also uses randomized playout caps and sampled handicaps, evaluates
the trained MCTS player against a uniformly random legal-action policy, and
prints sample games.

Example:
    python train_7x7_tabula_rasa.py

Fast smoke test:
    python train_7x7_tabula_rasa.py --episodes 4 --base-simulations 8 --eval-games 4 --show-games 1 --no-save
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from barricade_env import (
    BarricadeState as GameState,
    Move,
    MoveDirection,
    Player,
    WallOrientation,
    action_size_for_board_size,
    apply_selected_action,
    decode_action_for_board_size,
    encode_move_for_board_size,
    path_score_for_player,
    state_lead_for_player,
)
from mcts import MCTS, MCTSConfig


BOARD_SIZE = 7
WALL_BOARD_SIZE = BOARD_SIZE - 1
DEFAULT_WALLS_PER_PLAYER = 5
DEFAULT_MAX_STEPS = 96

MOVE_ACTIONS = 4
WALL_ACTIONS_PER_ORIENTATION = WALL_BOARD_SIZE * WALL_BOARD_SIZE
HORIZONTAL_WALL_OFFSET = MOVE_ACTIONS
VERTICAL_WALL_OFFSET = HORIZONTAL_WALL_OFFSET + WALL_ACTIONS_PER_ORIENTATION
ACTION_SIZE = action_size_for_board_size(BOARD_SIZE)

INPUT_PLANES = 9


def encode_move(move: Move) -> int:
    return encode_move_for_board_size(move, BOARD_SIZE)


def decode_action(action: int) -> Move:
    return decode_action_for_board_size(action, BOARD_SIZE)


def _build_canonical_flip_permutation() -> Tuple[int, ...]:
    """Action permutation applied when the side-to-move is BLUE.

    Both players are presented to the network in a canonical frame where the
    side-to-move always advances toward increasing row index. For BLUE this is
    the board flipped about the horizontal axis, so actions must be remapped:
    UP<->DOWN, LEFT/RIGHT unchanged, and wall rows reflected (r -> WB-1-r).
    The permutation is its own inverse (an involution).
    """
    perm = [0] * ACTION_SIZE
    perm[MoveDirection.UP.value] = MoveDirection.DOWN.value
    perm[MoveDirection.DOWN.value] = MoveDirection.UP.value
    perm[MoveDirection.LEFT.value] = MoveDirection.LEFT.value
    perm[MoveDirection.RIGHT.value] = MoveDirection.RIGHT.value
    for offset in (HORIZONTAL_WALL_OFFSET, VERTICAL_WALL_OFFSET):
        for row in range(WALL_BOARD_SIZE):
            for col in range(WALL_BOARD_SIZE):
                perm[offset + row * WALL_BOARD_SIZE + col] = (
                    offset + (WALL_BOARD_SIZE - 1 - row) * WALL_BOARD_SIZE + col
                )
    return tuple(perm)


CANONICAL_FLIP_PERMUTATION: Tuple[int, ...] = _build_canonical_flip_permutation()


def canonical_action(action: int, player: "Player") -> int:
    """Map a raw board action into the canonical (side-to-move) frame."""
    if player == Player.RED:
        return int(action)
    return CANONICAL_FLIP_PERMUTATION[int(action)]


def canonicalize_action_vector(vector: Tensor, player: "Player") -> Tensor:
    """Reindex a per-action vector (policy target / mask) into canonical frame."""
    if player == Player.RED:
        return vector
    index = torch.as_tensor(CANONICAL_FLIP_PERMUTATION, dtype=torch.long, device=vector.device)
    canonical = torch.empty_like(vector)
    canonical[index] = vector
    return canonical


def adjudicated_winner(state: GameState, *, enabled: bool) -> Optional[Player]:
    if state.winner is not None:
        return state.winner
    if not enabled:
        return None

    red_distance = state.shortest_path_length(Player.RED)
    blue_distance = state.shortest_path_length(Player.BLUE)
    if red_distance is None or blue_distance is None:
        return None
    if red_distance < blue_distance:
        return Player.RED
    if blue_distance < red_distance:
        return Player.BLUE
    return None


def encode_state(state: GameState, device: Optional[torch.device] = None) -> Tensor:
    """Encode the board in a canonical frame from the side-to-move perspective.

    The board is presented as if the side-to-move always advances toward the
    last row. For BLUE this flips the board about the horizontal axis (rows
    r -> BOARD_SIZE-1-r on the cell grid, r -> WALL_BOARD_SIZE-1-r on the wall
    grid). This makes mirrored RED/BLUE positions encode identically, so a
    single shared policy/value head no longer has to learn two opposite notions
    of "forward". Plane 0 is a constant so it carries no absolute-color signal.
    """
    planes = torch.zeros((INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
    current = state.current_player
    opponent = current.opposite()
    flip = current == Player.BLUE

    def cell_row(row: int) -> int:
        return BOARD_SIZE - 1 - row if flip else row

    def wall_row(row: int) -> int:
        return WALL_BOARD_SIZE - 1 - row if flip else row

    planes[0].fill_(1.0)

    own_row, own_col = state.pawns[current]
    opp_row, opp_col = state.pawns[opponent]
    planes[1, cell_row(own_row), own_col] = 1.0
    planes[2, cell_row(opp_row), opp_col] = 1.0

    for orientation, row, col in state.walls:
        plane = 3 if orientation == WallOrientation.HORIZONTAL else 4
        planes[plane, wall_row(row), col] = 1.0

    own_initial = max(1, state.initial_walls[current])
    opp_initial = max(1, state.initial_walls[opponent])
    planes[5].fill_(state.walls_left[current] / own_initial)
    planes[6].fill_(state.walls_left[opponent] / opp_initial)

    # In the canonical frame the side-to-move always targets the last row.
    planes[7, BOARD_SIZE - 1, :] = 1.0
    planes[8, 0, :] = 1.0

    if device is not None:
        return planes.to(device)
    return planes


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8 if channels >= 8 else 1, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8 if channels >= 8 else 1, channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.relu(x + residual)


class AlphaZeroNet(nn.Module):
    def __init__(self, hidden_channels: int = 64, residual_blocks: int = 4) -> None:
        super().__init__()
        groups = 8 if hidden_channels >= 8 else 1
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(
            *[ResidualBlock(hidden_channels) for _ in range(residual_blocks)]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, ACTION_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(BOARD_SIZE * BOARD_SIZE, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Tanh(),
        )
        self.lead_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(BOARD_SIZE * BOARD_SIZE, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )
        self.future_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
        )
        self.score_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(BOARD_SIZE * BOARD_SIZE, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Softplus(),
        )

    def features(self, x: Tensor) -> Tensor:
        return self.tower(self.stem(x))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        features = self.features(x)
        return (
            self.policy_head(features),
            self.value_head(features).squeeze(-1),
            self.lead_head(features).squeeze(-1),
            self.future_head(features),
            self.score_head(features).squeeze(-1),
        )

    def inference(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        features = self.features(x)
        return (
            self.policy_head(features),
            self.value_head(features).squeeze(-1),
            self.lead_head(features).squeeze(-1),
        )


@dataclass
class ReplaySample:
    state: Tensor
    mask: Tensor
    policy_target: Tensor
    side_to_move: Player
    ply: int
    playout_cap: int
    value_target: float = 0.0
    lead_target: float = 0.0
    future_target: Optional[Tensor] = None
    score_target: float = 0.0
    score_mask: float = 0.0


@dataclass
class MoveRecord:
    ply: int
    player: Player
    action: int
    description: str
    red_position: Tuple[int, int]
    blue_position: Tuple[int, int]
    red_walls: int
    blue_walls: int


@dataclass
class GameRecord:
    winner: Optional[Player]
    truncated: bool
    steps: int
    moves: List[MoveRecord]
    final_state: GameState


def masked_logits(logits: Tensor, mask: Tensor) -> Tensor:
    logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
    logits = logits.clamp(-30.0, 30.0)
    mask = mask.to(device=logits.device, dtype=torch.bool)
    return logits.masked_fill(~mask, -1.0e9)


def soft_target_cross_entropy(logits: Tensor, target: Tensor) -> Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(target * log_probs).sum(dim=1).mean()


def masked_smooth_l1_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    per_sample = F.smooth_l1_loss(prediction, target, reduction="none")
    mask = mask.to(device=per_sample.device, dtype=per_sample.dtype)
    denominator = mask.sum()
    if float(denominator.item()) <= 0.0:
        return prediction.sum() * 0.0
    return (per_sample * mask).sum() / denominator


def sample_playout_cap(base_simulations: int, rng: random.Random) -> int:
    roll = rng.random()
    if roll < 0.60:
        cap = base_simulations          # 60% — full
    elif roll < 0.80:
        cap = max(1, base_simulations // 2)   # 20% — half
    elif roll < 0.93:
        cap = max(1, base_simulations // 4)   # 13% — quarter
    else:
        cap = base_simulations * 2      # 7%  — double (*4 dropped)
    return max(1, int(cap))


def sample_handicap(base_walls: int, rng: random.Random) -> Dict[str, object]:
    center = BOARD_SIZE // 2
    starting_player = (
        Player.RED if rng.random() < 0.5 else Player.BLUE
    )

    # Default (standard)
    red_row = 0
    blue_row = BOARD_SIZE - 1
    red_col = center
    blue_col = center
    red_walls = base_walls
    blue_walls = base_walls

    active_modes = []

    # Majority of games use shifted columns
    if rng.random() < 0.70:
        red_col = rng.randint(1, BOARD_SIZE - 2)
        blue_col = rng.randint(1, BOARD_SIZE - 2)
        active_modes.append("column_shift")

    # Sometimes start one row inward
    if rng.random() < 0.20:
        red_row = rng.randint(0, 1)
        blue_row = rng.randint(
            BOARD_SIZE - 2,
            BOARD_SIZE - 1
        )
        active_modes.append("row_ahead")

    # Occasionally modify wall counts
    if rng.random() < 0.10:
        red_walls = max(
            0,
            base_walls + rng.randint(-2, 2)
        )
        blue_walls = max(
            0,
            base_walls + rng.randint(-2, 2)
        )
        active_modes.append("wall_handicap")

    # Nothing happened
    if not active_modes:
        active_modes.append("standard")

    return {
        "mode": "+".join(active_modes),
        "modes": active_modes,
        "red_start": (red_row, red_col),
        "blue_start": (blue_row, blue_col),
        "red_walls": red_walls,
        "blue_walls": blue_walls,
        "starting_player": starting_player,
    }

def make_state_from_handicap(handicap: Dict[str, object]) -> GameState:
    return GameState(
        red_start=handicap["red_start"],
        blue_start=handicap["blue_start"],
        red_walls=int(handicap["red_walls"]),
        blue_walls=int(handicap["blue_walls"]),
        starting_player=handicap["starting_player"],
        board_size=BOARD_SIZE,
    )


def future_traverse_target(
    side_to_move: Player,
    ply: int,
    pawn_visits: Sequence[Tuple[int, Player, Tuple[int, int]]],
) -> Tensor:
    """Future pawn-visit map in the canonical (side-to-move) frame.

    Rows are flipped for BLUE so the spatial target lines up with the
    canonical board fed to the network in encode_state.
    """
    flip = side_to_move == Player.BLUE
    target = torch.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
    for visit_ply, player, position in pawn_visits:
        if visit_ply < ply:
            continue
        row, col = position
        if flip:
            row = BOARD_SIZE - 1 - row
        channel = 0 if player == side_to_move else 1
        target[channel, int(row), int(col)] = 1.0
    return target


def value_target(
    side_to_move: Player,
    winner: Optional[Player],
    *,
    plies_to_end: int = 0,
    discount: float = 1.0,
) -> float:
    """Outcome from the side-to-move perspective, optionally discounted.

    With discount < 1.0 the reward decays by ``discount ** plies_to_end`` so a
    win reached sooner is worth more than the same win reached later. This gives
    the value head (and therefore MCTS) a gradient toward finishing the game
    instead of shuffling pawns until adjudication.
    """
    if winner is None:
        return 0.0
    magnitude = discount ** max(0, int(plies_to_end)) if discount < 1.0 else 1.0
    return magnitude if side_to_move == winner else -magnitude


def score_target(state: GameState, side_to_move: Player) -> Tuple[float, float]:
    return path_score_for_player(state, side_to_move), 1.0


def action_description(action: int) -> str:
    move = decode_action(action)
    if move[0] == "move":
        return f"move {MoveDirection(move[1]).name}"
    _, orientation, row, col = move
    return f"wall {WallOrientation(orientation).name} r={row} c={col}"


def encode_state_for_mcts(
    state: GameState,
    history: Sequence[GameState],
    history_length: int,
) -> Tensor:
    return encode_state(state)


def policy_action_for_mcts(action: int, state: GameState) -> int:
    return canonical_action(action, state.current_player)


def lead_for_mcts(lead: float, state: GameState) -> float:
    return float(lead)


def build_mcts(
    model: AlphaZeroNet,
    *,
    device: torch.device,
    rng: random.Random,
    config: MCTSConfig,
) -> MCTS:
    return MCTS(
        model,
        config=config,
        device=device,
        rng=rng,
        state_encoder=encode_state_for_mcts,
        action_size=ACTION_SIZE,
        encode_move_fn=encode_move,
        policy_action_transform=policy_action_for_mcts,
        lead_transform=lead_for_mcts,
    )


def generate_self_play_episode(
    model: AlphaZeroNet,
    *,
    device: torch.device,
    rng: random.Random,
    base_walls: int,
    max_steps: int,
    base_simulations: int,
    temperature_drop_ply: int,
    mcts_policy_temperature: float,
    adjudicate_step_limit: bool,
    value_discount: float = 1.0,
    no_progress_limit: int = 50,
) -> Tuple[List[ReplaySample], Optional[Player], bool, int, Dict[str, object]]:
    handicap = sample_handicap(base_walls, rng)
    state = make_state_from_handicap(handicap)
    samples: List[ReplaySample] = []
    pawn_visits: List[Tuple[int, Player, Tuple[int, int]]] = []
    truncated = False

    best_distance = {
        Player.RED: state.shortest_path_length(Player.RED),
        Player.BLUE: state.shortest_path_length(Player.BLUE),
    }
    stalled_plies = 0

    for ply in range(max_steps):
        if state.winner is not None:
            break

        legal_actions = state.legal_actions()
        if not legal_actions:
            state.winner = state.current_player.opposite()
            break

        side_to_move = state.current_player
        simulations = sample_playout_cap(base_simulations, rng)
        mcts = build_mcts(
            model,
            device=device,
            rng=rng,
            config=MCTSConfig(
                num_simulations=simulations,
                cpuct_init=1.5,
                policy_temperature=mcts_policy_temperature,
                policy_target_temperature=1.0,
                action_temperature=1.0 if ply < temperature_drop_ply else 0.0,
                add_root_noise=True,
                lead_weight=0.02,
                lead_scale=5.0,
            ),
        )
        result = mcts.search(state)
        action = int(result.action)
        if action not in legal_actions:
            action = rng.choice(legal_actions)

        sample_score, sample_score_mask = score_target(state, side_to_move)
        samples.append(
            ReplaySample(
                state=encode_state(state).cpu(),
                mask=canonicalize_action_vector(
                    torch.as_tensor(state.action_mask(), dtype=torch.bool),
                    side_to_move,
                ).cpu(),
                policy_target=canonicalize_action_vector(
                    torch.as_tensor(result.policy_target, dtype=torch.float32),
                    side_to_move,
                ).cpu(),
                side_to_move=side_to_move,
                ply=ply,
                playout_cap=simulations,
                lead_target=state_lead_for_player(state, side_to_move),
                score_target=sample_score,
                score_mask=sample_score_mask,
            )
        )

        state = apply_selected_action(state, action, legal_actions)
        if decode_action(action)[0] == "move":
            pawn_visits.append((ply, side_to_move, state.pawns[side_to_move]))

        # No-progress tracking: a ply counts as progress only if it strictly
        # reduces some player's shortest distance to goal (a pawn step forward
        # or a wall that lengthens the opponent does not reset the mover's own
        # best distance). Pure pawn oscillation never improves either best, so
        # the counter climbs and the game is cut off and adjudicated.
        if no_progress_limit > 0:
            progressed = False
            for player in (Player.RED, Player.BLUE):
                distance = state.shortest_path_length(player)
                previous = best_distance[player]
                if distance is not None and (previous is None or distance < previous):
                    best_distance[player] = distance
                    progressed = True
            stalled_plies = 0 if progressed else stalled_plies + 1
            if stalled_plies >= no_progress_limit and state.winner is None:
                truncated = True
                break
    else:
        truncated = state.winner is None

    final_winner = state.winner
    if truncated:
        final_winner = adjudicated_winner(state, enabled=adjudicate_step_limit)
        if final_winner is not None:
            state.winner = final_winner
            truncated = False

    terminal_steps = len(samples)
    for index, sample in enumerate(samples):
        plies_to_end = terminal_steps - index
        sample.value_target = value_target(
            sample.side_to_move,
            final_winner,
            plies_to_end=plies_to_end,
            discount=value_discount,
        )
        sample.future_target = future_traverse_target(
            sample.side_to_move,
            sample.ply,
            pawn_visits,
        )

    return samples, final_winner, truncated, terminal_steps, handicap


def train_on_samples(
    model: AlphaZeroNet,
    optimizer: torch.optim.Optimizer,
    samples: Sequence[ReplaySample],
    *,
    device: torch.device,
    batch_size: int,
    epochs: int,
    value_loss_weight: float,
    lead_loss_weight: float,
    future_loss_weight: float,
    score_loss_weight: float,
    grad_clip: float,
    rng: random.Random,
) -> Dict[str, float]:
    valid_samples = [
        sample
        for sample in samples
        if sample.policy_target.numel() == ACTION_SIZE
        and float(sample.policy_target.sum().item()) > 0.0
        and sample.future_target is not None
    ]
    if not valid_samples:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "lead_loss": 0.0,
            "future_loss": 0.0,
            "score_loss": 0.0,
        }

    model.train()
    indices = list(range(len(valid_samples)))
    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "lead_loss": 0.0,
        "future_loss": 0.0,
        "score_loss": 0.0,
    }
    batches = 0

    for _ in range(epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            states = torch.stack([valid_samples[i].state for i in batch_indices]).to(device)
            masks = torch.stack([valid_samples[i].mask for i in batch_indices]).to(device)
            policy_targets = torch.stack(
                [valid_samples[i].policy_target for i in batch_indices]
            ).to(device)
            value_targets = torch.as_tensor(
                [valid_samples[i].value_target for i in batch_indices],
                dtype=torch.float32,
                device=device,
            )
            lead_targets = torch.as_tensor(
                [valid_samples[i].lead_target for i in batch_indices],
                dtype=torch.float32,
                device=device,
            )
            future_targets = torch.stack(
                [valid_samples[i].future_target for i in batch_indices]
            ).to(device)
            score_targets = torch.as_tensor(
                [valid_samples[i].score_target for i in batch_indices],
                dtype=torch.float32,
                device=device,
            )
            score_masks = torch.as_tensor(
                [valid_samples[i].score_mask for i in batch_indices],
                dtype=torch.float32,
                device=device,
            )

            logits, value, lead, future_logits, score = model(states)
            logits = masked_logits(logits, masks)
            policy_loss = soft_target_cross_entropy(logits, policy_targets)
            value_loss = F.mse_loss(value, value_targets)
            lead_loss = F.smooth_l1_loss(lead, lead_targets)
            future_loss = F.binary_cross_entropy_with_logits(future_logits, future_targets)
            score_loss = masked_smooth_l1_loss(score, score_targets, score_masks)
            loss = (
                policy_loss
                + value_loss_weight * value_loss
                + lead_loss_weight * lead_loss
                + future_loss_weight * future_loss
                + score_loss_weight * score_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            totals["loss"] += float(loss.item())
            totals["policy_loss"] += float(policy_loss.item())
            totals["value_loss"] += float(value_loss.item())
            totals["lead_loss"] += float(lead_loss.item())
            totals["future_loss"] += float(future_loss.item())
            totals["score_loss"] += float(score_loss.item())
            batches += 1

    divisor = max(1, batches)
    return {key: value / divisor for key, value in totals.items()}

@torch.inference_mode()
def select_model_action(
    model: AlphaZeroNet,
    state: GameState,
    *,
    device: torch.device,
    rng: random.Random,
) -> int:
    legal_actions = state.legal_actions()
    if not legal_actions:
        return -1

    model.eval()
    logits, _, _ = model.inference(encode_state(state, device).unsqueeze(0))
    logits = torch.nan_to_num(logits.squeeze(0), nan=0.0, posinf=30.0, neginf=-30.0)
    player = state.current_player
    canonical_legal = [canonical_action(action, player) for action in legal_actions]
    legal_tensor = torch.as_tensor(canonical_legal, dtype=torch.long, device=device)
    legal_logits = logits.index_select(0, legal_tensor).clamp(-30.0, 30.0)
    return int(legal_actions[int(torch.argmax(legal_logits).item())])


def choose_trained_action(
    model: AlphaZeroNet,
    state: GameState,
    *,
    device: torch.device,
    rng: random.Random,
    simulations: int,
) -> int:
    if simulations <= 0:
        return select_model_action(model, state, device=device, rng=rng)

    result = build_mcts(
        model,
        device=device,
        rng=rng,
        config=MCTSConfig(
            num_simulations=simulations,
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
    if action not in legal_actions:
        return select_model_action(model, state, device=device, rng=rng)
    return action


def run_policy_game(
    model: AlphaZeroNet,
    *,
    trained_player: Optional[Player],
    device: torch.device,
    rng: random.Random,
    handicap: Dict[str, object],
    max_steps: int,
    simulations: int,
    adjudicate_step_limit: bool,
    record_moves: bool = False,
) -> GameRecord:
    state = make_state_from_handicap(handicap)
    moves: List[MoveRecord] = []
    truncated = False

    for ply in range(max_steps):
        if state.winner is not None:
            break

        legal_actions = state.legal_actions()
        if not legal_actions:
            state.winner = state.current_player.opposite()
            break

        player = state.current_player
        if trained_player is not None and player == trained_player:
            action = choose_trained_action(
                model,
                state,
                device=device,
                rng=rng,
                simulations=simulations,
            )
        else:
            action = rng.choice(legal_actions)

        state = apply_selected_action(state, action, legal_actions)

        if record_moves:
            moves.append(
                MoveRecord(
                    ply=ply,
                    player=player,
                    action=action,
                    description=action_description(action),
                    red_position=state.pawns[Player.RED],
                    blue_position=state.pawns[Player.BLUE],
                    red_walls=state.walls_left[Player.RED],
                    blue_walls=state.walls_left[Player.BLUE],
                )
            )
    else:
        truncated = state.winner is None

    if truncated:
        winner = adjudicated_winner(state, enabled=adjudicate_step_limit)
        if winner is not None:
            state.winner = winner
            truncated = False

    return GameRecord(
        winner=state.winner,
        truncated=truncated,
        steps=len(moves) if record_moves else max_steps if truncated else 0,
        moves=moves,
        final_state=state,
    )


def evaluate_against_random(
    model: AlphaZeroNet,
    *,
    device: torch.device,
    rng: random.Random,
    games: int,
    base_walls: int,
    max_steps: int,
    simulations: int,
    adjudicate_step_limit: bool,
) -> Dict[str, float]:
    model.eval()
    wins = 0
    losses = 0
    draws = 0
    red_games = 0
    blue_games = 0
    red_wins = 0
    blue_wins = 0
    step_counts: List[int] = []

    for game_index in range(games):
        trained_player = Player.RED if game_index % 2 == 0 else Player.BLUE
        if trained_player == Player.RED:
            red_games += 1
        else:
            blue_games += 1

        handicap = sample_handicap(base_walls, rng)
        starts_as_trained = (game_index // 2) % 2 == 0
        handicap["starting_player"] = (
            trained_player if starts_as_trained else trained_player.opposite()
        )
        record = run_policy_game(
            model,
            trained_player=trained_player,
            device=device,
            rng=rng,
            handicap=handicap,
            max_steps=max_steps,
            simulations=simulations,
            adjudicate_step_limit=adjudicate_step_limit,
            record_moves=True,
        )
        step_counts.append(len(record.moves))

        if record.winner is None:
            draws += 1
        elif record.winner == trained_player:
            wins += 1
            if trained_player == Player.RED:
                red_wins += 1
            else:
                blue_wins += 1
        else:
            losses += 1

    total = max(1, games)
    return {
        "games": float(games),
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
        "win_rate": wins / total,
        "red_win_rate": red_wins / max(1, red_games),
        "blue_win_rate": blue_wins / max(1, blue_games),
        "average_steps": sum(step_counts) / max(1, len(step_counts)),
    }


def render_state(state: GameState) -> str:
    lines = [
        f"Current: {state.current_player.name}",
        f"Walls: RED={state.walls_left[Player.RED]} BLUE={state.walls_left[Player.BLUE]}",
    ]
    if state.winner is not None:
        lines.append(f"Winner: {state.winner.name}")

    for row in range(BOARD_SIZE):
        cells = []
        for col in range(BOARD_SIZE):
            cell = "."
            if state.pawns[Player.RED] == (row, col):
                cell = "R"
            elif state.pawns[Player.BLUE] == (row, col):
                cell = "B"
            cells.append(cell)
            if col < BOARD_SIZE - 1:
                cells.append("|" if state.is_blocked(row, col, row, col + 1) else " ")
        lines.append(" ".join(cells))

        if row < BOARD_SIZE - 1:
            gaps = []
            for col in range(BOARD_SIZE):
                gaps.append("-" if state.is_blocked(row, col, row + 1, col) else " ")
                if col < BOARD_SIZE - 1:
                    gaps.append("+")
            lines.append(" ".join(gaps))

    return "\n".join(lines)


def print_sample_games(
    model: AlphaZeroNet,
    *,
    device: torch.device,
    rng: random.Random,
    games: int,
    base_walls: int,
    max_steps: int,
    simulations: int,
    adjudicate_step_limit: bool,
    max_moves_to_print: int,
) -> None:
    for game_index in range(games):
        trained_player = Player.RED if game_index % 2 == 0 else Player.BLUE
        handicap = sample_handicap(base_walls, rng)
        handicap["starting_player"] = trained_player
        record = run_policy_game(
            model,
            trained_player=trained_player,
            device=device,
            rng=rng,
            handicap=handicap,
            max_steps=max_steps,
            simulations=simulations,
            adjudicate_step_limit=adjudicate_step_limit,
            record_moves=True,
        )
        result = "DRAW" if record.winner is None else record.winner.name
        print()
        print(
            f"sample_game={game_index + 1} trained={trained_player.name} "
            f"winner={result} steps={len(record.moves)} handicap={handicap['mode']}"
        )
        for move in record.moves[:max_moves_to_print]:
            print(
                f"{move.ply + 1:03d} {move.player.name:<4} "
                f"{move.description:<22} "
                f"R={move.red_position} B={move.blue_position} "
                f"walls=({move.red_walls},{move.blue_walls})"
            )
        if len(record.moves) > max_moves_to_print:
            print(f"... {len(record.moves) - max_moves_to_print} more moves")
        print(render_state(record.final_state))


def format_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"games={int(metrics['games'])} "
        f"wins={int(metrics['wins'])} "
        f"losses={int(metrics['losses'])} "
        f"draws={int(metrics['draws'])} "
        f"win_rate={metrics['win_rate']:.3f} "
        f"red={metrics['red_win_rate']:.3f} "
        f"blue={metrics['blue_win_rate']:.3f} "
        f"avg_steps={metrics['average_steps']:.1f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 7x7 Barricade AlphaZero-style policy from self-play MCTS."
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--walls", type=int, default=DEFAULT_WALLS_PER_PLAYER)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--base-simulations", type=int, default=512)
    parser.add_argument("--eval-simulations", type=int, default=256)
    parser.add_argument("--mcts-policy-temperature", type=float, default=1.0)
    parser.add_argument("--temperature-drop-ply", type=int, default=16)
    parser.add_argument("--update-every", type=int, default=50)
    parser.add_argument("--train-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=30000,
        help="Sliding-window replay buffer capacity (samples kept across updates).",
    )
    parser.add_argument(
        "--samples-per-update",
        type=int,
        default=4096,
        help="Number of replay samples drawn (with reuse) for each training update.",
    )
    parser.add_argument(
        "--value-discount",
        type=float,
        default=0.98,
        help="Per-ply discount on the outcome target so faster wins are worth "
        "more. Set to 1.0 to disable discounting.",
    )
    parser.add_argument(
        "--no-progress-limit",
        type=int,
        default=BOARD_SIZE**2,
        help="End a self-play game as a draw after this many consecutive plies "
        "without any player reducing its shortest distance to goal. 0 disables.",
    )
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--lead-loss-weight", type=float, default=0.15)
    parser.add_argument("--future-loss-weight", type=float, default=0.1)
    parser.add_argument("--score-loss-weight", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--eval-games", type=int, default=256)
    parser.add_argument("--show-games", type=int, default=50)
    parser.add_argument("--max-moves-to-print", type=int, default=80)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-adjudicate-step-limit", action="store_true")
    parser.add_argument(
        "--checkpoint-out",
        type=str,
        default="checkpoints/7x7_mcts_aux.pt",
    )
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.walls < 0:
        raise ValueError("--walls must be non-negative.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = random.Random(args.seed)
    model = AlphaZeroNet(
        hidden_channels=args.hidden_channels,
        residual_blocks=args.residual_blocks,
    ).to(device)
    # load model
    state_dict = torch.load("checkpoint_copies/best_1506.pt", map_location=device, weights_only=False)
    model.load_state_dict(state_dict["model_state"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(
        "mcts_aux_7x7 "
        f"episodes={args.episodes} walls={args.walls} "
        f"base_simulations={args.base_simulations} action_size={ACTION_SIZE} device={device}"
    )
    print(
        "pre_training_vs_random:",
        format_metrics(
            evaluate_against_random(
                model,
                device=device,
                rng=rng,
                games=max(8, min(args.eval_games, 24)),
                base_walls=args.walls,
                max_steps=args.max_steps,
                simulations=args.eval_simulations,
                adjudicate_step_limit=not args.no_adjudicate_step_limit,
            )
        ),
    )

    buffer: "deque[ReplaySample]" = deque(maxlen=max(1, args.replay_capacity))
    recent_winners: List[Optional[Player]] = []
    recent_lengths: List[int] = []
    recent_draws = 0
    recent_caps: List[int] = []
    last_stats = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "lead_loss": 0.0,
        "future_loss": 0.0,
        "score_loss": 0.0,
    }
    started_at = time.time()

    for episode in range(1, args.episodes + 1):
        samples, winner, truncated, length, _ = generate_self_play_episode(
            model,
            device=device,
            rng=rng,
            base_walls=args.walls,
            max_steps=args.max_steps,
            base_simulations=args.base_simulations,
            temperature_drop_ply=args.temperature_drop_ply,
            mcts_policy_temperature=args.mcts_policy_temperature,
            adjudicate_step_limit=not args.no_adjudicate_step_limit,
            value_discount=args.value_discount,
            no_progress_limit=args.no_progress_limit,
        )
        buffer.extend(samples)
        recent_winners.append(winner)
        recent_lengths.append(length)
        recent_caps.extend(sample.playout_cap for sample in samples)
        if truncated:
            recent_draws += 1

        if episode % args.update_every == 0 and buffer:
            if len(buffer) > args.samples_per_update:
                training_pool = rng.sample(list(buffer), args.samples_per_update)
            else:
                training_pool = list(buffer)
            last_stats = train_on_samples(
                model,
                optimizer,
                training_pool,
                device=device,
                batch_size=args.batch_size,
                epochs=args.train_epochs,
                value_loss_weight=args.value_loss_weight,
                lead_loss_weight=args.lead_loss_weight,
                future_loss_weight=args.future_loss_weight,
                score_loss_weight=args.score_loss_weight,
                grad_clip=args.grad_clip,
                rng=rng,
            )

        if episode % args.log_every == 0 or episode == args.episodes:
            window = max(1, len(recent_winners))
            red_wins = sum(1 for winner_value in recent_winners if winner_value == Player.RED)
            blue_wins = sum(1 for winner_value in recent_winners if winner_value == Player.BLUE)
            avg_length = sum(recent_lengths) / max(1, len(recent_lengths))
            avg_cap = sum(recent_caps) / max(1, len(recent_caps))
            elapsed = time.time() - started_at
            print(
                f"episode={episode} "
                f"red_win={red_wins / window:.3f} "
                f"blue_win={blue_wins / window:.3f} "
                f"draw={recent_draws / window:.3f} "
                f"avg_len={avg_length:.1f} "
                f"avg_cap={avg_cap:.1f} "
                f"loss={last_stats['loss']:.4f} "
                f"policy={last_stats['policy_loss']:.4f} "
                f"value={last_stats['value_loss']:.4f} "
                f"lead={last_stats['lead_loss']:.4f} "
                f"future={last_stats['future_loss']:.4f} "
                f"score={last_stats['score_loss']:.4f} "
                f"elapsed={elapsed:.1f}s"
            )
            recent_winners.clear()
            recent_lengths.clear()
            recent_draws = 0
            recent_caps.clear()

    if buffer:
        if len(buffer) > args.samples_per_update:
            training_pool = rng.sample(list(buffer), args.samples_per_update)
        else:
            training_pool = list(buffer)
        last_stats = train_on_samples(
            model,
            optimizer,
            training_pool,
            device=device,
            batch_size=args.batch_size,
            epochs=args.train_epochs,
            value_loss_weight=args.value_loss_weight,
            lead_loss_weight=args.lead_loss_weight,
            future_loss_weight=args.future_loss_weight,
            score_loss_weight=args.score_loss_weight,
            grad_clip=args.grad_clip,
            rng=rng,
        )
        print(
            f"final_update loss={last_stats['loss']:.4f} "
            f"policy={last_stats['policy_loss']:.4f} "
            f"value={last_stats['value_loss']:.4f} "
            f"lead={last_stats['lead_loss']:.4f} "
            f"future={last_stats['future_loss']:.4f} "
            f"score={last_stats['score_loss']:.4f}"
        )

    post_metrics = evaluate_against_random(
        model,
        device=device,
        rng=rng,
        games=args.eval_games,
        base_walls=args.walls,
        max_steps=args.max_steps,
        simulations=args.eval_simulations,
        adjudicate_step_limit=not args.no_adjudicate_step_limit,
    )
    print("post_training_vs_random:", format_metrics(post_metrics))

    if not args.no_save:
        checkpoint_path = Path(args.checkpoint_out)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "args": vars(args),
                "board_size": BOARD_SIZE,
                "action_size": ACTION_SIZE,
                "post_training_vs_random": post_metrics,
            },
            checkpoint_path,
        )
        print(f"saved_checkpoint={checkpoint_path}")

    if args.show_games > 0:
        print_sample_games(
            model,
            device=device,
            rng=rng,
            games=args.show_games,
            base_walls=args.walls,
            max_steps=args.max_steps,
            simulations=args.eval_simulations,
            adjudicate_step_limit=not args.no_adjudicate_step_limit,
            max_moves_to_print=args.max_moves_to_print,
        )


if __name__ == "__main__":
    main()
