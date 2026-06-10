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
from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


BOARD_SIZE = 7
WALL_BOARD_SIZE = BOARD_SIZE - 1
DEFAULT_WALLS_PER_PLAYER = 2
DEFAULT_MAX_STEPS = 96

MOVE_ACTIONS = 4
WALL_ACTIONS_PER_ORIENTATION = WALL_BOARD_SIZE * WALL_BOARD_SIZE
HORIZONTAL_WALL_OFFSET = MOVE_ACTIONS
VERTICAL_WALL_OFFSET = HORIZONTAL_WALL_OFFSET + WALL_ACTIONS_PER_ORIENTATION
ACTION_SIZE = MOVE_ACTIONS + WALL_ACTIONS_PER_ORIENTATION * 2

INPUT_PLANES = 9
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class Player(Enum):
    RED = 0
    BLUE = 1

    def opposite(self) -> "Player":
        return Player.BLUE if self == Player.RED else Player.RED


class WallOrientation(Enum):
    HORIZONTAL = 0
    VERTICAL = 1


class MoveDirection(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


Move = Tuple[object, ...]
WallPlacement = Tuple[WallOrientation, int, int]

MOVE_DELTAS = {
    MoveDirection.UP: (-1, 0),
    MoveDirection.DOWN: (1, 0),
    MoveDirection.LEFT: (0, -1),
    MoveDirection.RIGHT: (0, 1),
}

ALL_WALL_PLACEMENTS: Tuple[WallPlacement, ...] = tuple(
    (orientation, row, col)
    for orientation in (WallOrientation.HORIZONTAL, WallOrientation.VERTICAL)
    for row in range(WALL_BOARD_SIZE)
    for col in range(WALL_BOARD_SIZE)
)


def encode_move(move: Move) -> int:
    if move[0] == "move":
        return MoveDirection(move[1]).value

    _, orientation, row, col = move
    orientation = WallOrientation(orientation)
    offset = (
        HORIZONTAL_WALL_OFFSET
        if orientation == WallOrientation.HORIZONTAL
        else VERTICAL_WALL_OFFSET
    )
    return offset + int(row) * WALL_BOARD_SIZE + int(col)


def decode_action(action: int) -> Move:
    action = int(action)
    if 0 <= action < MOVE_ACTIONS:
        return ("move", MoveDirection(action))

    if HORIZONTAL_WALL_OFFSET <= action < VERTICAL_WALL_OFFSET:
        index = action - HORIZONTAL_WALL_OFFSET
        return (
            "wall",
            WallOrientation.HORIZONTAL,
            index // WALL_BOARD_SIZE,
            index % WALL_BOARD_SIZE,
        )

    if VERTICAL_WALL_OFFSET <= action < ACTION_SIZE:
        index = action - VERTICAL_WALL_OFFSET
        return (
            "wall",
            WallOrientation.VERTICAL,
            index // WALL_BOARD_SIZE,
            index % WALL_BOARD_SIZE,
        )

    raise ValueError(f"Action must be in [0, {ACTION_SIZE - 1}], got {action}.")


def coerce_position(position: Sequence[int], name: str) -> Tuple[int, int]:
    if len(position) != 2:
        raise ValueError(f"{name} must contain exactly two integers.")
    row, col = int(position[0]), int(position[1])
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError(f"{name} must be on the {BOARD_SIZE}x{BOARD_SIZE} board.")
    return row, col


class GameState:
    def __init__(
        self,
        *,
        red_start: Sequence[int] = (0, BOARD_SIZE // 2),
        blue_start: Sequence[int] = (BOARD_SIZE - 1, BOARD_SIZE // 2),
        red_walls: int = DEFAULT_WALLS_PER_PLAYER,
        blue_walls: int = DEFAULT_WALLS_PER_PLAYER,
        starting_player: Player = Player.RED,
    ) -> None:
        red_start = coerce_position(red_start, "red_start")
        blue_start = coerce_position(blue_start, "blue_start")
        if red_start == blue_start:
            raise ValueError("red_start and blue_start cannot be the same square.")

        self.pawns = {Player.RED: red_start, Player.BLUE: blue_start}
        self.walls: set[WallPlacement] = set()
        self.current_player = starting_player
        self.winner: Optional[Player] = None
        self.initial_walls = {Player.RED: int(red_walls), Player.BLUE: int(blue_walls)}
        self.walls_left = dict(self.initial_walls)
        self._valid_actions_cache_key: Optional[Tuple[object, ...]] = None
        self._valid_actions_cache: Optional[Tuple[int, ...]] = None
        self._path_cache: Dict[Tuple[object, ...], Optional[int]] = {}

    def copy(self) -> "GameState":
        new_state = GameState.__new__(GameState)
        new_state.pawns = dict(self.pawns)
        new_state.walls = set(self.walls)
        new_state.current_player = self.current_player
        new_state.winner = self.winner
        new_state.initial_walls = dict(self.initial_walls)
        new_state.walls_left = dict(self.walls_left)
        new_state._valid_actions_cache_key = None
        new_state._valid_actions_cache = None
        new_state._path_cache = self._path_cache
        return new_state

    def cache_key(self) -> Tuple[object, ...]:
        return (
            self.pawns[Player.RED],
            self.pawns[Player.BLUE],
            self.current_player,
            self.winner,
            self.walls_left[Player.RED],
            self.walls_left[Player.BLUE],
            frozenset(self.walls),
        )

    def legal_actions(self) -> List[int]:
        if self.winner is not None:
            return []

        key = self.cache_key()
        if key == self._valid_actions_cache_key and self._valid_actions_cache is not None:
            return list(self._valid_actions_cache)

        actions = [encode_move(move) for move in self._pawn_moves()]
        if self.walls_left[self.current_player] > 0:
            actions.extend(
                encode_move(("wall", orientation, row, col))
                for orientation, row, col in ALL_WALL_PLACEMENTS
                if self.is_valid_wall_placement(orientation, row, col)
            )

        self._valid_actions_cache_key = key
        self._valid_actions_cache = tuple(actions)
        return actions

    def action_mask(self) -> Tensor:
        mask = torch.zeros(ACTION_SIZE, dtype=torch.bool)
        for action in self.legal_actions():
            mask[action] = True
        return mask

    def _pawn_moves(self) -> List[Move]:
        moves: List[Move] = []
        row, col = self.pawns[self.current_player]
        occupied = set(self.pawns.values())

        for direction, (row_delta, col_delta) in MOVE_DELTAS.items():
            next_row = row + row_delta
            next_col = col + col_delta
            if not (0 <= next_row < BOARD_SIZE and 0 <= next_col < BOARD_SIZE):
                continue
            if (next_row, next_col) in occupied:
                continue
            if self.is_blocked(row, col, next_row, next_col):
                continue
            moves.append(("move", direction))

        return moves

    def is_blocked(self, row1: int, col1: int, row2: int, col2: int) -> bool:
        if row1 == row2:
            col_min = min(col1, col2)
            for wall_row in (row1, row1 - 1):
                if (
                    0 <= wall_row < WALL_BOARD_SIZE
                    and (WallOrientation.VERTICAL, wall_row, col_min) in self.walls
                ):
                    return True
        elif col1 == col2:
            row_min = min(row1, row2)
            for wall_col in (col1, col1 - 1):
                if (
                    0 <= wall_col < WALL_BOARD_SIZE
                    and (WallOrientation.HORIZONTAL, row_min, wall_col) in self.walls
                ):
                    return True
        return False

    def is_wall_shape_available(
        self,
        orientation: WallOrientation,
        row: int,
        col: int,
    ) -> bool:
        if row < 0 or row >= WALL_BOARD_SIZE or col < 0 or col >= WALL_BOARD_SIZE:
            return False

        if orientation == WallOrientation.HORIZONTAL:
            return not (
                (WallOrientation.HORIZONTAL, row, col) in self.walls
                or (WallOrientation.HORIZONTAL, row, col - 1) in self.walls
                or (WallOrientation.HORIZONTAL, row, col + 1) in self.walls
                or (WallOrientation.VERTICAL, row, col) in self.walls
            )

        return not (
            (WallOrientation.VERTICAL, row, col) in self.walls
            or (WallOrientation.VERTICAL, row - 1, col) in self.walls
            or (WallOrientation.VERTICAL, row + 1, col) in self.walls
            or (WallOrientation.HORIZONTAL, row, col) in self.walls
        )

    def is_valid_wall_placement(
        self,
        orientation: WallOrientation,
        row: int,
        col: int,
    ) -> bool:
        if not self.is_wall_shape_available(orientation, row, col):
            return False

        wall = (orientation, int(row), int(col))
        self.walls.add(wall)
        red_has_path = self.has_path(Player.RED)
        blue_has_path = self.has_path(Player.BLUE)
        self.walls.remove(wall)
        return red_has_path and blue_has_path

    def has_path(self, player: Player) -> bool:
        return self.shortest_path_length(player) is not None

    def shortest_path_length(self, player: Player) -> Optional[int]:
        key = ("distance", player, self.pawns[player], frozenset(self.walls))
        if key in self._path_cache:
            return self._path_cache[key]

        start = self.pawns[player]
        target_row = BOARD_SIZE - 1 if player == Player.RED else 0
        queue = [(start, 0)]
        visited = {start}
        index = 0

        while index < len(queue):
            (row, col), distance = queue[index]
            index += 1
            if row == target_row:
                self._path_cache[key] = distance
                return distance

            for row_delta, col_delta in DIRECTIONS:
                next_row = row + row_delta
                next_col = col + col_delta
                next_cell = (next_row, next_col)
                if not (0 <= next_row < BOARD_SIZE and 0 <= next_col < BOARD_SIZE):
                    continue
                if next_cell in visited:
                    continue
                if self.is_blocked(row, col, next_row, next_col):
                    continue
                visited.add(next_cell)
                queue.append((next_cell, distance + 1))

        self._path_cache[key] = None
        return None

    def apply_action(self, action: int, *, validate: bool = True) -> "GameState":
        if validate and action not in set(self.legal_actions()):
            raise ValueError(f"Illegal action {action} for current state.")

        new_state = self.copy()
        move = decode_action(action)
        if move[0] == "move":
            _, direction = move
            row_delta, col_delta = MOVE_DELTAS[MoveDirection(direction)]
            row, col = new_state.pawns[new_state.current_player]
            next_row = row + row_delta
            next_col = col + col_delta
            new_state.pawns[new_state.current_player] = (next_row, next_col)
            if next_row == BOARD_SIZE - 1 and new_state.current_player == Player.RED:
                new_state.winner = Player.RED
            elif next_row == 0 and new_state.current_player == Player.BLUE:
                new_state.winner = Player.BLUE
        else:
            _, orientation, row, col = move
            new_state.walls.add((WallOrientation(orientation), int(row), int(col)))
            new_state.walls_left[new_state.current_player] -= 1

        new_state.current_player = new_state.current_player.opposite()
        return new_state


def apply_selected_action(state: GameState, action: int, legal_actions: Sequence[int]) -> GameState:
    if action not in legal_actions:
        raise RuntimeError(
            f"Policy selected illegal action {action}; legal actions were {list(legal_actions)}"
        )
    return state.apply_action(action, validate=False)


def state_lead_for_player(state: GameState, player: Player) -> float:
    own_distance = state.shortest_path_length(player)
    opponent_distance = state.shortest_path_length(player.opposite())
    if own_distance is None or opponent_distance is None:
        return 0.0
    return float(opponent_distance - own_distance)


def path_score_for_player(state: GameState, player: Player) -> float:
    distance = state.shortest_path_length(player)
    if distance is None:
        return float(BOARD_SIZE * BOARD_SIZE)
    return float(distance)


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
    planes = torch.zeros((INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
    current = state.current_player
    opponent = current.opposite()

    if current == Player.RED:
        planes[0].fill_(1.0)

    own_row, own_col = state.pawns[current]
    opp_row, opp_col = state.pawns[opponent]
    planes[1, own_row, own_col] = 1.0
    planes[2, opp_row, opp_col] = 1.0

    for orientation, row, col in state.walls:
        plane = 3 if orientation == WallOrientation.HORIZONTAL else 4
        planes[plane, row, col] = 1.0

    own_initial = max(1, state.initial_walls[current])
    opp_initial = max(1, state.initial_walls[opponent])
    planes[5].fill_(state.walls_left[current] / own_initial)
    planes[6].fill_(state.walls_left[opponent] / opp_initial)

    own_goal = BOARD_SIZE - 1 if current == Player.RED else 0
    opp_goal = BOARD_SIZE - 1 if opponent == Player.RED else 0
    planes[7, own_goal, :] = 1.0
    planes[8, opp_goal, :] = 1.0

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
class MCTSConfig:
    num_simulations: int = 64
    cpuct: float = 1.5
    fpu_reduction: float = 0.25
    policy_temperature: float = 1.0
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    policy_target_temperature: float = 1.0
    action_temperature: float = 1.0
    lead_weight: float = 0.02
    lead_scale: float = 5.0
    add_root_noise: bool = True


@dataclass
class SearchEvaluation:
    value: float
    lead: float


@dataclass
class SearchEdge:
    action: int
    prior: float
    child: Optional["SearchNode"] = None
    visits: int = 0
    value_sum: float = 0.0
    lead_sum: float = 0.0

    @property
    def q(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    @property
    def lead_q(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.lead_sum / self.visits


@dataclass
class SearchNode:
    state: GameState
    edges: Dict[int, SearchEdge]
    is_expanded: bool = False
    value_estimate: float = 0.0
    lead_estimate: float = 0.0

    @property
    def visits(self) -> int:
        return sum(edge.visits for edge in self.edges.values())


@dataclass
class MCTSResult:
    action: int
    policy_target: Tensor
    root_value: float
    root_lead: float
    diagnostics: Dict[str, int]


class MCTS:
    def __init__(
        self,
        model: AlphaZeroNet,
        *,
        device: torch.device,
        rng: random.Random,
        config: MCTSConfig,
    ) -> None:
        self.model = model
        self.device = device
        self.rng = rng
        self.config = config

    def search(self, state: GameState) -> MCTSResult:
        root = SearchNode(state=state.copy(), edges={})
        evaluation = self._terminal_evaluation(root)
        if evaluation is None:
            self._expand(root)
            if self.config.add_root_noise:
                self._add_root_noise(root)

        completed = 0
        while completed < self.config.num_simulations:
            node = root
            path: List[SearchEdge] = []

            while node.is_expanded and node.edges:
                edge = self._select_edge(node)
                path.append(edge)
                if edge.child is None:
                    child_state = apply_selected_action(
                        node.state,
                        edge.action,
                        list(node.edges.keys()),
                    )
                    edge.child = SearchNode(state=child_state, edges={})
                node = edge.child

            evaluation = self._terminal_evaluation(node)
            if evaluation is None:
                evaluation = self._expand(node)

            self._backpropagate(path, evaluation)
            completed += 1

        policy_target = self._policy_target(root, self.config.policy_target_temperature)
        action_policy = self._policy_target(root, self.config.action_temperature)
        action = self._sample_policy(action_policy)
        return MCTSResult(
            action=action,
            policy_target=torch.as_tensor(policy_target, dtype=torch.float32),
            root_value=self._root_value(root),
            root_lead=self._root_lead(root),
            diagnostics={"completed_simulations": completed},
        )

    def _select_edge(self, node: SearchNode) -> SearchEdge:
        parent_visits = max(1, node.visits)
        best_score = -math.inf
        best_edge: Optional[SearchEdge] = None
        parent_q = self._parent_q(node)
        explored_prior = sum(
            edge.prior for edge in node.edges.values() if edge.visits > 0
        )
        fpu_q = parent_q - self.config.fpu_reduction * math.sqrt(
            max(0.0, explored_prior)
        )

        for edge in node.edges.values():
            q_value = edge.q if edge.visits > 0 else fpu_q
            u_value = (
                self.config.cpuct
                * edge.prior
                * math.sqrt(parent_visits)
                / (1 + edge.visits)
            )
            lead_bonus = self.config.lead_weight * math.tanh(
                edge.lead_q / max(1.0e-6, self.config.lead_scale)
            )
            score = q_value + u_value + lead_bonus
            if score > best_score:
                best_score = score
                best_edge = edge

        if best_edge is None:
            raise RuntimeError("Cannot select from an unexpanded node.")
        return best_edge

    @torch.no_grad()
    def _expand(self, node: SearchNode) -> SearchEvaluation:
        legal_actions = node.state.legal_actions()
        if not legal_actions:
            node.state.winner = node.state.current_player.opposite()
            node.is_expanded = True
            return SearchEvaluation(value=-1.0, lead=state_lead_for_player(node.state, node.state.current_player))

        self.model.eval()
        state_tensor = encode_state(node.state, self.device).unsqueeze(0)
        logits, value, lead = self.model.inference(state_tensor)
        logits = torch.nan_to_num(logits.squeeze(0), nan=0.0, posinf=30.0, neginf=-30.0)
        legal_tensor = torch.as_tensor(legal_actions, dtype=torch.long, device=self.device)
        legal_logits = logits.index_select(0, legal_tensor).clamp(-30.0, 30.0)
        temperature = max(self.config.policy_temperature, 1.0e-6)
        priors = torch.softmax(legal_logits / temperature, dim=0).detach().cpu().tolist()
        if not all(math.isfinite(float(prob)) for prob in priors) or sum(priors) <= 0:
            priors = [1.0 / len(legal_actions)] * len(legal_actions)

        total_prior = sum(float(prob) for prob in priors)
        node.edges = {
            action: SearchEdge(action=action, prior=float(prob) / total_prior)
            for action, prob in zip(legal_actions, priors)
        }
        node.is_expanded = True
        node.value_estimate = float(value.squeeze(0).item())
        node.lead_estimate = float(lead.squeeze(0).item())
        return SearchEvaluation(value=node.value_estimate, lead=node.lead_estimate)

    def _add_root_noise(self, root: SearchNode) -> None:
        if not root.edges:
            return

        alpha = self.config.root_dirichlet_alpha
        fraction = self.config.root_exploration_fraction
        if alpha <= 0.0 or fraction <= 0.0:
            return

        actions = list(root.edges.keys())
        noise = [self.rng.gammavariate(alpha, 1.0) for _ in actions]
        total = sum(noise)
        if total <= 0.0:
            return
        noise = [value / total for value in noise]
        for action, noise_value in zip(actions, noise):
            edge = root.edges[action]
            edge.prior = (1.0 - fraction) * edge.prior + fraction * noise_value

    def _backpropagate(
        self,
        path: Sequence[SearchEdge],
        leaf_evaluation: SearchEvaluation,
    ) -> None:
        node_value = float(leaf_evaluation.value)
        node_lead = float(leaf_evaluation.lead)
        for edge in reversed(path):
            parent_value = -node_value
            parent_lead = -node_lead
            edge.value_sum += parent_value
            edge.lead_sum += parent_lead
            edge.visits += 1
            node_value = parent_value
            node_lead = parent_lead

    def _terminal_evaluation(self, node: SearchNode) -> Optional[SearchEvaluation]:
        if node.state.winner is not None:
            value = 1.0 if node.state.winner == node.state.current_player else -1.0
            return SearchEvaluation(
                value=value,
                lead=state_lead_for_player(node.state, node.state.current_player),
            )
        return None

    @staticmethod
    def _parent_q(node: SearchNode) -> float:
        visits = sum(edge.visits for edge in node.edges.values())
        if visits <= 0:
            return 0.0
        return sum(edge.value_sum for edge in node.edges.values()) / visits

    def _policy_target(self, root: SearchNode, temperature: float) -> List[float]:
        policy = [0.0] * ACTION_SIZE
        visited_edges = [edge for edge in root.edges.values() if edge.visits > 0]
        if not visited_edges:
            legal_actions = list(root.edges.keys())
            if not legal_actions:
                return policy
            for action in legal_actions:
                policy[action] = 1.0 / len(legal_actions)
            return policy

        if temperature <= 0.0:
            best = max(visited_edges, key=lambda edge: (edge.visits, edge.q, edge.lead_q))
            policy[best.action] = 1.0
            return policy

        weights = [
            (edge.action, float(edge.visits) ** (1.0 / max(temperature, 1.0e-6)))
            for edge in visited_edges
        ]
        total = sum(weight for _, weight in weights)
        if total <= 0.0:
            for edge in visited_edges:
                policy[edge.action] = 1.0 / len(visited_edges)
            return policy

        for action, weight in weights:
            policy[action] = weight / total
        return policy

    def _sample_policy(self, policy: Sequence[float]) -> int:
        total = float(sum(policy))
        if total <= 0.0:
            return -1
        return int(self.rng.choices(range(len(policy)), weights=policy, k=1)[0])

    @staticmethod
    def _root_value(root: SearchNode) -> float:
        visits = root.visits
        if visits <= 0:
            return root.value_estimate
        return sum(edge.value_sum for edge in root.edges.values()) / visits

    @staticmethod
    def _root_lead(root: SearchNode) -> float:
        visits = root.visits
        if visits <= 0:
            return root.lead_estimate
        return sum(edge.lead_sum for edge in root.edges.values()) / visits


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
    if roll < 0.70:
        cap = base_simulations
    elif roll < 0.90:
        cap = max(1, base_simulations // 2)
    else:
        cap = base_simulations * 2
    return max(1, int(cap))


def sample_handicap(base_walls: int, rng: random.Random) -> Dict[str, object]:
    center = BOARD_SIZE // 2
    roll = rng.random()
    starting_player = Player.RED if rng.random() < 0.5 else Player.BLUE
    if roll < 0.60:
        config = {
            "mode": "standard",
            "red_start": (0, center),
            "blue_start": (BOARD_SIZE - 1, center),
            "red_walls": base_walls,
            "blue_walls": base_walls,
            "starting_player": starting_player,
        }
    elif roll < 0.80:
        config = {
            "mode": "column_shift",
            "red_start": (0, rng.randint(1, BOARD_SIZE - 2)),
            "blue_start": (BOARD_SIZE - 1, rng.randint(1, BOARD_SIZE - 2)),
            "red_walls": base_walls,
            "blue_walls": base_walls,
            "starting_player": starting_player,
        }
    elif roll < 0.93:
        config = {
            "mode": "row_ahead",
            "red_start": (rng.randint(0, 1), rng.randint(1, BOARD_SIZE - 2)),
            "blue_start": (rng.randint(BOARD_SIZE - 2, BOARD_SIZE - 1), rng.randint(1, BOARD_SIZE - 2)),
            "red_walls": base_walls,
            "blue_walls": base_walls,
            "starting_player": starting_player,
        }
    else:
        config = {
            "mode": "wall_handicap",
            "red_start": (0, center),
            "blue_start": (BOARD_SIZE - 1, center),
            "red_walls": max(0, base_walls + rng.randint(-2, 2)),
            "blue_walls": max(0, base_walls + rng.randint(-2, 2)),
            "starting_player": starting_player,
        }
    return config


def make_state_from_handicap(handicap: Dict[str, object]) -> GameState:
    return GameState(
        red_start=handicap["red_start"],
        blue_start=handicap["blue_start"],
        red_walls=int(handicap["red_walls"]),
        blue_walls=int(handicap["blue_walls"]),
        starting_player=handicap["starting_player"],
    )


def future_traverse_target(
    side_to_move: Player,
    ply: int,
    pawn_visits: Sequence[Tuple[int, Player, Tuple[int, int]]],
) -> Tensor:
    target = torch.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
    for visit_ply, player, position in pawn_visits:
        if visit_ply < ply:
            continue
        row, col = position
        channel = 0 if player == side_to_move else 1
        target[channel, int(row), int(col)] = 1.0
    return target


def value_target(side_to_move: Player, winner: Optional[Player]) -> float:
    if winner is None:
        return 0.0
    return 1.0 if side_to_move == winner else -1.0


def score_target(state: GameState, side_to_move: Player) -> Tuple[float, float]:
    return path_score_for_player(state, side_to_move), 1.0


def action_description(action: int) -> str:
    move = decode_action(action)
    if move[0] == "move":
        return f"move {MoveDirection(move[1]).name}"
    _, orientation, row, col = move
    return f"wall {WallOrientation(orientation).name} r={row} c={col}"


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
) -> Tuple[List[ReplaySample], Optional[Player], bool, int, Dict[str, object]]:
    handicap = sample_handicap(base_walls, rng)
    state = make_state_from_handicap(handicap)
    samples: List[ReplaySample] = []
    pawn_visits: List[Tuple[int, Player, Tuple[int, int]]] = []
    truncated = False

    for ply in range(max_steps):
        if state.winner is not None:
            break

        legal_actions = state.legal_actions()
        if not legal_actions:
            state.winner = state.current_player.opposite()
            break

        side_to_move = state.current_player
        simulations = sample_playout_cap(base_simulations, rng)
        mcts = MCTS(
            model,
            device=device,
            rng=rng,
            config=MCTSConfig(
                num_simulations=simulations,
                policy_temperature=mcts_policy_temperature,
                action_temperature=1.0 if ply < temperature_drop_ply else 0.0,
                add_root_noise=True,
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
                mask=state.action_mask().cpu(),
                policy_target=result.policy_target.cpu(),
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
    else:
        truncated = state.winner is None

    final_winner = state.winner
    if truncated:
        final_winner = adjudicated_winner(state, enabled=adjudicate_step_limit)
        if final_winner is not None:
            state.winner = final_winner
            truncated = False

    terminal_steps = len(samples)
    for sample in samples:
        sample.value_target = value_target(sample.side_to_move, final_winner)
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


@torch.no_grad()
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
    legal_tensor = torch.as_tensor(legal_actions, dtype=torch.long, device=device)
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

    result = MCTS(
        model,
        device=device,
        rng=rng,
        config=MCTSConfig(
            num_simulations=simulations,
            action_temperature=0.0,
            add_root_noise=False,
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
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--walls", type=int, default=DEFAULT_WALLS_PER_PLAYER)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--base-simulations", type=int, default=32)
    parser.add_argument("--eval-simulations", type=int, default=16)
    parser.add_argument("--mcts-policy-temperature", type=float, default=1.0)
    parser.add_argument("--temperature-drop-ply", type=int, default=16)
    parser.add_argument("--update-every", type=int, default=10)
    parser.add_argument("--train-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--lead-loss-weight", type=float, default=0.1)
    parser.add_argument("--future-loss-weight", type=float, default=0.1)
    parser.add_argument("--score-loss-weight", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--eval-games", type=int, default=80)
    parser.add_argument("--show-games", type=int, default=3)
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

    buffer: List[ReplaySample] = []
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
        )
        buffer.extend(samples)
        recent_winners.append(winner)
        recent_lengths.append(length)
        recent_caps.extend(sample.playout_cap for sample in samples)
        if truncated:
            recent_draws += 1

        if episode % args.update_every == 0:
            last_stats = train_on_samples(
                model,
                optimizer,
                buffer,
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
            buffer.clear()

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
        last_stats = train_on_samples(
            model,
            optimizer,
            buffer,
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
        buffer.clear()
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
