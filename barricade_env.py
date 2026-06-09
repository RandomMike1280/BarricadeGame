"""
Headless Barricade reinforcement-learning environment.

This module mirrors the rules used by ``barricade_pygame.py`` without importing
Pygame. It exposes a Gymnasium-style API:

    observation, info = env.reset()
    observation, reward, terminated, truncated, info = env.step(action)

The action space is fixed at 132 discrete actions:
    0        move pawn one square up
    1        move pawn one square down
    2        move pawn one square left
    3        move pawn one square right
    4..67    place a horizontal wall at an 8x8 wall anchor
    68..131  place a vertical wall at an 8x8 wall anchor
"""

from __future__ import annotations

from enum import Enum
import heapq
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only when numpy is absent.
    np = None

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised only when gymnasium is absent.
    gym = None
    spaces = None


BOARD_SIZE = 9
WALL_BOARD_SIZE = BOARD_SIZE - 1
DEFAULT_WALLS_PER_PLAYER = 10
DEFAULT_RED_START = (0, 4)
DEFAULT_BLUE_START = (8, 4)

MOVE_ACTIONS = 4
WALL_ACTIONS_PER_ORIENTATION = WALL_BOARD_SIZE * WALL_BOARD_SIZE
HORIZONTAL_WALL_OFFSET = MOVE_ACTIONS
VERTICAL_WALL_OFFSET = HORIZONTAL_WALL_OFFSET + WALL_ACTIONS_PER_ORIENTATION
ACTION_SIZE = MOVE_ACTIONS + WALL_ACTIONS_PER_ORIENTATION * 2

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


Move = Tuple[Any, ...]
WallPlacement = Tuple[WallOrientation, int, int]

MOVE_DIRECTION_DELTAS = {
    MoveDirection.UP: (-1, 0),
    MoveDirection.DOWN: (1, 0),
    MoveDirection.LEFT: (0, -1),
    MoveDirection.RIGHT: (0, 1),
}

ALL_WALL_PLACEMENTS = tuple(
    (orient, r, c)
    for orient in (WallOrientation.HORIZONTAL, WallOrientation.VERTICAL)
    for r in range(WALL_BOARD_SIZE)
    for c in range(WALL_BOARD_SIZE)
)


def coerce_player(player: Player | str | int) -> Player:
    if isinstance(player, Player):
        return player
    if isinstance(player, str):
        return Player[player.upper()]
    return Player(player)


def coerce_orientation(orientation: WallOrientation | str | int) -> WallOrientation:
    if isinstance(orientation, WallOrientation):
        return orientation
    if isinstance(orientation, str):
        return WallOrientation[orientation.upper()]
    return WallOrientation(orientation)


def coerce_move_direction(direction: MoveDirection | str | int) -> MoveDirection:
    if isinstance(direction, MoveDirection):
        return direction
    if isinstance(direction, str):
        return MoveDirection[direction.upper()]
    return MoveDirection(direction)


def coerce_position(position: Sequence[int], name: str) -> Tuple[int, int]:
    if len(position) != 2:
        raise ValueError(f"{name} must contain exactly two values: (row, col).")
    row, col = int(position[0]), int(position[1])
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError(f"{name} must be on the {BOARD_SIZE}x{BOARD_SIZE} board.")
    return row, col


def coerce_wall_count(count: int, name: str) -> int:
    count = int(count)
    if count < 0:
        raise ValueError(f"{name} must be non-negative.")
    return count


def encode_move(move: Move) -> int:
    """Encode a move tuple into the fixed discrete action id."""
    move_type = move[0]
    if move_type == "move":
        if len(move) != 2:
            raise ValueError("Move actions must be ('move', direction).")
        _, direction = move
        return coerce_move_direction(direction).value

    if move_type == "wall":
        _, orientation, row, col = move
        orientation = coerce_orientation(orientation)
        row = int(row)
        col = int(col)
        if not (0 <= row < WALL_BOARD_SIZE and 0 <= col < WALL_BOARD_SIZE):
            raise ValueError(f"Wall anchor out of bounds: {(row, col)}")
        offset = (
            HORIZONTAL_WALL_OFFSET
            if orientation == WallOrientation.HORIZONTAL
            else VERTICAL_WALL_OFFSET
        )
        return offset + row * WALL_BOARD_SIZE + col

    raise ValueError(f"Unknown move type: {move_type!r}")


def decode_action(action: int) -> Move:
    """Decode a discrete action id into a game move tuple."""
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

    raise ValueError(f"Action must be in [0, {ACTION_SIZE - 1}], got {action}")


class BarricadeState:
    """Pure game state and rule engine for Barricade."""

    def __init__(
        self,
        *,
        red_start: Sequence[int] = DEFAULT_RED_START,
        blue_start: Sequence[int] = DEFAULT_BLUE_START,
        red_walls: int = DEFAULT_WALLS_PER_PLAYER,
        blue_walls: int = DEFAULT_WALLS_PER_PLAYER,
    ) -> None:
        red_start = coerce_position(red_start, "red_start")
        blue_start = coerce_position(blue_start, "blue_start")
        if red_start == blue_start:
            raise ValueError("red_start and blue_start cannot be the same square.")

        red_walls = coerce_wall_count(red_walls, "red_walls")
        blue_walls = coerce_wall_count(blue_walls, "blue_walls")

        self.pawns = {Player.RED: red_start, Player.BLUE: blue_start}
        self.walls: Set[WallPlacement] = set()
        self.current_player = Player.RED
        self.winner: Optional[Player] = None
        self.initial_walls = {Player.RED: red_walls, Player.BLUE: blue_walls}
        self.walls_left = {Player.RED: red_walls, Player.BLUE: blue_walls}

        self._valid_moves_cache_key: Optional[Tuple[Any, ...]] = None
        self._valid_moves_cache: Optional[Tuple[Move, ...]] = None
        self._path_cache: Dict[Tuple[Any, ...], Optional[int]] = {}
        self._route_cache: Dict[Tuple[Any, ...], Optional[Tuple[Tuple[int, int], ...]]] = {}

    def copy(self) -> "BarricadeState":
        new_state = BarricadeState.__new__(BarricadeState)
        new_state.pawns = dict(self.pawns)
        new_state.walls = set(self.walls)
        new_state.current_player = self.current_player
        new_state.winner = self.winner
        new_state.initial_walls = dict(self.initial_walls)
        new_state.walls_left = dict(self.walls_left)
        new_state._valid_moves_cache_key = None
        new_state._valid_moves_cache = None
        new_state._path_cache = self._path_cache
        new_state._route_cache = self._route_cache
        return new_state

    def state_cache_key(self) -> Tuple[Any, ...]:
        return (
            self.pawns[Player.RED],
            self.pawns[Player.BLUE],
            self.current_player,
            self.winner,
            self.walls_left[Player.RED],
            self.walls_left[Player.BLUE],
            frozenset(self.walls),
        )

    def get_valid_moves(self) -> List[Move]:
        if self.winner is not None:
            return []

        key = self.state_cache_key()
        if key == self._valid_moves_cache_key and self._valid_moves_cache is not None:
            return list(self._valid_moves_cache)

        moves = self.get_pawn_moves()
        moves.extend(self.get_valid_wall_moves())
        self._valid_moves_cache_key = key
        self._valid_moves_cache = tuple(moves)
        return moves

    def get_pawn_moves(self) -> List[Move]:
        if self.winner is not None:
            return []

        moves: List[Move] = []
        pawn_row, pawn_col = self.pawns[self.current_player]

        for direction, (row_delta, col_delta) in MOVE_DIRECTION_DELTAS.items():
            next_row = pawn_row + row_delta
            next_col = pawn_col + col_delta

            if not (0 <= next_row < BOARD_SIZE and 0 <= next_col < BOARD_SIZE):
                continue
            if self.is_blocked(pawn_row, pawn_col, next_row, next_col):
                continue
            if (next_row, next_col) in self.pawns.values():
                continue

            moves.append(("move", direction))

        return moves

    def get_valid_wall_moves(
        self, candidates: Optional[Iterable[WallPlacement]] = None
    ) -> List[Move]:
        if self.walls_left[self.current_player] <= 0:
            return []

        wall_moves: List[Move] = []
        placements = ALL_WALL_PLACEMENTS if candidates is None else candidates
        for orientation, row, col in placements:
            if self.is_valid_wall_placement(orientation, row, col):
                wall_moves.append(("wall", orientation, row, col))

        return wall_moves

    def is_blocked(self, row1: int, col1: int, row2: int, col2: int) -> bool:
        """Return True when a direct adjacent step is blocked by a wall."""
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

    def is_valid_wall_placement(
        self, orientation: WallOrientation | str | int, row: int, col: int
    ) -> bool:
        orientation = coerce_orientation(orientation)
        if not self.is_wall_shape_available(orientation, row, col):
            return False

        wall = (orientation, row, col)
        self.walls.add(wall)
        can_reach_red = self.has_path(Player.RED)
        can_reach_blue = self.has_path(Player.BLUE)
        self.walls.remove(wall)

        return can_reach_red and can_reach_blue

    def is_wall_shape_available(
        self, orientation: WallOrientation | str | int, row: int, col: int
    ) -> bool:
        orientation = coerce_orientation(orientation)
        if row < 0 or row >= WALL_BOARD_SIZE or col < 0 or col >= WALL_BOARD_SIZE:
            return False

        if orientation == WallOrientation.HORIZONTAL:
            if (WallOrientation.HORIZONTAL, row, col) in self.walls:
                return False
            if (WallOrientation.HORIZONTAL, row, col - 1) in self.walls:
                return False
            if (WallOrientation.HORIZONTAL, row, col + 1) in self.walls:
                return False
            if (WallOrientation.VERTICAL, row, col) in self.walls:
                return False
        else:
            if (WallOrientation.VERTICAL, row, col) in self.walls:
                return False
            if (WallOrientation.VERTICAL, row - 1, col) in self.walls:
                return False
            if (WallOrientation.VERTICAL, row + 1, col) in self.walls:
                return False
            if (WallOrientation.HORIZONTAL, row, col) in self.walls:
                return False

        return True

    def has_path(self, player: Player | str | int) -> bool:
        player = coerce_player(player)
        key = ("greedy_reachable", player, self.pawns[player], frozenset(self.walls))
        if key in self._path_cache:
            return bool(self._path_cache[key])

        start = self.pawns[player]
        target_row = BOARD_SIZE - 1 if player == Player.RED else 0
        frontier = [(self._path_heuristic(start, target_row), start)]
        visited = {start}

        while frontier:
            _, (row, col) = heapq.heappop(frontier)
            if row == target_row:
                self._path_cache[key] = 1
                return True

            for row_delta, col_delta in DIRECTIONS:
                next_row = row + row_delta
                next_col = col + col_delta
                next_cell = (next_row, next_col)
                if (
                    0 <= next_row < BOARD_SIZE
                    and 0 <= next_col < BOARD_SIZE
                    and next_cell not in visited
                    and not self.is_blocked(row, col, next_row, next_col)
                ):
                    visited.add(next_cell)
                    heapq.heappush(
                        frontier,
                        (self._path_heuristic(next_cell, target_row), next_cell),
                    )

        self._path_cache[key] = 0
        return False

    def greedy_path_length(self, player: Player | str | int) -> Optional[int]:
        player = coerce_player(player)
        key = ("greedy_distance", player, self.pawns[player], frozenset(self.walls))
        if key in self._path_cache:
            return self._path_cache[key]

        start = self.pawns[player]
        target_row = BOARD_SIZE - 1 if player == Player.RED else 0
        frontier = [(self._path_heuristic(start, target_row), 0, start)]
        visited = {start}

        while frontier:
            _, distance, (row, col) = heapq.heappop(frontier)
            if row == target_row:
                self._path_cache[key] = distance
                return distance

            for row_delta, col_delta in DIRECTIONS:
                next_row = row + row_delta
                next_col = col + col_delta
                next_cell = (next_row, next_col)
                if 0 <= next_row < BOARD_SIZE and 0 <= next_col < BOARD_SIZE:
                    if next_cell not in visited and not self.is_blocked(
                        row, col, next_row, next_col
                    ):
                        visited.add(next_cell)
                        heapq.heappush(
                            frontier,
                            (
                                self._path_heuristic(next_cell, target_row),
                                distance + 1,
                                next_cell,
                            ),
                        )

        self._path_cache[key] = None
        return None

    def greedy_path_cells(
        self, player: Player | str | int
    ) -> Optional[Tuple[Tuple[int, int], ...]]:
        player = coerce_player(player)
        key = ("greedy_route", player, self.pawns[player], frozenset(self.walls))
        if key in self._route_cache:
            return self._route_cache[key]

        start = self.pawns[player]
        target_row = BOARD_SIZE - 1 if player == Player.RED else 0
        frontier = [(self._path_heuristic(start, target_row), 0, start)]
        parent = {start: None}

        while frontier:
            _, distance, (row, col) = heapq.heappop(frontier)
            if row == target_row:
                path = []
                node: Optional[Tuple[int, int]] = (row, col)
                while node is not None:
                    path.append(node)
                    node = parent[node]
                result = tuple(reversed(path))
                self._route_cache[key] = result
                return result

            for row_delta, col_delta in DIRECTIONS:
                next_row = row + row_delta
                next_col = col + col_delta
                next_cell = (next_row, next_col)
                if (
                    0 <= next_row < BOARD_SIZE
                    and 0 <= next_col < BOARD_SIZE
                    and next_cell not in parent
                    and not self.is_blocked(row, col, next_row, next_col)
                ):
                    parent[next_cell] = (row, col)
                    heapq.heappush(
                        frontier,
                        (
                            self._path_heuristic(next_cell, target_row),
                            distance + 1,
                            next_cell,
                        ),
                    )

        self._route_cache[key] = None
        return None

    @staticmethod
    def _path_heuristic(cell: Tuple[int, int], target_row: int) -> int:
        return abs(cell[0] - target_row)

    def apply_move(self, move: Move) -> "BarricadeState":
        new_state = self.copy()
        move_type = move[0]

        if move_type == "move":
            _, direction = move
            direction = coerce_move_direction(direction)
            row_delta, col_delta = MOVE_DIRECTION_DELTAS[direction]
            current_row, current_col = new_state.pawns[new_state.current_player]
            row = current_row + row_delta
            col = current_col + col_delta
            new_state.pawns[new_state.current_player] = (row, col)
            if row == BOARD_SIZE - 1 and new_state.current_player == Player.RED:
                new_state.winner = Player.RED
            elif row == 0 and new_state.current_player == Player.BLUE:
                new_state.winner = Player.BLUE
        elif move_type == "wall":
            _, orientation, row, col = move
            orientation = coerce_orientation(orientation)
            new_state.walls.add((orientation, int(row), int(col)))
            new_state.walls_left[new_state.current_player] -= 1
        else:
            raise ValueError(f"Unknown move type: {move_type!r}")

        new_state.current_player = new_state.current_player.opposite()
        return new_state


class SimpleDiscrete:
    """Small fallback for ``action_space`` when Gymnasium is not installed."""

    def __init__(self, n: int) -> None:
        self.n = n

    def sample(self, mask: Optional[Sequence[int]] = None) -> int:
        if mask is None:
            return random.randrange(self.n)
        legal_actions = [index for index, is_legal in enumerate(mask) if is_legal]
        if not legal_actions:
            raise ValueError("Cannot sample from an empty legal action mask.")
        return random.choice(legal_actions)


BaseEnv = gym.Env if gym is not None else object


class BarricadeEnv(BaseEnv):
    """
    Alternating-turn RL environment for Barricade.

    Rewards are from the perspective of the player that submitted the action:
    non-terminal legal moves receive ``step_penalty``; a winning move receives
    ``win_reward``; invalid actions terminate the episode with
    ``invalid_action_penalty`` by default.
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 0}

    def __init__(
        self,
        *,
        starting_player: Player | str | int = Player.RED,
        red_start: Sequence[int] = DEFAULT_RED_START,
        blue_start: Sequence[int] = DEFAULT_BLUE_START,
        red_walls: int = DEFAULT_WALLS_PER_PLAYER,
        blue_walls: int = DEFAULT_WALLS_PER_PLAYER,
        max_steps: int = 500,
        invalid_action_mode: str = "terminate",
        render_mode: Optional[str] = None,
        win_reward: float = 1.0,
        loss_reward: float = -1.0,
        step_penalty: float = 0.0,
        invalid_action_penalty: float = -1.0,
    ) -> None:
        if invalid_action_mode not in {"terminate", "raise"}:
            raise ValueError("invalid_action_mode must be 'terminate' or 'raise'.")
        if render_mode not in {None, "ansi", "human"}:
            raise ValueError("render_mode must be None, 'ansi', or 'human'.")

        self.starting_player = coerce_player(starting_player)
        self.red_start = coerce_position(red_start, "red_start")
        self.blue_start = coerce_position(blue_start, "blue_start")
        self.red_walls = coerce_wall_count(red_walls, "red_walls")
        self.blue_walls = coerce_wall_count(blue_walls, "blue_walls")
        self._validate_initial_config(
            self.red_start,
            self.blue_start,
            self.red_walls,
            self.blue_walls,
        )

        self.max_steps = int(max_steps)
        self.invalid_action_mode = invalid_action_mode
        self.render_mode = render_mode
        self.win_reward = float(win_reward)
        self.loss_reward = float(loss_reward)
        self.step_penalty = float(step_penalty)
        self.invalid_action_penalty = float(invalid_action_penalty)

        self.rng = random.Random()
        self.state = self._new_state(
            self.red_start,
            self.blue_start,
            self.red_walls,
            self.blue_walls,
        )
        self.steps = 0
        self.pawn_moves = {Player.RED: 0, Player.BLUE: 0}
        self.terminated = False
        self.truncated = False

        if spaces is not None and np is not None:
            self.action_space = spaces.Discrete(ACTION_SIZE)
            self.observation_space = spaces.Dict(
                {
                    "board": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(4, BOARD_SIZE, BOARD_SIZE),
                        dtype=np.float32,
                    ),
                    "features": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(5,),
                        dtype=np.float32,
                    ),
                    "action_mask": spaces.MultiBinary(ACTION_SIZE),
                }
            )
        else:
            self.action_space = SimpleDiscrete(ACTION_SIZE)
            self.observation_space = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            self.rng.seed(seed)

        options = options or {}
        starting_player = coerce_player(options.get("starting_player", self.starting_player))
        red_start = coerce_position(options.get("red_start", self.red_start), "red_start")
        blue_start = coerce_position(options.get("blue_start", self.blue_start), "blue_start")
        red_walls = coerce_wall_count(options.get("red_walls", self.red_walls), "red_walls")
        blue_walls = coerce_wall_count(options.get("blue_walls", self.blue_walls), "blue_walls")
        self._validate_initial_config(red_start, blue_start, red_walls, blue_walls)

        self.state = self._new_state(red_start, blue_start, red_walls, blue_walls)
        self.state.current_player = starting_player
        self.steps = 0
        self.pawn_moves = {Player.RED: 0, Player.BLUE: 0}
        self.terminated = False
        self.truncated = False

        return self._get_observation(), self._get_info()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self.terminated or self.truncated:
            raise RuntimeError("step() called after episode ended. Call reset() first.")

        acting_player = self.state.current_player

        try:
            decoded_move = decode_action(int(action))
        except (TypeError, ValueError) as exc:
            return self._handle_invalid_action(action, acting_player, str(exc))

        legal_action_set = set(self.legal_actions())
        if int(action) not in legal_action_set:
            return self._handle_invalid_action(
                action, acting_player, f"Illegal action for current state: {action}"
            )

        self.state = self.state.apply_move(decoded_move)
        if decoded_move[0] == "move":
            self.pawn_moves[acting_player] += 1
        self.steps += 1

        reward = self.step_penalty
        if self.state.winner is not None:
            self.terminated = True
            reward = (
                self.win_reward
                if self.state.winner == acting_player
                else self.loss_reward
            )
        elif not self.state.get_valid_moves():
            self.state.winner = acting_player
            self.terminated = True
            reward = self.win_reward
        elif self.steps >= self.max_steps:
            self.truncated = True

        info = self._get_info()
        info.update(
            {
                "acting_player": acting_player.name,
                "last_action": int(action),
                "last_move": self.move_to_dict(decoded_move),
                "invalid_action": False,
            }
        )
        return self._get_observation(), reward, self.terminated, self.truncated, info

    def legal_moves(self) -> List[Move]:
        return self.state.get_valid_moves()

    def legal_actions(self) -> List[int]:
        return [encode_move(move) for move in self.state.get_valid_moves()]

    def legal_action_mask(self) -> Any:
        mask = [0] * ACTION_SIZE
        for action in self.legal_actions():
            mask[action] = 1
        if np is not None:
            return np.array(mask, dtype=np.int8)
        return mask

    def sample_legal_action(self) -> Optional[int]:
        legal_actions = self.legal_actions()
        if not legal_actions:
            return None
        return self.rng.choice(legal_actions)

    def render(self) -> Optional[str]:
        output = self._render_ansi()
        if self.render_mode == "human":
            print(output)
            return None
        return output

    def close(self) -> None:
        pass

    def _new_state(
        self,
        red_start: Tuple[int, int],
        blue_start: Tuple[int, int],
        red_walls: int,
        blue_walls: int,
    ) -> BarricadeState:
        return BarricadeState(
            red_start=red_start,
            blue_start=blue_start,
            red_walls=red_walls,
            blue_walls=blue_walls,
        )

    @staticmethod
    def _validate_initial_config(
        red_start: Tuple[int, int],
        blue_start: Tuple[int, int],
        red_walls: int,
        blue_walls: int,
    ) -> None:
        if red_start == blue_start:
            raise ValueError("red_start and blue_start cannot be the same square.")
        if red_walls < 0 or blue_walls < 0:
            raise ValueError("Initial wall counts must be non-negative.")

    @staticmethod
    def encode_move(move: Move) -> int:
        return encode_move(move)

    @staticmethod
    def decode_action(action: int) -> Move:
        return decode_action(action)

    @staticmethod
    def move_to_dict(move: Move) -> Dict[str, Any]:
        if move[0] == "move":
            _, direction = move
            direction = coerce_move_direction(direction)
            row_delta, col_delta = MOVE_DIRECTION_DELTAS[direction]
            return {
                "type": "move",
                "direction": direction.name,
                "row_delta": row_delta,
                "col_delta": col_delta,
            }

        _, orientation, row, col = move
        return {
            "type": "wall",
            "orientation": coerce_orientation(orientation).name,
            "row": int(row),
            "col": int(col),
        }

    def _handle_invalid_action(
        self, action: Any, acting_player: Player, message: str
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self.invalid_action_mode == "raise":
            raise ValueError(message)

        self.steps += 1
        self.state.winner = acting_player.opposite()
        self.terminated = True

        info = self._get_info()
        info.update(
            {
                "acting_player": acting_player.name,
                "last_action": action,
                "last_move": None,
                "invalid_action": True,
                "invalid_action_message": message,
            }
        )
        return self._get_observation(), self.invalid_action_penalty, True, False, info

    def _get_observation(self) -> Dict[str, Any]:
        board = self._zeros((4, BOARD_SIZE, BOARD_SIZE))

        red_row, red_col = self.state.pawns[Player.RED]
        blue_row, blue_col = self.state.pawns[Player.BLUE]
        board[0][red_row][red_col] = 1.0
        board[1][blue_row][blue_col] = 1.0

        for orientation, row, col in self.state.walls:
            plane = 2 if orientation == WallOrientation.HORIZONTAL else 3
            board[plane][row][col] = 1.0

        red_distance = self.state.greedy_path_length(Player.RED)
        blue_distance = self.state.greedy_path_length(Player.BLUE)
        features_data = [
            float(self.state.current_player.value),
            self._normalize_wall_count(Player.RED),
            self._normalize_wall_count(Player.BLUE),
            self._normalize_distance(red_distance),
            self._normalize_distance(blue_distance),
        ]

        if np is not None:
            features = np.array(features_data, dtype=np.float32)
        else:
            features = features_data

        return {
            "board": board,
            "features": features,
            "action_mask": self.legal_action_mask(),
        }

    def _get_info(self) -> Dict[str, Any]:
        red_distance = self.state.greedy_path_length(Player.RED)
        blue_distance = self.state.greedy_path_length(Player.BLUE)
        lead = self._terminal_lead(red_distance, blue_distance)
        return {
            "current_player": self.state.current_player.name,
            "winner": self.state.winner.name if self.state.winner else None,
            "steps": self.steps,
            "lead": lead,
            "red_position": self.state.pawns[Player.RED],
            "blue_position": self.state.pawns[Player.BLUE],
            "red_pawn_moves": self.pawn_moves[Player.RED],
            "blue_pawn_moves": self.pawn_moves[Player.BLUE],
            "red_moves_to_win": red_distance,
            "blue_moves_to_win": blue_distance,
            "n_moves": {
                "RED": self.pawn_moves[Player.RED],
                "BLUE": self.pawn_moves[Player.BLUE],
            },
            "N_moves": {
                "RED": self.pawn_moves[Player.RED],
                "BLUE": self.pawn_moves[Player.BLUE],
            },
            "moves_to_win": {
                "RED": red_distance,
                "BLUE": blue_distance,
            },
            "red_walls_left": self.state.walls_left[Player.RED],
            "blue_walls_left": self.state.walls_left[Player.BLUE],
            "red_initial_walls": self.state.initial_walls[Player.RED],
            "blue_initial_walls": self.state.initial_walls[Player.BLUE],
            "red_greedy_path_length": red_distance,
            "blue_greedy_path_length": blue_distance,
            "legal_actions": self.legal_actions(),
            "action_mask": self.legal_action_mask(),
        }

    def _render_ansi(self) -> str:
        lines = [
            f"Current: {self.state.current_player.name}",
            f"Walls: RED={self.state.walls_left[Player.RED]} BLUE={self.state.walls_left[Player.BLUE]}",
        ]
        if self.state.winner:
            lines.append(f"Winner: {self.state.winner.name}")

        for row in range(BOARD_SIZE):
            cells = []
            for col in range(BOARD_SIZE):
                cell = "."
                if self.state.pawns[Player.RED] == (row, col):
                    cell = "R"
                elif self.state.pawns[Player.BLUE] == (row, col):
                    cell = "B"
                cells.append(cell)
                if col < BOARD_SIZE - 1:
                    cells.append("|" if self.state.is_blocked(row, col, row, col + 1) else " ")
            lines.append(" ".join(cells))

            if row < BOARD_SIZE - 1:
                gaps = []
                for col in range(BOARD_SIZE):
                    gaps.append("-" if self.state.is_blocked(row, col, row + 1, col) else " ")
                    if col < BOARD_SIZE - 1:
                        gaps.append("+")
                lines.append(" ".join(gaps))

        return "\n".join(lines)

    @staticmethod
    def _normalize_distance(distance: Optional[int]) -> float:
        if distance is None:
            return 1.0
        return min(float(distance) / float(BOARD_SIZE * BOARD_SIZE), 1.0)

    def _normalize_wall_count(self, player: Player) -> float:
        initial_count = self.state.initial_walls[player]
        if initial_count <= 0:
            return 0.0
        return self.state.walls_left[player] / initial_count

    def _terminal_lead(
        self,
        red_moves_to_win: Optional[int],
        blue_moves_to_win: Optional[int],
    ) -> Optional[int]:
        if self.state.winner is None:
            return None
        if red_moves_to_win is None or blue_moves_to_win is None:
            return None
        return red_moves_to_win - blue_moves_to_win

    @staticmethod
    def _zeros(shape: Tuple[int, int, int]) -> Any:
        if np is not None:
            return np.zeros(shape, dtype=np.float32)

        planes, rows, cols = shape
        return [
            [[0.0 for _ in range(cols)] for _ in range(rows)]
            for _ in range(planes)
        ]


def _demo() -> None:
    env = BarricadeEnv(render_mode="ansi")
    _, info = env.reset(seed=1)
    action = env.sample_legal_action()
    if action is not None:
        _, reward, terminated, truncated, info = env.step(action)
        print(
            f"action={action} reward={reward} terminated={terminated} "
            f"truncated={truncated} next_player={info['current_player']}"
        )
    print(env.render())


if __name__ == "__main__":
    _demo()
