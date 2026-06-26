"""
Headless Barricade reinforcement-learning environment.

This module mirrors the rules used by ``barricade_pygame.py`` without importing
Pygame. It exposes a Gymnasium-style API:

    observation, info = env.reset()
    observation, reward, terminated, truncated, info = env.step(action)

The action space is fixed at 136 discrete actions:
    0        move pawn one square up
    1        move pawn one square down
    2        move pawn one square left
    3        move pawn one square right
    4..67    place a horizontal wall at an 8x8 wall anchor
    68..131  place a vertical wall at an 8x8 wall anchor
    132..135 diagonal side-hop pawn moves when a straight jump is blocked
"""

from __future__ import annotations

from enum import Enum, IntEnum
import heapq
import random
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

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


# Upper bound on the per-game path/route memo caches. These dicts are shared by
# reference across every cloned ``BarricadeState`` in a single game's MCTS tree
# (see ``BarricadeState.copy``) and were previously insert-only, so a long game
# with a reused search tree could accumulate millions of entries (keyed by a
# big-int wall mask) and drive the host into swap. Bounding them keeps the peak
# resident set flat; evicting is always safe because every value is recomputable.
PATH_CACHE_LIMIT = 50_000
REPETITION_DRAW_COUNT = 3


def _bounded_cache_put(cache: Dict[Any, Any], key: Any, value: Any, limit: int) -> Any:
    """Insert ``key -> value`` into ``cache``, evicting oldest entries past ``limit``.

    Returns ``value`` so call sites can ``return _bounded_cache_put(...)``.
    """
    if len(cache) >= limit:
        # Evict a batch of the oldest entries (dicts preserve insertion order) so
        # the amortized eviction cost stays negligible relative to a path search.
        for _ in range(max(1, limit // 8)):
            try:
                cache.pop(next(iter(cache)))
            except StopIteration:
                break
    cache[key] = value
    return value


BOARD_SIZE = 9
WALL_BOARD_SIZE = BOARD_SIZE - 1
DEFAULT_WALLS_PER_PLAYER = 10
DEFAULT_RED_START = (0, 4)
DEFAULT_BLUE_START = (8, 4)

MOVE_ACTIONS = 4
WALL_ACTIONS_PER_ORIENTATION = WALL_BOARD_SIZE * WALL_BOARD_SIZE
HORIZONTAL_WALL_OFFSET = MOVE_ACTIONS
VERTICAL_WALL_OFFSET = HORIZONTAL_WALL_OFFSET + WALL_ACTIONS_PER_ORIENTATION
DIAGONAL_HOP_ACTIONS = 4
DIAGONAL_HOP_OFFSET = MOVE_ACTIONS + WALL_ACTIONS_PER_ORIENTATION * 2
ACTION_SIZE = DIAGONAL_HOP_OFFSET + DIAGONAL_HOP_ACTIONS

DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def wall_board_size_for_board_size(board_size: int) -> int:
    board_size = int(board_size)
    if board_size < 2:
        raise ValueError("board_size must be at least 2.")
    return board_size - 1


def action_size_for_board_size(board_size: int) -> int:
    wall_board_size = wall_board_size_for_board_size(board_size)
    return MOVE_ACTIONS + wall_board_size * wall_board_size * 2 + DIAGONAL_HOP_ACTIONS


def diagonal_hop_offset_for_board_size(board_size: int) -> int:
    wall_board_size = wall_board_size_for_board_size(board_size)
    return MOVE_ACTIONS + wall_board_size * wall_board_size * 2


class Player(IntEnum):
    RED = 0
    BLUE = 1

    def opposite(self) -> "Player":
        return Player.BLUE if self == Player.RED else Player.RED


class WallOrientation(IntEnum):
    HORIZONTAL = 0
    VERTICAL = 1


class MoveDirection(IntEnum):
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
DIAGONAL_HOP_DELTAS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

ALL_WALL_PLACEMENTS = tuple(
    (orient, r, c)
    for orient in (WallOrientation.HORIZONTAL, WallOrientation.VERTICAL)
    for r in range(WALL_BOARD_SIZE)
    for c in range(WALL_BOARD_SIZE)
)
_CELL_ADJACENCY_CACHE: Dict[int, Tuple[Tuple[Any, ...], ...]] = {}
_PATH_GRAPH_CACHE: Dict[int, "PathGraph"] = {}


class PathGraph(NamedTuple):
    adjacency: Tuple[Tuple[Tuple[int, int], ...], ...]
    horizontal_wall_masks: Tuple[int, ...]
    vertical_wall_masks: Tuple[int, ...]
    red_heuristics: Tuple[int, ...]
    blue_heuristics: Tuple[int, ...]
    red_goal_mask: int
    blue_goal_mask: int
    index_to_cell: Tuple[Tuple[int, int], ...]


def all_wall_placements_for_board_size(board_size: int) -> Tuple[WallPlacement, ...]:
    wall_board_size = wall_board_size_for_board_size(board_size)
    return tuple(
        (orient, row, col)
        for orient in (WallOrientation.HORIZONTAL, WallOrientation.VERTICAL)
        for row in range(wall_board_size)
        for col in range(wall_board_size)
    )


def cell_adjacency_for_board_size(board_size: int) -> Tuple[Tuple[Any, ...], ...]:
    board_size = int(board_size)
    cached = _CELL_ADJACENCY_CACHE.get(board_size)
    if cached is not None:
        return cached

    adjacency = []
    for row in range(board_size):
        for col in range(board_size):
            cell_index = row * board_size + col
            neighbors = []
            for row_delta, col_delta in DIRECTIONS:
                next_row = row + row_delta
                next_col = col + col_delta
                if not (0 <= next_row < board_size and 0 <= next_col < board_size):
                    continue

                if row_delta:
                    row_min = row if row < next_row else next_row
                    neighbors.append(
                        (
                            (next_row, next_col),
                            True,
                            (row_min, col),
                            (row_min, col - 1),
                        )
                    )
                else:
                    col_min = col if col < next_col else next_col
                    neighbors.append(
                        (
                            (next_row, next_col),
                            False,
                            (row, col_min),
                            (row - 1, col_min),
                        )
                    )
            adjacency.append(tuple(neighbors))

    cached = tuple(adjacency)
    _CELL_ADJACENCY_CACHE[board_size] = cached
    return cached


def path_graph_for_board_size(board_size: int) -> PathGraph:
    board_size = int(board_size)
    cached = _PATH_GRAPH_CACHE.get(board_size)
    if cached is not None:
        return cached

    wall_board_size = wall_board_size_for_board_size(board_size)
    index_to_cell = tuple(
        (row, col)
        for row in range(board_size)
        for col in range(board_size)
    )

    edge_bits: Dict[Tuple[int, int], int] = {}
    next_bit = 1
    adjacency: List[List[Tuple[int, int]]] = [
        [] for _ in range(board_size * board_size)
    ]

    for row in range(board_size):
        for col in range(board_size):
            cell_index = row * board_size + col
            for row_delta, col_delta in DIRECTIONS:
                next_row = row + row_delta
                next_col = col + col_delta
                if not (0 <= next_row < board_size and 0 <= next_col < board_size):
                    continue
                next_index = next_row * board_size + next_col
                edge_key = (
                    (cell_index, next_index)
                    if cell_index < next_index
                    else (next_index, cell_index)
                )
                edge_bit = edge_bits.get(edge_key)
                if edge_bit is None:
                    edge_bit = next_bit
                    edge_bits[edge_key] = edge_bit
                    next_bit <<= 1
                adjacency[cell_index].append((next_index, edge_bit))

    def edge_between(row1: int, col1: int, row2: int, col2: int) -> int:
        first = row1 * board_size + col1
        second = row2 * board_size + col2
        edge_key = (first, second) if first < second else (second, first)
        return edge_bits[edge_key]

    horizontal_wall_masks: List[int] = []
    vertical_wall_masks: List[int] = []
    for row in range(wall_board_size):
        for col in range(wall_board_size):
            horizontal_wall_masks.append(
                edge_between(row, col, row + 1, col)
                | edge_between(row, col + 1, row + 1, col + 1)
            )
            vertical_wall_masks.append(
                edge_between(row, col, row, col + 1)
                | edge_between(row + 1, col, row + 1, col + 1)
            )

    red_target = board_size - 1
    blue_target = 0
    red_goal_mask = 0
    blue_goal_mask = 0
    red_heuristics: List[int] = []
    blue_heuristics: List[int] = []
    for index, (row, _) in enumerate(index_to_cell):
        red_heuristics.append(abs(row - red_target))
        blue_heuristics.append(abs(row - blue_target))
        if row == red_target:
            red_goal_mask |= 1 << index
        if row == blue_target:
            blue_goal_mask |= 1 << index

    cached = PathGraph(
        adjacency=tuple(tuple(neighbors) for neighbors in adjacency),
        horizontal_wall_masks=tuple(horizontal_wall_masks),
        vertical_wall_masks=tuple(vertical_wall_masks),
        red_heuristics=tuple(red_heuristics),
        blue_heuristics=tuple(blue_heuristics),
        red_goal_mask=red_goal_mask,
        blue_goal_mask=blue_goal_mask,
        index_to_cell=index_to_cell,
    )
    _PATH_GRAPH_CACHE[board_size] = cached
    return cached


def coerce_player(player: Player | str | int) -> Player:
    if isinstance(player, Player):
        return player
    if isinstance(player, str):
        return Player[player.upper()]
    if isinstance(player, int):
        return Player(player)
    # Handle cases where it might be a different Player enum from another module
    if hasattr(player, 'value'):
        return Player(player.value)
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
    return coerce_position_for_board_size(position, name, BOARD_SIZE)


def coerce_position_for_board_size(
    position: Sequence[int],
    name: str,
    board_size: int,
) -> Tuple[int, int]:
    board_size = int(board_size)
    if len(position) != 2:
        raise ValueError(f"{name} must contain exactly two values: (row, col).")
    row, col = int(position[0]), int(position[1])
    if not (0 <= row < board_size and 0 <= col < board_size):
        raise ValueError(f"{name} must be on the {board_size}x{board_size} board.")
    return row, col


def coerce_wall_count(count: int, name: str) -> int:
    count = int(count)
    if count < 0:
        raise ValueError(f"{name} must be non-negative.")
    return count


def encode_move(move: Move) -> int:
    """Encode a move tuple into the fixed discrete action id."""
    return encode_move_for_board_size(move, BOARD_SIZE)


def encode_move_for_board_size(move: Move, board_size: int) -> int:
    """Encode a move tuple for a specific board size."""
    wall_board_size = wall_board_size_for_board_size(board_size)
    move_type = move[0]
    if move_type == "move":
        if len(move) != 2:
            raise ValueError(
                "Move actions must be (\"move\", direction); use state.encode_move "
                "for target-cell pawn moves."
            )
        _, direction = move
        return coerce_move_direction(direction).value

    if move_type == "move_diagonal":
        _, row_delta, col_delta = move
        delta = (int(row_delta), int(col_delta))
        if delta not in DIAGONAL_HOP_DELTAS:
            raise ValueError(f"Invalid diagonal hop delta: {delta}")
        return diagonal_hop_offset_for_board_size(board_size) + DIAGONAL_HOP_DELTAS.index(delta)

    if move_type == "wall":
        _, orientation, row, col = move
        orientation = coerce_orientation(orientation)
        row = int(row)
        col = int(col)
        if not (0 <= row < wall_board_size and 0 <= col < wall_board_size):
            raise ValueError(f"Wall anchor out of bounds: {(row, col)}")
        offset = (
            HORIZONTAL_WALL_OFFSET
            if orientation == WallOrientation.HORIZONTAL
            else HORIZONTAL_WALL_OFFSET + wall_board_size * wall_board_size
        )
        return offset + row * wall_board_size + col

    raise ValueError(f"Unknown move type: {move_type!r}")


def decode_action(action: int) -> Move:
    """Decode a discrete action id into a game move tuple."""
    return decode_action_for_board_size(action, BOARD_SIZE)


def decode_action_for_board_size(action: int, board_size: int) -> Move:
    """Decode a discrete action id for a specific board size."""
    wall_board_size = wall_board_size_for_board_size(board_size)
    horizontal_wall_offset = MOVE_ACTIONS
    vertical_wall_offset = horizontal_wall_offset + wall_board_size * wall_board_size
    action_size = action_size_for_board_size(board_size)
    action = int(action)
    if 0 <= action < MOVE_ACTIONS:
        return ("move", MoveDirection(action))

    if horizontal_wall_offset <= action < vertical_wall_offset:
        index = action - horizontal_wall_offset
        return (
            "wall",
            WallOrientation.HORIZONTAL,
            index // wall_board_size,
            index % wall_board_size,
        )

    if vertical_wall_offset <= action < action_size:
        index = action - vertical_wall_offset
        wall_action_count = wall_board_size * wall_board_size
        if index >= wall_action_count:
            diagonal_index = index - wall_action_count
            row_delta, col_delta = DIAGONAL_HOP_DELTAS[diagonal_index]
            return ("move_diagonal", row_delta, col_delta)
        return (
            "wall",
            WallOrientation.VERTICAL,
            index // wall_board_size,
            index % wall_board_size,
        )

    raise ValueError(f"Action must be in [0, {action_size - 1}], got {action}")


class BarricadeState:
    """Pure game state and rule engine for Barricade."""

    def __init__(
        self,
        *,
        red_start: Sequence[int] = DEFAULT_RED_START,
        blue_start: Sequence[int] = DEFAULT_BLUE_START,
        red_walls: int = DEFAULT_WALLS_PER_PLAYER,
        blue_walls: int = DEFAULT_WALLS_PER_PLAYER,
        starting_player: Player | str | int = Player.RED,
        board_size: int = BOARD_SIZE,
    ) -> None:
        board_size = int(board_size)
        wall_board_size = wall_board_size_for_board_size(board_size)
        red_start = coerce_position_for_board_size(red_start, "red_start", board_size)
        blue_start = coerce_position_for_board_size(blue_start, "blue_start", board_size)
        if red_start == blue_start:
            raise ValueError("red_start and blue_start cannot be the same square.")

        red_walls = coerce_wall_count(red_walls, "red_walls")
        blue_walls = coerce_wall_count(blue_walls, "blue_walls")
        starting_player = coerce_player(starting_player)

        self.board_size = board_size
        self.wall_board_size = wall_board_size
        self.action_size = action_size_for_board_size(board_size)
        self.all_wall_placements = all_wall_placements_for_board_size(board_size)
        self._cell_adjacency = cell_adjacency_for_board_size(board_size)
        self._path_graph = path_graph_for_board_size(board_size)
        self._blocked_edge_mask = 0
        self._walls_frozenset: frozenset[WallPlacement] = frozenset()
        self._blocked_edge_mask_cache: int = 0
        self._blocked_edge_mask_cache_key: frozenset[WallPlacement] = frozenset()
        self._wall_lookup_cache_key: frozenset[WallPlacement] = frozenset()
        self._horizontal_walls_cache: Set[Tuple[int, int]] = set()
        self._vertical_walls_cache: Set[Tuple[int, int]] = set()
        self.pawns = {Player.RED: red_start, Player.BLUE: blue_start}
        self.walls: Set[WallPlacement] = set()
        self.current_player = starting_player
        self.winner: Optional[Player] = None
        self.is_draw = False
        self.draw_reason: Optional[str] = None
        self.initial_walls = {Player.RED: red_walls, Player.BLUE: blue_walls}
        self.walls_left = {Player.RED: red_walls, Player.BLUE: blue_walls}

        self._valid_moves_cache_key: Optional[Tuple[Any, ...]] = None
        self._valid_moves_cache: Optional[Tuple[Move, ...]] = None
        self._valid_action_moves_cache: Optional[Tuple[Tuple[int, Move], ...]] = None
        self._path_cache: Dict[Tuple[Any, ...], Optional[int]] = {}
        self._route_cache: Dict[Tuple[Any, ...], Optional[Tuple[Tuple[int, int], ...]]] = {}
        self._repetition_counts: Dict[Tuple[Any, ...], int] = {
            self.repetition_key(): 1
        }

    def copy(self) -> "BarricadeState":
        new_state = BarricadeState.__new__(BarricadeState)
        new_state.board_size = self.board_size
        new_state.wall_board_size = self.wall_board_size
        new_state.action_size = self.action_size
        new_state.all_wall_placements = self.all_wall_placements
        new_state._cell_adjacency = self._cell_adjacency
        new_state._path_graph = self._path_graph
        new_state._blocked_edge_mask = self._blocked_edge_mask
        new_state._walls_frozenset = self._walls_frozenset
        new_state._blocked_edge_mask_cache = self._blocked_edge_mask_cache
        new_state._blocked_edge_mask_cache_key = self._blocked_edge_mask_cache_key
        new_state._wall_lookup_cache_key = self._wall_lookup_cache_key
        new_state._horizontal_walls_cache = set(self._horizontal_walls_cache)
        new_state._vertical_walls_cache = set(self._vertical_walls_cache)
        new_state.pawns = dict(self.pawns)
        new_state.walls = set(self.walls)
        new_state.current_player = self.current_player
        new_state.winner = self.winner
        new_state.is_draw = self.is_draw
        new_state.draw_reason = self.draw_reason
        new_state.initial_walls = dict(self.initial_walls)
        new_state.walls_left = dict(self.walls_left)
        new_state._valid_moves_cache_key = None
        new_state._valid_moves_cache = None
        new_state._valid_action_moves_cache = None
        new_state._path_cache = self._path_cache
        new_state._route_cache = self._route_cache
        new_state._repetition_counts = dict(self._repetition_counts)
        return new_state

    def state_cache_key(self) -> Tuple[Any, ...]:
        return (
            self.board_size,
            self.pawns[Player.RED],
            self.pawns[Player.BLUE],
            self.current_player,
            self.winner,
            self.is_draw,
            self.walls_left[Player.RED],
            self.walls_left[Player.BLUE],
            self._walls_frozenset,
        )

    def repetition_key(self) -> Tuple[Any, ...]:
        return (
            self.board_size,
            self.pawns[Player.RED],
            self.pawns[Player.BLUE],
            self.current_player,
            self.walls_left[Player.RED],
            self.walls_left[Player.BLUE],
            self._walls_frozenset,
        )

    def is_terminal(self) -> bool:
        return self.winner is not None or self.is_draw

    def get_valid_moves(self) -> List[Move]:
        if self.is_terminal():
            return []

        key = self.state_cache_key()
        if key == self._valid_moves_cache_key and self._valid_moves_cache is not None:
            return list(self._valid_moves_cache)

        action_moves = self._compute_valid_action_moves()
        self._cache_valid_action_moves(key, action_moves)
        return [move for _, move in action_moves]

    def legal_moves(self) -> List[Move]:
        """Alias for ``get_valid_moves`` used by action-centric callers."""
        return self.get_valid_moves()

    def legal_actions(self) -> List[int]:
        """Return legal moves encoded in the fixed discrete action space."""
        return [action for action, _ in self.legal_action_moves()]

    def legal_action_moves(self) -> List[Tuple[int, Move]]:
        """Return legal actions paired with their rule-engine move tuples."""
        if self.is_terminal():
            return []

        key = self.state_cache_key()
        if (
            key == self._valid_moves_cache_key
            and self._valid_action_moves_cache is not None
        ):
            return self._valid_action_moves_cache

        action_moves = self._compute_valid_action_moves()
        self._cache_valid_action_moves(key, action_moves)
        return action_moves

    def _cache_valid_action_moves(
        self,
        key: Tuple[Any, ...],
        action_moves: List[Tuple[int, Move]],
    ) -> None:
        self._valid_moves_cache_key = key
        self._valid_action_moves_cache = tuple(action_moves)
        self._valid_moves_cache = tuple(move for _, move in action_moves)

    def _compute_valid_action_moves(self) -> List[Tuple[int, Move]]:
        action_moves = self._get_pawn_action_moves()
        action_moves.extend(self._get_valid_wall_action_moves())
        action_moves.sort(key=lambda action_move: action_move[0])
        return action_moves

    def action_mask(self) -> List[int]:
        """Return a 0/1 legal-action mask as a plain Python list."""
        mask = [0] * self.action_size
        for action, _ in self.legal_action_moves():
            mask[action] = 1
        return mask

    def action_mask_numpy(self) -> Any:
        """Return a 0/1 legal-action mask as a NumPy array."""
        if np is None:
            return self.action_mask()
        mask = np.zeros(self.action_size, dtype=np.int8)
        for action, _ in self.legal_action_moves():
            mask[action] = 1
        return mask

    def get_pawn_moves(self) -> List[Move]:
        return [move for _, move in self._get_pawn_action_moves()]

    def _get_pawn_action_moves(self) -> List[Tuple[int, Move]]:
        if self.is_terminal():
            return []

        moves: List[Tuple[int, Move]] = []
        pawn_row, pawn_col = self.pawns[self.current_player]
        opponent_position = self.pawns[self.current_player.opposite()]

        for direction, (row_delta, col_delta) in MOVE_DIRECTION_DELTAS.items():
            next_row = pawn_row + row_delta
            next_col = pawn_col + col_delta

            if not (
                0 <= next_row < self.board_size
                and 0 <= next_col < self.board_size
            ):
                continue
            if self.is_blocked(pawn_row, pawn_col, next_row, next_col):
                continue
            if (next_row, next_col) == opponent_position:
                jump_row = next_row + row_delta
                jump_col = next_col + col_delta
                can_jump_straight = (
                    0 <= jump_row < self.board_size
                    and 0 <= jump_col < self.board_size
                    and not self.is_blocked(next_row, next_col, jump_row, jump_col)
                )
                if can_jump_straight:
                    moves.append((direction.value, ("move_to", jump_row, jump_col)))
                    continue

                side_deltas = (
                    ((0, -1), (0, 1))
                    if row_delta
                    else ((-1, 0), (1, 0))
                )
                for side_row_delta, side_col_delta in side_deltas:
                    side_row = next_row + side_row_delta
                    side_col = next_col + side_col_delta
                    if not (
                        0 <= side_row < self.board_size
                        and 0 <= side_col < self.board_size
                    ):
                        continue
                    if self.is_blocked(next_row, next_col, side_row, side_col):
                        continue

                    target_delta = (
                        side_row - pawn_row,
                        side_col - pawn_col,
                    )
                    action = self._diagonal_hop_action_for_delta(target_delta)
                    moves.append((action, ("move_to", side_row, side_col)))
                continue

            moves.append((direction.value, ("move_to", next_row, next_col)))

        return moves

    def _diagonal_hop_action_for_delta(self, delta: Tuple[int, int]) -> int:
        if delta not in DIAGONAL_HOP_DELTAS:
            raise ValueError(f"Invalid diagonal hop delta: {delta}")
        return (
            diagonal_hop_offset_for_board_size(self.board_size)
            + DIAGONAL_HOP_DELTAS.index(delta)
        )

    def get_valid_wall_moves(
        self, candidates: Optional[Iterable[WallPlacement]] = None
    ) -> List[Move]:
        return [move for _, move in self._get_valid_wall_action_moves(candidates)]

    def _get_valid_wall_action_moves(
        self, candidates: Optional[Iterable[WallPlacement]] = None
    ) -> List[Tuple[int, Move]]:
        if self.is_terminal():
            return []
        if self.walls_left[self.current_player] <= 0:
            return []

        wall_moves: List[Tuple[int, Move]] = []
        placements = self.all_wall_placements if candidates is None else candidates
        horizontal_walls, vertical_walls = self._wall_lookup(self.walls)
        base_blocked_edge_mask = self._current_blocked_edge_mask()
        red_route = self.greedy_path_cells(Player.RED)
        blue_route = self.greedy_path_cells(Player.BLUE)
        red_route_blockers = (
            self._path_blocking_walls(red_route) if red_route is not None else None
        )
        blue_route_blockers = (
            self._path_blocking_walls(blue_route) if blue_route is not None else None
        )

        # Split the enum-keyed blocker sets into orientation-specific (row, col)
        # sets once, so the per-placement membership test stays int-only.
        # ``None`` means "every wall needs a path search" (route unavailable).
        def _split(blockers):
            if blockers is None:
                return None, None
            h: Set[Tuple[int, int]] = set()
            v: Set[Tuple[int, int]] = set()
            for b_orientation, b_row, b_col in blockers:
                if b_orientation == WallOrientation.HORIZONTAL:
                    h.add((b_row, b_col))
                else:
                    v.add((b_row, b_col))
            return h, v

        red_h_blockers, red_v_blockers = _split(red_route_blockers)
        blue_h_blockers, blue_v_blockers = _split(blue_route_blockers)
        red_route_unavailable = red_route_blockers is None
        blue_route_unavailable = blue_route_blockers is None

        # Hoist frequently used attributes/locals out of the per-placement loop.
        wall_board_size = self.wall_board_size
        path_graph = self._path_graph
        horizontal_masks = path_graph.horizontal_wall_masks
        vertical_masks = path_graph.vertical_wall_masks
        has_path = self._has_path_with_mask
        red = Player.RED
        blue = Player.BLUE
        horizontal_enum = WallOrientation.HORIZONTAL
        vertical_offset = HORIZONTAL_WALL_OFFSET + wall_board_size * wall_board_size

        for orientation, row, col in placements:
            if row < 0 or row >= wall_board_size or col < 0 or col >= wall_board_size:
                continue

            is_horizontal = orientation == horizontal_enum

            # Inline the shape-availability test (no overlap with other walls).
            if is_horizontal:
                if (
                    (row, col) in horizontal_walls
                    or (row, col - 1) in horizontal_walls
                    or (row, col + 1) in horizontal_walls
                    or (row, col) in vertical_walls
                ):
                    continue
            else:
                if (
                    (row, col) in vertical_walls
                    or (row - 1, col) in vertical_walls
                    or (row + 1, col) in vertical_walls
                    or (row, col) in horizontal_walls
                ):
                    continue

            # Only run the (expensive) BFS path check when the candidate wall
            # lies on a player's current greedy route.
            cell = (row, col)
            if is_horizontal:
                red_needs_search = red_route_unavailable or cell in red_h_blockers
                blue_needs_search = blue_route_unavailable or cell in blue_h_blockers
            else:
                red_needs_search = red_route_unavailable or cell in red_v_blockers
                blue_needs_search = blue_route_unavailable or cell in blue_v_blockers

            if red_needs_search or blue_needs_search:
                index = row * wall_board_size + col
                w_mask = horizontal_masks[index] if is_horizontal else vertical_masks[index]
                blocked_edge_mask = base_blocked_edge_mask | w_mask
                if red_needs_search and not has_path(red, blocked_edge_mask):
                    continue
                if blue_needs_search and not has_path(blue, blocked_edge_mask):
                    continue

            offset = HORIZONTAL_WALL_OFFSET if is_horizontal else vertical_offset
            action = offset + row * wall_board_size + col
            wall_moves.append((action, ("wall", orientation, row, col)))

        return wall_moves

    def is_blocked(self, row1: int, col1: int, row2: int, col2: int) -> bool:
        """Return True when a direct adjacent step is blocked by a wall."""
        horizontal = WallOrientation.HORIZONTAL
        vertical = WallOrientation.VERTICAL
        walls = self.walls
        wall_board_size = self.wall_board_size
        if row1 == row2:
            col_min = col1 if col1 < col2 else col2
            return (
                0 <= row1 < wall_board_size
                and (vertical, row1, col_min) in walls
            ) or (
                0 <= row1 - 1 < wall_board_size
                and (vertical, row1 - 1, col_min) in walls
            )
        elif col1 == col2:
            row_min = row1 if row1 < row2 else row2
            return (
                0 <= col1 < wall_board_size
                and (horizontal, row_min, col1) in walls
            ) or (
                0 <= col1 - 1 < wall_board_size
                and (horizontal, row_min, col1 - 1) in walls
            )
        return False

    def is_valid_wall_placement(
        self, orientation: WallOrientation | str | int, row: int, col: int
    ) -> bool:
        return self._is_valid_wall_placement(orientation, row, col, None, None)

    def _is_valid_wall_placement(
        self,
        orientation: WallOrientation | str | int,
        row: int,
        col: int,
        red_route_blockers: Optional[Set[WallPlacement]],
        blue_route_blockers: Optional[Set[WallPlacement]],
        horizontal_walls: Optional[Set[Tuple[int, int]]] = None,
        vertical_walls: Optional[Set[Tuple[int, int]]] = None,
        base_blocked_edge_mask: Optional[int] = None,
    ) -> bool:
        # Avoid calling coerce_orientation if already an enum
        if not isinstance(orientation, WallOrientation):
            orientation = coerce_orientation(orientation)
            
        if horizontal_walls is None or vertical_walls is None:
            horizontal_walls, vertical_walls = self._wall_lookup(self.walls)
        
        # Inline availability check for speed
        if row < 0 or row >= self.wall_board_size or col < 0 or col >= self.wall_board_size:
            return False

        if orientation == WallOrientation.HORIZONTAL:
            if (row, col) in horizontal_walls or (row, col - 1) in horizontal_walls or \
               (row, col + 1) in horizontal_walls or (row, col) in vertical_walls:
                return False
        else:
            if (row, col) in vertical_walls or (row - 1, col) in vertical_walls or \
               (row + 1, col) in vertical_walls or (row, col) in horizontal_walls:
                return False

        wall = (orientation, row, col)
        red_needs_search = red_route_blockers is None or wall in red_route_blockers
        blue_needs_search = blue_route_blockers is None or wall in blue_route_blockers
        if not red_needs_search and not blue_needs_search:
            return True

        if base_blocked_edge_mask is None:
            base_blocked_edge_mask = self._current_blocked_edge_mask()
            
        # Inline _wall_edge_mask
        index = int(row) * self.wall_board_size + int(col)
        if orientation == WallOrientation.HORIZONTAL:
            w_mask = self._path_graph.horizontal_wall_masks[index]
        else:
            w_mask = self._path_graph.vertical_wall_masks[index]
            
        blocked_edge_mask = base_blocked_edge_mask | w_mask
        
        if red_needs_search:
            if not self._has_path_with_mask(Player.RED, blocked_edge_mask):
                return False
        
        if blue_needs_search:
            if not self._has_path_with_mask(Player.BLUE, blocked_edge_mask):
                return False

        return True

    def is_wall_shape_available(
        self, orientation: WallOrientation | str | int, row: int, col: int
    ) -> bool:
        orientation = coerce_orientation(orientation)
        horizontal_walls, vertical_walls = self._wall_lookup(self.walls)
        return self._is_wall_shape_available_from_sets(
            orientation,
            row,
            col,
            horizontal_walls,
            vertical_walls,
        )

    def _is_wall_shape_available_from_sets(
        self,
        orientation: WallOrientation,
        row: int,
        col: int,
        horizontal_walls: Set[Tuple[int, int]],
        vertical_walls: Set[Tuple[int, int]],
    ) -> bool:
        if (
            row < 0
            or row >= self.wall_board_size
            or col < 0
            or col >= self.wall_board_size
        ):
            return False

        if orientation == WallOrientation.HORIZONTAL:
            if (row, col) in horizontal_walls:
                return False
            if (row, col - 1) in horizontal_walls:
                return False
            if (row, col + 1) in horizontal_walls:
                return False
            if (row, col) in vertical_walls:
                return False
        else:
            if (row, col) in vertical_walls:
                return False
            if (row - 1, col) in vertical_walls:
                return False
            if (row + 1, col) in vertical_walls:
                return False
            if (row, col) in horizontal_walls:
                return False

        return True

    def _path_blocking_walls(
        self,
        path: Tuple[Tuple[int, int], ...],
    ) -> Set[WallPlacement]:
        blockers: Set[WallPlacement] = set()
        wall_board_size = self.wall_board_size
        for index in range(len(path) - 1):
            row1, col1 = path[index]
            row2, col2 = path[index + 1]
            if col1 == col2:
                row_min = row1 if row1 < row2 else row2
                if 0 <= row_min < wall_board_size:
                    if 0 <= col1 < wall_board_size:
                        blockers.add((WallOrientation.HORIZONTAL, row_min, col1))
                    if 0 <= col1 - 1 < wall_board_size:
                        blockers.add((WallOrientation.HORIZONTAL, row_min, col1 - 1))
            elif row1 == row2:
                col_min = col1 if col1 < col2 else col2
                if 0 <= col_min < wall_board_size:
                    if 0 <= row1 < wall_board_size:
                        blockers.add((WallOrientation.VERTICAL, row1, col_min))
                    if 0 <= row1 - 1 < wall_board_size:
                        blockers.add((WallOrientation.VERTICAL, row1 - 1, col_min))
        return blockers

    def has_path(self, player: Player | str | int) -> bool:
        player = coerce_player(player)
        return self._has_path_with_mask(player, self._current_blocked_edge_mask())

    def _has_path_with_mask(self, player: Player, blocked_edge_mask: int) -> bool:
        start = self.pawns[player]
        graph = self._path_graph
        goal_mask = graph.red_goal_mask if player == Player.RED else graph.blue_goal_mask
        start_index = start[0] * self.board_size + start[1]

        # Pawn already on its goal row -> trivially reachable.
        if goal_mask & (1 << start_index):
            return True

        # No memoisation: each candidate wall produces a unique ``blocked_edge_mask``,
        # so reachability keys virtually never recur (~1% hit rate measured) and the
        # tuple-key construction + big-int hashing cost more than the rare recompute.
        #
        # Stack-based DFS. Reachability does not need BFS ordering, so popping from
        # a stack avoids the per-iteration ``len(queue)`` call of an index-cursor
        # queue. A ``bytearray`` visited-set avoids the big-integer allocation churn
        # of a bitmask, and we early-exit the instant a goal neighbour is found.
        adjacency = graph.adjacency
        visited = bytearray(len(adjacency))
        visited[start_index] = 1
        stack = [start_index]
        while stack:
            for next_index, edge_mask in adjacency[stack.pop()]:
                if visited[next_index] or (blocked_edge_mask & edge_mask):
                    continue
                if goal_mask & (1 << next_index):
                    return True
                visited[next_index] = 1
                stack.append(next_index)
        return False

    def shortest_path_length(self, player: Player | str | int) -> Optional[int]:
        player = coerce_player(player)
        blocked_edge_mask = self._current_blocked_edge_mask()
        key = ("shortest_distance_edges", player, self.pawns[player], blocked_edge_mask)
        if key in self._path_cache:
            return self._path_cache[key]

        start = self.pawns[player]
        start_index = start[0] * self.board_size + start[1]
        graph = self._path_graph
        goal_mask = graph.red_goal_mask if player == Player.RED else graph.blue_goal_mask
        
        # Optimized BFS with two-list queue for distance tracking
        current_level = [start_index]
        visited_mask = 1 << start_index
        distance = 0
        adjacency = graph.adjacency

        while current_level:
            next_level = []
            for cell_index in current_level:
                if goal_mask & (1 << cell_index):
                    return _bounded_cache_put(
                        self._path_cache, key, distance, PATH_CACHE_LIMIT
                    )

                for next_index, edge_mask in adjacency[cell_index]:
                    next_bit = 1 << next_index
                    if not (visited_mask & next_bit) and not (blocked_edge_mask & edge_mask):
                        visited_mask |= next_bit
                        next_level.append(next_index)
            current_level = next_level
            distance += 1

        return _bounded_cache_put(self._path_cache, key, None, PATH_CACHE_LIMIT)

    def greedy_path_length(self, player: Player | str | int) -> Optional[int]:
        return self.shortest_path_length(player)

    def greedy_path_cells(
        self, player: Player | str | int
    ) -> Optional[Tuple[Tuple[int, int], ...]]:
        player = coerce_player(player)
        blocked_edge_mask = self._current_blocked_edge_mask()
        key = ("greedy_route_edges", player, self.pawns[player], blocked_edge_mask)
        if key in self._route_cache:
            return self._route_cache[key]

        start = self.pawns[player]
        start_index = start[0] * self.board_size + start[1]
        graph = self._path_graph
        if player == Player.RED:
            goal_mask = graph.red_goal_mask
            heuristics = graph.red_heuristics
        else:
            goal_mask = graph.blue_goal_mask
            heuristics = graph.blue_heuristics

        frontier = [(heuristics[start_index], 0, start_index)]
        parent = [-1] * (self.board_size * self.board_size)
        parent[start_index] = start_index
        adjacency = graph.adjacency

        while frontier:
            _, distance, cell_index = heapq.heappop(frontier)
            if goal_mask & (1 << cell_index):
                path: List[Tuple[int, int]] = []
                node = cell_index
                while True:
                    path.append(graph.index_to_cell[node])
                    if node == parent[node]:
                        break
                    node = parent[node]
                result = tuple(reversed(path))
                return _bounded_cache_put(
                    self._route_cache, key, result, PATH_CACHE_LIMIT
                )

            for next_index, edge_mask in adjacency[cell_index]:
                if parent[next_index] == -1 and not (blocked_edge_mask & edge_mask):
                    parent[next_index] = cell_index
                    heapq.heappush(
                        frontier,
                        (
                            heuristics[next_index],
                            distance + 1,
                            next_index,
                        ),
                    )

        return _bounded_cache_put(self._route_cache, key, None, PATH_CACHE_LIMIT)

    def _path_heuristic(cell: Tuple[int, int], target_row: int) -> int:
        return abs(cell[0] - target_row)

    def _current_blocked_edge_mask(self) -> int:
        if self._blocked_edge_mask_cache_key == self._walls_frozenset:
            return self._blocked_edge_mask_cache

        blocked_edge_mask = 0
        for orientation, row, col in self._walls_frozenset:
            blocked_edge_mask |= self._wall_edge_mask(orientation, row, col)
        self._blocked_edge_mask_cache_key = self._walls_frozenset
        self._blocked_edge_mask_cache = blocked_edge_mask
        return blocked_edge_mask

    def _wall_edge_mask(
        self,
        orientation: WallOrientation,
        row: int,
        col: int,
    ) -> int:
        index = int(row) * self.wall_board_size + int(col)
        if orientation == WallOrientation.HORIZONTAL:
            return self._path_graph.horizontal_wall_masks[index]
        return self._path_graph.vertical_wall_masks[index]

    def _wall_lookup(
        self,
        walls: Iterable[WallPlacement],
    ) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        walls_frozenset = frozenset(walls)
        if self._wall_lookup_cache_key == walls_frozenset:
            return self._horizontal_walls_cache, self._vertical_walls_cache

        horizontal_walls: Set[Tuple[int, int]] = set()
        vertical_walls: Set[Tuple[int, int]] = set()
        for orientation, row, col in walls_frozenset:
            if orientation == WallOrientation.HORIZONTAL:
                horizontal_walls.add((row, col))
            else:
                vertical_walls.add((row, col))
        self._wall_lookup_cache_key = walls_frozenset
        self._horizontal_walls_cache = horizontal_walls
        self._vertical_walls_cache = vertical_walls
        return horizontal_walls, vertical_walls

    @staticmethod
    def _is_blocked_by_lookup(
        row1: int,
        col1: int,
        row2: int,
        col2: int,
        horizontal_walls: Set[Tuple[int, int]],
        vertical_walls: Set[Tuple[int, int]],
    ) -> bool:
        if row1 == row2:
            col_min = col1 if col1 < col2 else col2
            return (row1, col_min) in vertical_walls or (
                row1 - 1,
                col_min,
            ) in vertical_walls
        if col1 == col2:
            row_min = row1 if row1 < row2 else row2
            return (row_min, col1) in horizontal_walls or (
                row_min,
                col1 - 1,
            ) in horizontal_walls
        return False

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
            if row == new_state.board_size - 1 and new_state.current_player == Player.RED:
                new_state.winner = Player.RED
            elif row == 0 and new_state.current_player == Player.BLUE:
                new_state.winner = Player.BLUE
        elif move_type == "move_diagonal":
            _, row_delta, col_delta = move
            current_row, current_col = new_state.pawns[new_state.current_player]
            row = current_row + int(row_delta)
            col = current_col + int(col_delta)
            new_state.pawns[new_state.current_player] = (row, col)
            if row == new_state.board_size - 1 and new_state.current_player == Player.RED:
                new_state.winner = Player.RED
            elif row == 0 and new_state.current_player == Player.BLUE:
                new_state.winner = Player.BLUE
        elif move_type == "move_to":
            _, row, col = move
            row = int(row)
            col = int(col)
            new_state.pawns[new_state.current_player] = (row, col)
            if row == new_state.board_size - 1 and new_state.current_player == Player.RED:
                new_state.winner = Player.RED
            elif row == 0 and new_state.current_player == Player.BLUE:
                new_state.winner = Player.BLUE
        elif move_type == "wall":
            _, orientation, row, col = move
            orientation = coerce_orientation(orientation)
            row = int(row)
            col = int(col)
            new_state.walls.add((orientation, row, col))
            new_state._walls_frozenset = frozenset(new_state.walls)
            new_state._blocked_edge_mask_cache = 0
            new_state._blocked_edge_mask_cache_key = frozenset()
            new_state._wall_lookup_cache_key = frozenset()
            new_state.walls_left[new_state.current_player] -= 1
        else:
            raise ValueError(f"Unknown move type: {move_type!r}")

        new_state.current_player = new_state.current_player.opposite()
        if new_state.winner is None:
            key = new_state.repetition_key()
            count = new_state._repetition_counts.get(key, 0) + 1
            new_state._repetition_counts[key] = count
            if count >= REPETITION_DRAW_COUNT:
                new_state.is_draw = True
                new_state.draw_reason = "threefold_repetition"
        return new_state

    def apply_action(self, action: int, *, validate: bool = True) -> "BarricadeState":
        action = int(action)
        if validate:
            legal_actions = self.legal_actions()
            if action not in legal_actions:
                raise ValueError(f"Illegal action {action} for current state.")
        return self.apply_move(self.move_for_action(action))

    def move_for_action(self, action: int) -> Move:
        action = int(action)
        if self._is_pawn_action(action):
            for legal_action, move in self._get_pawn_action_moves():
                if legal_action == action:
                    return move
        return decode_action_for_board_size(action, self.board_size)

    def _is_pawn_action(self, action: int) -> bool:
        action = int(action)
        diagonal_hop_offset = diagonal_hop_offset_for_board_size(self.board_size)
        return 0 <= action < MOVE_ACTIONS or (
            diagonal_hop_offset <= action < diagonal_hop_offset + DIAGONAL_HOP_ACTIONS
        )

    def encode_move(self, move: Move) -> int:
        move_type = move[0]
        if move_type == "move":
            if len(move) != 2:
                raise ValueError("Move actions must be (\"move\", direction).")
            _, direction = move
            if isinstance(direction, MoveDirection):
                return direction.value
            return coerce_move_direction(direction).value

        if move_type == "move_diagonal":
            _, row_delta, col_delta = move
            return self._diagonal_hop_action_for_delta((int(row_delta), int(col_delta)))

        if move_type == "move_to":
            _, row, col = move
            current_row, current_col = self.pawns[self.current_player]
            row_delta = int(row) - current_row
            col_delta = int(col) - current_col
            if row_delta == 0 and abs(col_delta) in (1, 2):
                return MoveDirection.LEFT.value if col_delta < 0 else MoveDirection.RIGHT.value
            if col_delta == 0 and abs(row_delta) in (1, 2):
                return MoveDirection.UP.value if row_delta < 0 else MoveDirection.DOWN.value
            if abs(row_delta) == 1 and abs(col_delta) == 1:
                return self._diagonal_hop_action_for_delta((row_delta, col_delta))
            raise ValueError(f"Cannot encode pawn target from current position: {(row, col)}")

        if move_type == "wall":
            _, orientation, row, col = move
            if not isinstance(orientation, WallOrientation):
                orientation = coerce_orientation(orientation)
            row = int(row)
            col = int(col)
            if not (0 <= row < self.wall_board_size and 0 <= col < self.wall_board_size):
                raise ValueError(f"Wall anchor out of bounds: {(row, col)}")
            offset = (
                HORIZONTAL_WALL_OFFSET
                if orientation == WallOrientation.HORIZONTAL
                else HORIZONTAL_WALL_OFFSET + self.wall_board_size * self.wall_board_size
            )
            return offset + row * self.wall_board_size + col

        raise ValueError(f"Unknown move type: {move_type!r}")

    def decode_action(self, action: int) -> Move:
        return decode_action_for_board_size(action, self.board_size)


def apply_selected_action(
    state: BarricadeState,
    action: int,
    legal_actions: Sequence[int],
) -> BarricadeState:
    # Optimized check
    is_legal = False
    action_int = int(action)
    for la in legal_actions:
        if int(la) == action_int:
            is_legal = True
            break
    if not is_legal:
        raise RuntimeError(
            f"Policy selected illegal action {action}; legal actions were {list(legal_actions)}"
        )
    return state.apply_action(action_int, validate=False)


def state_lead_for_player(state: BarricadeState, player: Player | str | int) -> float:
    player = coerce_player(player)
    own_distance = state.shortest_path_length(player)
    opponent_distance = state.shortest_path_length(player.opposite())
    if own_distance is None or opponent_distance is None:
        return 0.0
    return float(opponent_distance - own_distance)


def path_score_for_player(state: BarricadeState, player: Player | str | int) -> float:
    distance = state.shortest_path_length(player)
    if distance is None:
        return float(state.board_size * state.board_size)
    return float(distance)


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
            raise ValueError("invalid_action_mode must be \"terminate\" or \"raise\".")
        if render_mode not in {None, "ansi", "human"}:
            raise ValueError("render_mode must be None, \"ansi\", or \"human\".")

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
            self.starting_player,
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

        self.state = self._new_state(
            red_start,
            blue_start,
            red_walls,
            blue_walls,
            starting_player,
        )
        self.steps = 0
        self.pawn_moves = {Player.RED: 0, Player.BLUE: 0}
        self.terminated = False
        self.truncated = False

        obs = self._get_observation()
        # Extract distances from obs to avoid re-calculation
        red_distance = int(obs["features"][3] * (self.state.board_size - 1)) if obs["features"][3] < 1.0 else None
        blue_distance = int(obs["features"][4] * (self.state.board_size - 1)) if obs["features"][4] < 1.0 else None
        
        return obs, self._get_info(red_distance, blue_distance)

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self.terminated or self.truncated:
            raise RuntimeError("step() called after episode ended. Call reset() first.")

        acting_player = self.state.current_player
        action_int = int(action)

        # Faster legal check
        legal_actions = self.state.legal_actions()
        if action_int not in legal_actions:
            # Re-check with full validation if it might be a decoding error
            try:
                decode_action_for_board_size(action_int, self.state.board_size)
            except (TypeError, ValueError) as exc:
                return self._handle_invalid_action(action_int, acting_player, str(exc))
            
            return self._handle_invalid_action(
                action_int, acting_player, f"Illegal action for current state: {action_int}"
            )
        
        decoded_move = self.state.move_for_action(action_int)

        self.state = self.state.apply_move(decoded_move)
        if decoded_move[0] in {"move", "move_diagonal", "move_to"}:
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
        elif self.state.is_draw:
            self.terminated = True
            reward = 0.0
        elif not self.state.get_valid_moves():
            self.state.winner = acting_player
            self.terminated = True
            reward = self.win_reward
        elif self.steps >= self.max_steps:
            self.truncated = True

        obs = self._get_observation()
        # Extract distances from obs to avoid re-calculation
        red_distance = int(obs["features"][3] * (self.state.board_size - 1)) if obs["features"][3] < 1.0 else None
        blue_distance = int(obs["features"][4] * (self.state.board_size - 1)) if obs["features"][4] < 1.0 else None

        info = self._get_info(red_distance, blue_distance)
        info.update(
            {
                "acting_player": acting_player.name,
                "last_action": action_int,
                "last_move": self.move_to_dict(decoded_move),
                "invalid_action": False,
            }
        )
        return obs, reward, self.terminated, self.truncated, info

    def legal_moves(self) -> List[Move]:
        return self.state.legal_moves()

    def legal_actions(self) -> List[int]:
        return self.state.legal_actions()

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
        starting_player: Player,
    ) -> BarricadeState:
        return BarricadeState(
            red_start=red_start,
            blue_start=blue_start,
            red_walls=red_walls,
            blue_walls=blue_walls,
            starting_player=starting_player,
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
        if move[0] == "move_diagonal":
            _, row_delta, col_delta = move
            return {
                "type": "move",
                "direction": "DIAGONAL",
                "row_delta": int(row_delta),
                "col_delta": int(col_delta),
            }
        if move[0] == "move_to":
            _, row, col = move
            return {
                "type": "move",
                "target": (int(row), int(col)),
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
        if np is not None:
            board = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        else:
            board = self._zeros((4, BOARD_SIZE, BOARD_SIZE))

        red_row, red_col = self.state.pawns[Player.RED]
        blue_row, blue_col = self.state.pawns[Player.BLUE]
        board[0][red_row][red_col] = 1.0
        board[1][blue_row][blue_col] = 1.0

        for orientation, row, col in self.state.walls:
            plane = 2 if orientation == WallOrientation.HORIZONTAL else 3
            board[plane][row][col] = 1.0

        # Greedy distances are often requested together
        red_distance = self.state.shortest_path_length(Player.RED)
        blue_distance = self.state.shortest_path_length(Player.BLUE)
        
        features_data = [
            float(self.state.current_player.value),
            self._normalize_wall_count(Player.RED),
            self._normalize_wall_count(Player.BLUE),
            self._normalize_distance(red_distance),
            self._normalize_distance(blue_distance),
        ]

        if np is not None:
            features = np.array(features_data, dtype=np.float32)
            mask = self.state.action_mask_numpy()
        else:
            features = features_data
            mask = self.state.action_mask()

        return {
            "board": board,
            "features": features,
            "action_mask": mask,
        }

    def _get_info(self, red_distance=None, blue_distance=None) -> Dict[str, Any]:
        if red_distance is None:
            red_distance = self.state.greedy_path_length(Player.RED)
        if blue_distance is None:
            blue_distance = self.state.greedy_path_length(Player.BLUE)
        lead = self._terminal_lead(red_distance, blue_distance)
        return {
            "current_player": self.state.current_player.name,
            "winner": (
                self.state.winner.name if self.state.winner is not None else None
            ),
            "draw": self.state.is_draw,
            "draw_reason": self.state.draw_reason,
            "steps": self.steps,
            "lead": lead,
            "red_position": self.state.pawns[Player.RED],
            "blue_position": self.state.pawns[Player.BLUE],
            "red_walls_left": self.state.walls_left[Player.RED],
            "blue_walls_left": self.state.walls_left[Player.BLUE],
            "red_path_length": red_distance,
            "blue_path_length": blue_distance,
        }

    def _zeros(self, shape: Tuple[int, ...]) -> Any:
        if np is not None:
            return np.zeros(shape, dtype=np.float32)

        # Fallback for when numpy is not installed.
        if len(shape) == 3:
            return [
                [[0.0 for _ in range(shape[2])] for _ in range(shape[1])]
                for _ in range(shape[0])
            ]
        if len(shape) == 2:
            return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]
        if len(shape) == 1:
            return [0.0 for _ in range(shape[0])]
        raise ValueError(f"Unsupported shape: {shape}")

    def _normalize_wall_count(self, player: Player) -> float:
        initial = self.state.initial_walls[player]
        if initial == 0:
            return 0.0
        return self.state.walls_left[player] / initial

    def _normalize_distance(self, distance: Optional[int]) -> float:
        if distance is None:
            return 1.0
        return distance / (self.state.board_size - 1)

    def _terminal_lead(self, red_distance: Optional[int], blue_distance: Optional[int]) -> float:
        if red_distance is None and blue_distance is None:
            return 0.0
        if red_distance is None:
            return -1.0
        if blue_distance is None:
            return 1.0
        return (blue_distance - red_distance) / (self.state.board_size - 1)

    def _render_ansi(self) -> str:
        """Render the board as an ANSI string."""
        board_size = self.board_size
        wall_board_size = self.wall_board_size
        horizontal_walls, vertical_walls = self.state._wall_lookup(self.state.walls)
        
        red_pos = self.state.pawns[Player.RED]
        blue_pos = self.state.pawns[Player.BLUE]
        
        lines = []
        for r in range(board_size):
            # Pawn row
            row_str = ""
            for c in range(board_size):
                if (r, c) == red_pos:
                    row_str += "R"
                elif (r, c) == blue_pos:
                    row_str += "B"
                else:
                    row_str += "."
                
                if c < wall_board_size:
                    if (r, c) in vertical_walls or (r-1, c) in vertical_walls:
                        row_str += "|"
                    else:
                        row_str += " "
            lines.append(row_str)
            
            # Wall row
            if r < wall_board_size:
                wall_row_str = ""
                for c in range(board_size):
                    if (r, c) in horizontal_walls or (r, c-1) in horizontal_walls:
                        wall_row_str += "-"
                    else:
                        wall_row_str += " "
                    
                    if c < wall_board_size:
                        wall_row_str += "+"
                lines.append(wall_row_str)
        
        return "\n".join(lines)
