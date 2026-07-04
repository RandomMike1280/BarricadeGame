"""
Canonical frame utilities for Barricade.

These utilities transform board states and actions into a canonical frame
from the side-to-move perspective, such that the network always plays as
if it's advancing towards the last row.
"""

import torch
from torch import Tensor
from typing import Tuple, Dict

from barricade_env import (
    BOARD_SIZE,
    ACTION_SIZE,
    Player,
    MoveDirection,
    DIAGONAL_HOP_OFFSET,
    DIAGONAL_HOP_DELTAS,
)

WALL_BOARD_SIZE = BOARD_SIZE - 1
MOVE_ACTIONS = 4
WALL_ACTIONS_PER_ORIENTATION = WALL_BOARD_SIZE * WALL_BOARD_SIZE
HORIZONTAL_WALL_OFFSET = MOVE_ACTIONS
VERTICAL_WALL_OFFSET = HORIZONTAL_WALL_OFFSET + WALL_ACTIONS_PER_ORIENTATION

# Per-device cached permutation tensor. Without this, every call to
# ``canonicalize_action_vector`` would allocate a fresh long tensor via
# ``torch.as_tensor``. On post-Alder Lake CPUs this matters when batched
# training samples each touch a freshly-cloned policy vector.
_CANONICAL_FLIP_PERM_CACHE: Dict[torch.device, Tensor] = {}
# Same caching pattern for the LR-mirror permutation used by data
# augmentation; see ``LR_MIRROR_ACTION_PERMUTATION``.
_LR_MIRROR_PERM_CACHE: Dict[torch.device, Tensor] = {}

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
    perm[DIAGONAL_HOP_OFFSET + 0] = DIAGONAL_HOP_OFFSET + 2
    perm[DIAGONAL_HOP_OFFSET + 1] = DIAGONAL_HOP_OFFSET + 3
    perm[DIAGONAL_HOP_OFFSET + 2] = DIAGONAL_HOP_OFFSET + 0
    perm[DIAGONAL_HOP_OFFSET + 3] = DIAGONAL_HOP_OFFSET + 1
    return tuple(perm)


CANONICAL_FLIP_PERMUTATION: Tuple[int, ...] = _build_canonical_flip_permutation()


def canonical_action(action: int, player: Player) -> int:
    """Map a raw board action into the canonical (side-to-move) frame."""
    if player == Player.RED:
        return int(action)
    return CANONICAL_FLIP_PERMUTATION[int(action)]


def canonicalize_action_vector(vector: Tensor, player: Player) -> Tensor:
    """Reindex a per-action vector (policy target / mask) into canonical frame."""
    if player == Player.RED:
        return vector
    device = vector.device
    cached_index = _CANONICAL_FLIP_PERM_CACHE.get(device)
    if cached_index is None:
        cached_index = torch.as_tensor(
            CANONICAL_FLIP_PERMUTATION, dtype=torch.long, device=device
        )
        _CANONICAL_FLIP_PERM_CACHE[device] = cached_index
    canonical = torch.empty_like(vector)
    canonical[cached_index] = vector
    return canonical


def _build_canonical_lr_mirror_permutation() -> Tuple[int, ...]:
    """Action permutation applied under a left-right (column) mirror.

    Quoridor's 9x9 board is symmetric about the vertical axis through the
    center column, so a sample and its column-reflected counterpart describe
    identical game positions. The replay stores states in the canonical
    side-to-move frame, which has already collapsed the RED/BLUE player-swap
    symmetry; the only residual geometric symmetry is therefore the LR mirror.

    The permutation applied to the action index ``i`` under column reflection:

    - orthogonal pawn moves: UP and DOWN are unchanged; LEFT and RIGHT swap.
    - horizontal walls: each (row, col) -> (row, WB-1-col). The orientation is
      preserved because a horizontal wall reflected about a vertical axis is
      still horizontal.
    - vertical walls: each (row, col) -> (row, WB-1-col). Orientation is
      preserved because a vertical wall reflected about a vertical axis is
      still vertical.
    - diagonal hops: the four diagonal directions are reordered so each
      ``(row_delta, col_delta)`` pair maps to the reflected pair. With
      ``DIAGONAL_HOP_DELTAS = ((-1,-1), (-1,1), (1,-1), (1,1))`` and the
      LR mirror flipping the sign of ``col_delta``, the mapping is 0<->1
      and 2<->3.

    The permutation is its own inverse (an involution): applying it twice
    returns each action to its original index.
    """
    perm = [0] * ACTION_SIZE
    perm[MoveDirection.UP.value] = MoveDirection.UP.value
    perm[MoveDirection.DOWN.value] = MoveDirection.DOWN.value
    perm[MoveDirection.LEFT.value] = MoveDirection.RIGHT.value
    perm[MoveDirection.RIGHT.value] = MoveDirection.LEFT.value

    for offset in (HORIZONTAL_WALL_OFFSET, VERTICAL_WALL_OFFSET):
        for row in range(WALL_BOARD_SIZE):
            for col in range(WALL_BOARD_SIZE):
                perm[offset + row * WALL_BOARD_SIZE + col] = (
                    offset + row * WALL_BOARD_SIZE + (WALL_BOARD_SIZE - 1 - col)
                )

    # Diagonal hops: index into DIAGONAL_HOP_DELTAS, find the reflected
    # delta by flipping the column component, and emit the destination
    # index. This keeps the permutation correct even if the diagonal
    # ordering ever changes upstream.
    for diagonal_index, (row_delta, col_delta) in enumerate(DIAGONAL_HOP_DELTAS):
        reflected_delta = (row_delta, -col_delta)
        reflected_index = DIAGONAL_HOP_DELTAS.index(reflected_delta)
        perm[DIAGONAL_HOP_OFFSET + diagonal_index] = (
            DIAGONAL_HOP_OFFSET + reflected_index
        )

    return tuple(perm)


LR_MIRROR_ACTION_PERMUTATION: Tuple[int, ...] = _build_canonical_lr_mirror_permutation()


def canonicalize_lr_mirror_action_vector(vector: Tensor) -> Tensor:
    """Reindex a per-action vector (policy target / mask) under LR mirror.

    Used by data augmentation to produce the column-reflected counterpart of
    a canonical-frame replay sample. The total mass of the input vector is
    preserved (the permutation is a bijection), so a renormalized policy
    target stays normalized after the reindex.
    """
    device = vector.device
    cached_index = _LR_MIRROR_PERM_CACHE.get(device)
    if cached_index is None:
        cached_index = torch.as_tensor(
            LR_MIRROR_ACTION_PERMUTATION, dtype=torch.long, device=device
        )
        _LR_MIRROR_PERM_CACHE[device] = cached_index
    mirrored = torch.empty_like(vector)
    mirrored[cached_index] = vector
    return mirrored
