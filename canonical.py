"""
Canonical frame utilities for Barricade.

These utilities transform board states and actions into a canonical frame
from the side-to-move perspective, such that the network always plays as
if it's advancing towards the last row.
"""

import torch
from torch import Tensor
from typing import Tuple

from barricade_env import (
    BOARD_SIZE,
    ACTION_SIZE,
    Player,
    MoveDirection,
    DIAGONAL_HOP_OFFSET,
)

WALL_BOARD_SIZE = BOARD_SIZE - 1
MOVE_ACTIONS = 4
WALL_ACTIONS_PER_ORIENTATION = WALL_BOARD_SIZE * WALL_BOARD_SIZE
HORIZONTAL_WALL_OFFSET = MOVE_ACTIONS
VERTICAL_WALL_OFFSET = HORIZONTAL_WALL_OFFSET + WALL_ACTIONS_PER_ORIENTATION

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
    index = torch.as_tensor(CANONICAL_FLIP_PERMUTATION, dtype=torch.long, device=vector.device)
    canonical = torch.empty_like(vector)
    canonical[index] = vector
    return canonical
