"""
AlphaZero-style PyTorch network and state encoder for Barricade.

The model consumes a stack of 9x9 feature planes and returns:
    policy_logits: shape (batch, ACTION_SIZE)
    value: shape (batch, 1), bounded to [-1, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from barricade_env import (
    ACTION_SIZE,
    BOARD_SIZE,
    Player,
    WallOrientation,
)


BASE_PLANES_PER_POSITION = 9


@dataclass(frozen=True)
class EncoderConfig:
    """Configuration for stacking current and previous game-state planes."""

    history_length: int = 0

    @property
    def input_planes(self) -> int:
        return BASE_PLANES_PER_POSITION * (self.history_length + 1)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def encode_state_planes(state, *, dtype=torch.float32) -> Tensor:
    """
    Encode one BarricadeState into base 9x9 planes.

    Plane layout:
        0: side to move, all 1 for red and all 0 for blue
        1: red pawn one-hot
        2: blue pawn one-hot
        3: horizontal wall anchors
        4: vertical wall anchors
        5: red remaining walls, broadcast as a scalar ratio
        6: blue remaining walls, broadcast as a scalar ratio
        7: red goal row
        8: blue goal row
    """

    planes = torch.zeros((BASE_PLANES_PER_POSITION, BOARD_SIZE, BOARD_SIZE), dtype=dtype)

    if state.current_player == Player.RED:
        planes[0].fill_(1.0)

    red_row, red_col = state.pawns[Player.RED]
    blue_row, blue_col = state.pawns[Player.BLUE]
    planes[1, red_row, red_col] = 1.0
    planes[2, blue_row, blue_col] = 1.0

    for orientation, row, col in state.walls:
        if orientation == WallOrientation.HORIZONTAL:
            planes[3, row, col] = 1.0
            if col + 1 < BOARD_SIZE:
                planes[3, row, col + 1] = 1.0
        else:
            planes[4, row, col] = 1.0
            if row + 1 < BOARD_SIZE:
                planes[4, row + 1, col] = 1.0

    planes[5].fill_(
        _safe_ratio(state.walls_left[Player.RED], state.initial_walls[Player.RED])
    )
    planes[6].fill_(
        _safe_ratio(state.walls_left[Player.BLUE], state.initial_walls[Player.BLUE])
    )
    planes[7, BOARD_SIZE - 1, :] = 1.0
    planes[8, 0, :] = 1.0
    return planes


def encode_state_stack(
    current_state,
    history: Optional[Sequence] = None,
    *,
    history_length: int = 0,
    dtype=torch.float32,
) -> Tensor:
    """
    Encode current state plus previous states into a Cx9x9 tensor.

    ``history`` should be ordered from oldest to newest. The returned tensor is
    ordered from current state first, then most-recent previous states. Missing
    history slots are zero-filled to keep a fixed channel count.
    """

    planes: List[Tensor] = [encode_state_planes(current_state, dtype=dtype)]
    history = list(history or [])

    for state in reversed(history[-history_length:]):
        planes.append(encode_state_planes(state, dtype=dtype))

    missing = history_length - (len(planes) - 1)
    for _ in range(max(0, missing)):
        planes.append(torch.zeros_like(planes[0]))

    return torch.cat(planes, dim=0)


class ConvBlock(nn.Module):
    """3x3 convolution followed by batch norm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """AlphaZero-style residual block with two 3x3 convolutions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class PolicyHead(nn.Module):
    """1x1 policy head that emits unnormalized action logits."""

    def __init__(self, channels: int, action_size: int = ACTION_SIZE) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, action_size)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn(self.conv(x)))
        out = torch.flatten(out, start_dim=1)
        return self.fc(out)


class ValueHead(nn.Module):
    """1x1 value head that emits a scalar in [-1, 1]."""

    def __init__(self, channels: int, hidden_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn(self.conv(x)))
        out = torch.flatten(out, start_dim=1)
        out = F.relu(self.fc1(out))
        return torch.tanh(self.fc2(out))


class AlphaZeroNetwork(nn.Module):
    """
    AlphaZero-style policy/value network.

    Args:
        input_planes: Number of 9x9 input planes.
        conv_channels: M filters for the initial convolutional tower.
        residual_channels: N filters in each residual block. If different from
            ``conv_channels``, a 1x1 projection is inserted before the residual
            tower so skip connections remain valid.
        num_conv_layers: C initial 3x3 conv/bn/relu layers.
        num_residual_layers: L residual blocks.
        value_hidden_size: K hidden units in the value head.
    """

    def __init__(
        self,
        *,
        input_planes: int,
        action_size: int = ACTION_SIZE,
        conv_channels: int = 128,
        residual_channels: Optional[int] = None,
        num_conv_layers: int = 1,
        num_residual_layers: int = 10,
        value_hidden_size: int = 256,
    ) -> None:
        super().__init__()
        if input_planes <= 0:
            raise ValueError("input_planes must be positive.")
        if num_conv_layers <= 0:
            raise ValueError("num_conv_layers must be positive.")
        if num_residual_layers < 0:
            raise ValueError("num_residual_layers must be non-negative.")

        residual_channels = residual_channels or conv_channels
        conv_layers = []
        in_channels = input_planes
        for _ in range(num_conv_layers):
            conv_layers.append(ConvBlock(in_channels, conv_channels))
            in_channels = conv_channels
        self.conv_tower = nn.Sequential(*conv_layers)

        if conv_channels == residual_channels:
            self.residual_projection = nn.Identity()
        else:
            self.residual_projection = nn.Sequential(
                nn.Conv2d(conv_channels, residual_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(residual_channels),
                nn.ReLU(inplace=True),
            )

        self.residual_tower = nn.Sequential(
            *[ResidualBlock(residual_channels) for _ in range(num_residual_layers)]
        )
        self.policy_head = PolicyHead(residual_channels, action_size)
        self.value_head = ValueHead(residual_channels, value_hidden_size)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        out = self.conv_tower(x)
        out = self.residual_projection(out)
        out = self.residual_tower(out)
        return self.policy_head(out), self.value_head(out)

    @torch.no_grad()
    def predict(self, x: Tensor, action_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """
        Return masked policy probabilities and value for inference.

        ``forward`` returns logits for training. This helper is for MCTS/action
        selection and applies softmax, optionally masking illegal actions.
        """

        logits, value = self.forward(x)
        if action_mask is not None:
            action_mask = action_mask.to(device=logits.device, dtype=torch.bool)
            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        return torch.softmax(logits, dim=1), value


def build_network(
    *,
    history_length: int = 0,
    conv_channels: int = 128,
    residual_channels: Optional[int] = None,
    num_conv_layers: int = 1,
    num_residual_layers: int = 10,
    value_hidden_size: int = 256,
) -> AlphaZeroNetwork:
    config = EncoderConfig(history_length=history_length)
    return AlphaZeroNetwork(
        input_planes=config.input_planes,
        conv_channels=conv_channels,
        residual_channels=residual_channels,
        num_conv_layers=num_conv_layers,
        num_residual_layers=num_residual_layers,
        value_hidden_size=value_hidden_size,
    )
