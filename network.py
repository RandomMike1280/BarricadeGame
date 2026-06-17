"""
AlphaZero-style PyTorch network and state encoder for Barricade.

The model consumes a stack of 9x9 feature planes and returns:
    policy_logits: shape (batch, ACTION_SIZE)
    value: shape (batch, 1), bounded to [-1, 1]
    lead: shape (batch, 1), unbounded race-margin estimate
    future_logits: shape (batch, 2, 9, 9)
    score: shape (batch, 1), non-negative steps-to-win estimate
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


def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * (torch.expm1(torch.abs(x)))


def encode_state_planes(state, *, dtype=torch.float32) -> Tensor:
    """
    Encode one BarricadeState into base planes in a canonical frame.
    (Inlined row operations for Python-side loop speedup).
    """
    planes = torch.zeros((BASE_PLANES_PER_POSITION, BOARD_SIZE, BOARD_SIZE), dtype=dtype)
    current = state.current_player
    opponent = current.opposite()
    flip = current == Player.BLUE

    own_row, own_col = state.pawns[current]
    opp_row, opp_col = state.pawns[opponent]

    if flip:
        own_r = BOARD_SIZE - 1 - own_row
        opp_r = BOARD_SIZE - 1 - opp_row
    else:
        own_r = own_row
        opp_r = opp_row

    planes[0].fill_(1.0)
    planes[1, own_r, own_col] = 1.0
    planes[2, opp_r, opp_col] = 1.0

    for orientation, row, col in state.walls:
        plane = 3 if orientation == WallOrientation.HORIZONTAL else 4
        r = (BOARD_SIZE - 2 - row) if flip else row
        planes[plane, r, col] = 1.0

    own_initial = max(1, state.initial_walls[current])
    opp_initial = max(1, state.initial_walls[opponent])
    planes[5].fill_(state.walls_left[current] / own_initial)
    planes[6].fill_(state.walls_left[opponent] / opp_initial)

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
    (Pre-allocated lists to minimize dynamic array resizing overhead).
    """
    history = list(history or [])
    total_states = history_length + 1
    planes = [None] * total_states
    
    planes[0] = encode_state_planes(current_state, dtype=dtype)
    
    if history_length > 0:
        recent_history = history[-history_length:]
        for i, state in enumerate(reversed(recent_history)):
            planes[i + 1] = encode_state_planes(state, dtype=dtype)
            
    for i in range(1, total_states):
        if planes[i] is None:
            planes[i] = torch.zeros_like(planes[0])
            
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
        return F.relu(self.bn(self.conv(x)), inplace=True)


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
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += residual  # In-place addition equivalent to out = out + residual
        return F.relu(out, inplace=True)


class PolicyHead(nn.Module):
    """1x1 policy head that emits unnormalized action logits."""

    def __init__(self, channels: int, action_size: int = ACTION_SIZE) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, action_size)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn(self.conv(x)), inplace=True)
        out = out.flatten(1)  # Slightly faster than torch.flatten
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
        out = F.relu(self.bn(self.conv(x)), inplace=True)
        out = out.flatten(1)
        out = F.relu(self.fc1(out), inplace=True)
        return torch.tanh(self.fc2(out))


class ScalarHead(nn.Module):
    """1x1 scalar head for auxiliary regression targets."""

    def __init__(self, channels: int, hidden_size: int, *, positive: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.positive = positive

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn(self.conv(x)), inplace=True)
        out = out.flatten(1)
        out = F.relu(self.fc1(out), inplace=True)
        out = self.fc2(out)
        if self.positive:
            out = F.softplus(out)
        return out


class FutureMapHead(nn.Module):
    """Per-square future-occupancy logits for side-to-move and opponent."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, 2, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return self.conv2(out)


class AlphaZeroNetwork(nn.Module):
    """
    AlphaZero-style policy/value network.
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
        self.lead_head = ScalarHead(residual_channels, value_hidden_size)
        self.future_map_head = FutureMapHead(residual_channels)
        self.score_head = ScalarHead(
            residual_channels,
            value_hidden_size,
            positive=True,
        )

        # Optimize for modern hardware (Tensor Cores) using channels_last memory format
        # This heavily accelerates convolutions on RTX 20/30/40+ series cards
        self.to(memory_format=torch.channels_last)

    def _features(self, x: Tensor) -> Tensor:
        # Ensure inputs are contiguous in channels_last format for conv optimization
        if x.dim() == 4 and not x.is_contiguous(memory_format=torch.channels_last):
            x = x.to(memory_format=torch.channels_last)
            
        out = self.conv_tower(x)
        out = self.residual_projection(out)
        return self.residual_tower(out)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        out = self._features(x)
        return (
            self.policy_head(out),
            self.value_head(out),
            self.lead_head(out),
            self.future_map_head(out),
            self.score_head(out),
        )

    def inference(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return only the heads needed by MCTS/inference."""
        out = self._features(x)
        return self.policy_head(out), self.value_head(out), symexp(self.lead_head(out))

    @torch.inference_mode()
    def predict(
        self,
        x: Tensor,
        action_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Return masked policy probabilities, value, and lead for inference.
        """
        logits, value, lead = self.inference(x)
        if action_mask is not None:
            action_mask = action_mask.to(device=logits.device, dtype=torch.bool)
            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        return torch.softmax(logits, dim=1), value, lead  # Bug fixed: lead was getting symexp twice


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