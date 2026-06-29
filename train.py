"""
Self-play and training pipeline for Barricade AlphaZero experiments.

Supports parallel self-play with cross-worker batched neural-network inference
via a background BatchInferenceServer thread.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field, replace
import multiprocessing as mp
import os
import queue
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, TextIO, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from barricade_env import (
    ACTION_SIZE,
    BOARD_SIZE,
    DEFAULT_WALLS_PER_PLAYER,
    BarricadeEnv,
    BarricadeState,
    Player,
)
from canonical import canonical_action, canonicalize_action_vector
from mcts import MCTS, MCTSConfig
from network import EncoderConfig, build_network, encode_state_stack


SELFPLAY_DIR = Path("data/selfplay")
CHECKPOINT_DIR = Path("checkpoints")
TRAIN_LOG_DIR = Path(".")


# ======================================================================
# Configuration dataclasses
# ======================================================================

@dataclass(frozen=True)
class NetworkConfig:
    history_length: int = 0
    conv_channels: int = 128
    residual_channels: Optional[int] = None
    num_conv_layers: int = 1
    num_residual_layers: int = 10
    value_hidden_size: int = 256


@dataclass(frozen=True)
class SelfPlayConfig:
    iteration: int = 1
    games: int = 16
    base_simulations: int = 128
    batch_size: int = 16
    max_steps: int = 100
    chunk_size: int = 2048
    temperature_drop_ply: int = 20
    seed: int = 1
    output_dir: str = str(SELFPLAY_DIR)
    base_walls: int = DEFAULT_WALLS_PER_PLAYER
    num_workers: int = 4
    # Resignation: end clearly-lost games early so they become decisive (kept)
    # samples instead of burning compute on a long truncated draw (kept=0). A
    # fraction of games disable resignation to calibrate the false-positive rate.
    resign_threshold: float = -0.85
    resign_plies: int = 6
    resign_disable_fraction: float = 0.1
    # Adjudicate threefold-repetition (shuffle) draws by race lead so self-play
    # stays decisive and the value head keeps a win/loss signal instead of
    # collapsing to 0 everywhere. Set False to keep pure rules draws.
    adjudicate_repetition_draws: bool = True
    # Force process-based (vs thread+server) self-play; None => auto by device.
    use_processes: Optional[bool] = None


@dataclass(frozen=True)
class TrainConfig:
    iteration: int = 1
    replay_dir: str = str(SELFPLAY_DIR)
    checkpoint_dir: str = str(CHECKPOINT_DIR)
    replay_limit: Optional[int] = None
    replay_max_iteration: Optional[int] = None
    # Sliding replay window: train on the current iteration plus the previous N
    # iterations of self-play data. Default 1 == current + previous iteration.
    # 0/None == unbounded.
    replay_window_size: Optional[int] = 1
    epochs: int = 1
    batch_size: int = 256
    learning_rate: float = 1e-3
    value_loss_weight: float = 1.0
    lead_loss_weight: float = 0.1
    future_loss_weight: float = 0.1
    score_loss_weight: float = 0.01
    grad_clip: float = 1.0
    weight_decay: float = 1e-4
    seed: int = 1


@dataclass(frozen=True)
class EvalConfig:
    games: int = 10
    simulations: int = 64
    batch_size: int = 16
    max_steps: int = 500
    seed: int = 1
    base_walls: int = DEFAULT_WALLS_PER_PLAYER


@dataclass(frozen=True)
class PipelineConfig:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# ======================================================================
# Utility helpers
# ======================================================================

def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m{secs:.0f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h{mins}m"


def _timestamp() -> str:
    return time.strftime("%H:%M:%S")


def _system_mem_stats() -> Tuple[Optional[float], Optional[float]]:
    """Return ``(mem_available_mb, swap_used_mb)`` from ``/proc/meminfo``.

    Returns ``(None, None)`` when unavailable (non-Linux). Used to make the
    self-play slowdown observable: the catastrophic per-game blow-up was caused
    by the host crossing into swap, so logging available RAM + swap-in per game
    lets us confirm the fix keeps memory flat.
    """
    try:
        fields: Dict[str, int] = {}
        with open("/proc/meminfo", "r") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 2:
                    fields[parts[0].rstrip(":")] = int(parts[1])  # kB
        mem_avail = fields.get("MemAvailable")
        swap_total = fields.get("SwapTotal")
        swap_free = fields.get("SwapFree")
        mem_avail_mb = mem_avail / 1024.0 if mem_avail is not None else None
        swap_used_mb = (
            (swap_total - swap_free) / 1024.0
            if swap_total is not None and swap_free is not None
            else None
        )
        return mem_avail_mb, swap_used_mb
    except (OSError, ValueError):
        return None, None


class _TeeStream:
    """Write a stream to the terminal and a log file, flushing each write."""

    def __init__(
        self,
        terminal_stream: TextIO,
        log_file: TextIO,
        lock: threading.Lock,
    ) -> None:
        self._terminal_stream = terminal_stream
        self._log_file = log_file
        self._lock = lock

    def write(self, text: str) -> int:
        with self._lock:
            written = self._terminal_stream.write(text)
            self._terminal_stream.flush()
            self._log_file.write(text)
            self._log_file.flush()
        return written

    def flush(self) -> None:
        with self._lock:
            self._terminal_stream.flush()
            self._log_file.flush()

    def isatty(self) -> bool:
        return self._terminal_stream.isatty()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._terminal_stream, name)


def _open_next_train_log_file(log_dir: Path = TRAIN_LOG_DIR) -> Tuple[Path, TextIO]:
    log_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"train_log_{time.strftime('%d_%m')}"

    for suffix in [""] + [f"_{index}" for index in range(1, 10000)]:
        path = log_dir / f"{base_name}{suffix}.txt"
        try:
            return path, path.open("x", encoding="utf-8", buffering=1)
        except FileExistsError:
            continue

    raise RuntimeError(f"Could not find an unused train log path in {log_dir}")


def _start_train_log_capture() -> Path:
    log_path, log_file = _open_next_train_log_file()
    lock = threading.Lock()
    sys.stdout = _TeeStream(sys.stdout, log_file, lock)  # type: ignore[assignment]
    sys.stderr = _TeeStream(sys.stderr, log_file, lock)  # type: ignore[assignment]
    print(f"[{_timestamp()}] [log] writing training output to {log_path}", flush=True)
    return log_path


# ======================================================================
# BatchedInferenceServer
# ======================================================================

class BatchInferenceServer:
    """Background-thread inference server that batches NN requests across
    all parallel game workers.

    Worker threads call ``infer(batch)`` which enqueues the request and blocks
    until the result is ready.  The server thread collects pending requests
    (up to ``max_batch_size``), concatenates them into a single mega-batch,
    runs one forward pass, and distributes the results back.

    This dramatically improves GPU utilization when running many parallel
    games: instead of N workers each running batch_size=16 inferences
    separately, the server runs a single batch of N*16 (or more) states.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        max_batch_size: int = 512,
        collection_timeout: float = 0.001,
    ) -> None:
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.max_batch_size = max_batch_size
        self.collection_timeout = collection_timeout

        self._queue: queue.Queue = queue.Queue()
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._total_batches = 0
        self._total_inferences = 0
        self._max_batch_seen = 0

    def start(self) -> None:
        self._stop = False
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

    def infer(
        self, batch: Tensor
    ) -> Tuple[List[List[float]], List[float], List[float]]:
        """Submit a batch of states for inference and block until results are
        ready.

        Returns ``(logits_list, values_list, leads_list)`` as plain Python
        lists (one entry per state in *batch*).
        """
        result_queue: queue.Queue = queue.Queue()
        self._queue.put((batch, result_queue))
        return result_queue.get()

    def _serve(self) -> None:
        while not self._stop:
            try:
                first = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            items: List[Tuple[Tensor, queue.Queue]] = [first]
            total_size = first[0].shape[0]

            # Try to collect more items to form a larger batch
            if self.collection_timeout > 0 and total_size < self.max_batch_size:
                deadline = time.monotonic() + self.collection_timeout
                while total_size < self.max_batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = self._queue.get(timeout=remaining)
                        items.append(item)
                        total_size += item[0].shape[0]
                    except queue.Empty:
                        break
            else:
                while total_size < self.max_batch_size:
                    try:
                        item = self._queue.get_nowait()
                        items.append(item)
                        total_size += item[0].shape[0]
                    except queue.Empty:
                        break

            # Concatenate all worker batches into one mega-batch
            batches = [item[0] for item in items]
            if len(batches) == 1:
                batch = batches[0].to(self.device, non_blocking=True)
            else:
                batch = torch.cat(batches, dim=0).to(self.device, non_blocking=True)

            # Single forward pass for all workers
            with torch.inference_mode():
                logits_t, values_t, leads_t = self.model.inference(batch)
                # Convert to Python lists — one GPU sync point
                logits_list = logits_t.tolist()
                values_list = values_t.view(-1).tolist()
                leads_list = leads_t.view(-1).tolist()

            # Distribute results back to waiting workers
            offset = 0
            for item_batch, result_queue in items:
                size = item_batch.shape[0]
                result_queue.put(
                    (
                        logits_list[offset : offset + size],
                        values_list[offset : offset + size],
                        leads_list[offset : offset + size],
                    )
                )
                offset += size

            with self._lock:
                self._total_batches += 1
                self._total_inferences += total_size
                if total_size > self._max_batch_seen:
                    self._max_batch_seen = total_size

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_batches": self._total_batches,
                "total_inferences": self._total_inferences,
                "avg_batch_size": self._total_inferences / max(1, self._total_batches),
                "max_batch_size": self._max_batch_seen,
            }


# ======================================================================
# Replay dataset
# ======================================================================

def _is_trainable_game(winner: Any, truncated: bool) -> bool:
    """Whether a finished game's samples are worth training on.

    Keeps decisive games (a winner) and rules draws (a terminal draw reached by
    the game's own rules, ``winner is None`` with ``truncated`` False — value
    target 0.0 is the true outcome). Skips ONLY truncation draws: games that hit
    the step limit with no result, whose 0.0 value target labels an unfinished
    position and is therefore noise.
    """
    return not (winner is None and bool(truncated))


def _is_trainable_replay_sample(sample: Dict[str, Any]) -> bool:
    return _is_trainable_game(
        sample.get("winner"),
        bool(sample.get("truncated", False)),
    )


_REPLAY_ITERATION_RE = re.compile(r"^iter_(\d+)_chunk_\d+\.pt$")


def _replay_iteration_from_path(path: Path) -> Optional[int]:
    match = _REPLAY_ITERATION_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def _policy_target_is_legal(sample: Dict[str, Any]) -> bool:
    target = sample.get("policy_target")
    if target is None:
        return False
    policy_target = torch.as_tensor(target, dtype=torch.float32)
    mask = torch.as_tensor(
        sample.get("mask", torch.ones_like(policy_target)),
        dtype=torch.bool,
    )
    if policy_target.numel() != mask.numel():
        return False
    total_mass = float(policy_target.sum().item())
    if not torch.isfinite(policy_target).all() or total_mass <= 1.0e-8:
        return False
    illegal_mass = float(policy_target[~mask].abs().sum().item())
    return illegal_mass <= 1.0e-5


def _renormalize_policy_target(policy_target: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(device=policy_target.device, dtype=torch.bool)
    legal_target = policy_target.masked_fill(~mask, 0.0)
    row_sum = legal_target.sum(dim=1, keepdim=True)
    legal_count = mask.sum(dim=1, keepdim=True)
    uniform = mask.to(dtype=policy_target.dtype) / legal_count.clamp_min(1)
    normalized = legal_target / row_sum.clamp_min(1.0e-8)
    return torch.where(row_sum > 1.0e-8, normalized, uniform)


class ReplayDataset(Dataset):
    def __init__(
        self,
        replay_dir: str | Path,
        limit: Optional[int] = None,
        *,
        max_iteration: Optional[int] = None,
        min_iteration: Optional[int] = None,
    ) -> None:
        self.replay_dir = Path(replay_dir)
        samples: List[Dict[str, Any]] = []
        self.skipped_future_chunks = 0
        self.skipped_old_chunks = 0
        for path in sorted(self.replay_dir.glob("*.pt")):
            replay_iteration = _replay_iteration_from_path(path)
            if (
                max_iteration is not None
                and replay_iteration is not None
                and replay_iteration > max_iteration
            ):
                self.skipped_future_chunks += 1
                continue
            if (
                min_iteration is not None
                and replay_iteration is not None
                and replay_iteration < min_iteration
            ):
                self.skipped_old_chunks += 1
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False)
            samples.extend(payload.get("samples", []))

        self.total_samples_loaded = len(samples)
        trainable_samples = [
            sample for sample in samples if _is_trainable_replay_sample(sample)
        ]
        bad_policy_games = set()
        bad_policy_ungrouped = set()
        for index, sample in enumerate(trainable_samples):
            if _policy_target_is_legal(sample):
                continue
            game_id = sample.get("game_id")
            if game_id is None:
                bad_policy_ungrouped.add(index)
            else:
                bad_policy_games.add(game_id)

        self.samples = [
            sample
            for index, sample in enumerate(trainable_samples)
            if sample.get("game_id") not in bad_policy_games
            and index not in bad_policy_ungrouped
        ]
        self.filtered_samples = self.total_samples_loaded - len(trainable_samples)
        self.filtered_policy_samples = len(trainable_samples) - len(self.samples)
        self.filtered_policy_games = len(bad_policy_games)
        self.replay_max_iteration = max_iteration
        self.replay_min_iteration = min_iteration

        if limit is not None and limit > 0 and len(self.samples) > limit:
            self.samples = self.samples[-limit:]
        if not self.samples:
            raise ValueError(
                f"No trainable (non-truncation) replay samples found in {self.replay_dir}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, index: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        sample = self.samples[index]
        state_planes = sample["state_planes"].float()
        mask = torch.as_tensor(
            sample.get("mask", torch.ones(ACTION_SIZE)), dtype=torch.bool
        )
        policy_target = sample["policy_target"].float()
        policy_target = _renormalize_policy_target(
            policy_target.unsqueeze(0),
            mask.unsqueeze(0),
        ).squeeze(0)
        value_target = torch.as_tensor(sample["value_target"], dtype=torch.float32)
        lead_target = torch.as_tensor(
            sample.get("lead_target", sample.get("lead", 0.0)) or 0.0,
            dtype=torch.float32,
        )
        lead_mask = torch.as_tensor(
            sample.get(
                "lead_mask",
                1.0 if sample.get("lead", None) is not None else 0.0,
            ),
            dtype=torch.float32,
        )

        future_map_target = sample.get("future_map_target")
        if future_map_target is None:
            future_map_target = torch.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
            future_map_mask = 0.0
        else:
            future_map_target = torch.as_tensor(future_map_target, dtype=torch.float32)
            future_map_mask = sample.get("future_map_mask", 1.0)

        score_target = torch.as_tensor(sample.get("score_target", 0.0), dtype=torch.float32)
        score_mask = torch.as_tensor(sample.get("score_mask", 0.0), dtype=torch.float32)
        return (
            state_planes,
            mask,
            policy_target,
            value_target,
            lead_target,
            lead_mask,
            future_map_target,
            torch.as_tensor(future_map_mask, dtype=torch.float32),
            score_target,
            score_mask,
        )


# ======================================================================
# Self-play helpers
# ======================================================================

def sample_playout_cap(base_simulations: int, rng: random.Random) -> int:
    roll = rng.random()
    if roll < 0.60:
        cap = base_simulations
    elif roll < 0.80:
        cap = max(1, base_simulations // 2)
    elif roll < 0.93:
        cap = max(1, base_simulations // 4)
    else:
        cap = base_simulations * 2
    return max(1, int(cap))


def sample_handicap(base_walls: int, rng: random.Random) -> Dict[str, Any]:
    center = BOARD_SIZE // 2
    starting_player = Player.RED if rng.random() < 0.5 else Player.BLUE

    red_row = 0
    blue_row = BOARD_SIZE - 1
    red_col = center
    blue_col = center
    red_walls = base_walls
    blue_walls = base_walls

    active_modes = []

    if rng.random() < 0.70:
        red_col = rng.randint(1, BOARD_SIZE - 2)
        blue_col = rng.randint(1, BOARD_SIZE - 2)
        active_modes.append("column_shift")

    if rng.random() < 0.20:
        red_row = rng.randint(0, 1)
        blue_row = rng.randint(BOARD_SIZE - 2, BOARD_SIZE - 1)
        active_modes.append("row_ahead")

    if rng.random() < 0.10:
        red_walls = max(0, base_walls + rng.randint(-2, 2))
        blue_walls = max(0, base_walls + rng.randint(-2, 2))
        active_modes.append("wall_handicap")

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


def sample_valid_handicap(
    base_walls: int, rng: random.Random, max_attempts: int = 100
) -> Dict[str, Any]:
    for _ in range(max_attempts):
        config = sample_handicap(base_walls, rng)
        env = BarricadeEnv(max_steps=1)
        try:
            env.reset(options=_env_options(config))
            return config
        except ValueError:
            continue
    raise RuntimeError("Failed to sample a valid handicap configuration.")


def masked_logits(logits: Tensor, mask: Tensor) -> Tensor:
    logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
    logits = logits.clamp(-30.0, 30.0)
    mask = mask.to(device=logits.device, dtype=torch.bool)
    return logits.masked_fill(~mask, -1.0e9)


def policy_action_for_mcts(action: int, state: BarricadeState) -> int:
    return canonical_action(action, state.current_player)


# ======================================================================
# Single-game worker (runs in a thread)
# ======================================================================

def _run_single_game_worker(
    game_index: int,
    iteration: int,
    model: nn.Module,
    network_config: NetworkConfig,
    config: SelfPlayConfig,
    inference_server: Optional[BatchInferenceServer],
    seed: int,
    *,
    device: str = "cpu",
) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
    """Run one complete self-play game.

    When ``inference_server`` is ``None`` the game evaluates the network
    synchronously in-process (used by the process-based CPU path, which has no
    shared server); otherwise it delegates to the cross-worker batched server.

    Returns ``(samples, elapsed_seconds, game_info)``.
    """
    rng = random.Random(seed)
    game_id = f"iter_{iteration:06d}_game_{game_index:06d}"
    game_start = time.time()

    handicap = sample_valid_handicap(config.base_walls, rng)
    env = BarricadeEnv(max_steps=config.max_steps, invalid_action_mode="raise")
    _, info = env.reset(options=_env_options(handicap))
    history: List[Any] = []
    pending: List[Dict[str, Any]] = []
    pawn_visits: List[Tuple[int, str, Tuple[int, int]]] = []
    root = None
    terminated = False
    truncated = False
    ply = 0

    # Resignation bookkeeping: a fraction of games never resign (calibration set).
    resign_enabled = rng.random() >= config.resign_disable_fraction
    resign_low_streak = {Player.RED: 0, Player.BLUE: 0}
    resigned_winner: Optional[str] = None

    resolved_device = (
        str(inference_server.device) if inference_server is not None else str(device)
    )

    # Create MCTS once per game and update config per-move via replace()
    mcts_config = MCTSConfig(
        num_simulations=config.base_simulations,
        batch_size=config.batch_size,
        history_length=network_config.history_length,
        device=resolved_device,
        add_root_noise=True,
        action_temperature=1.5,
        # Policy TARGET (training label) is the true normalized visit
        # distribution at a fixed temperature, decoupled from the per-move
        # action-SELECTION temperature schedule (which still varies below).
        # Without this it falls back to action_temperature -> one-hot targets
        # after the temperature drop, which collapses the policy onto walls.
        policy_target_temperature=1.0,
        # Guarantee every legal move keeps nonzero target mass so the 4 pawn
        # moves can never be driven to zero prior by the 128-wide wall action
        # space (the observed policy-collapse failure mode).
        policy_target_floor=0.03,
        # Dirichlet alpha scaled to the ~130-move branching factor (~10/N) so
        # low-prior pawn moves are actually surfaced and visited at the root.
        root_dirichlet_alpha=0.1,
        # FPU=0: do NOT penalize unvisited moves. With the default 0.33 and a
        # near-flat value head, every unvisited move is scored ~parent_q-0.31,
        # so low-prior pawn moves never earn a first visit even at 16k sims and
        # MCTS just follows the (wall-biased) policy. 0 lets the search actually
        # probe pawn moves and discover terminal wins at the 512-sim budget.
        fpu_reduction=0.0,
    )
    mcts = MCTS(
        model,
        mcts_config,
        policy_action_transform=policy_action_for_mcts,
        inference_server=inference_server,
    )

    while not terminated and not truncated:
        player_to_move = env.state.current_player
        state_before = env.state.copy()
        history_before = _history_window(history, network_config.history_length)
        state_planes = encode_state_stack(
            state_before,
            history_before,
            history_length=network_config.history_length,
        ).cpu()

        simulations = sample_playout_cap(config.base_simulations, rng)
        temp = 2.0 if ply < config.temperature_drop_ply else 0.0
        mcts.config = replace(
            mcts.config,
            num_simulations=simulations,
            action_temperature=temp,
        )

        result = mcts.search(state_before, history=history_before, root=root)
        action = mcts.select_action(result, temperature=temp)
        legal_actions = list(env.legal_actions())
        if int(action) not in set(legal_actions):
            raise RuntimeError(
                f"MCTS selected illegal action {action} at ply {ply} "
                f"for {player_to_move.name}; legal actions were {legal_actions}"
            )

        # Resignation: if the side to move has been decisively losing
        # (root value <= threshold) for ``resign_plies`` of its own turns,
        # concede now and award the win to the opponent. Only after the
        # exploratory opening, since early root values are noisy.
        if resign_enabled and ply >= config.temperature_drop_ply:
            if result.root_value <= config.resign_threshold:
                resign_low_streak[player_to_move] += 1
            else:
                resign_low_streak[player_to_move] = 0
            if resign_low_streak[player_to_move] >= config.resign_plies:
                resigned_winner = player_to_move.opposite().name
                break

        pending.append(
            {
                "state_planes": state_planes,
                "mask": canonicalize_action_vector(
                    torch.as_tensor(state_before.action_mask(), dtype=torch.bool),
                    player_to_move,
                ),
                "policy_target": canonicalize_action_vector(
                    torch.as_tensor(result.policy_target, dtype=torch.float32),
                    player_to_move,
                ),
                "value_target": 0.0,
                "legal_actions": legal_actions,
                "side_to_move": player_to_move.name,
                "action": int(action),
                "game_id": game_id,
                "ply": ply,
                "handicap": dict(handicap),
                "playout_cap": simulations,
            }
        )

        history.append(state_before)
        _, _, terminated, truncated, info = env.step(action)
        last_move = info.get("last_move") or {}
        acting_player = info.get("acting_player")
        if last_move.get("type") == "move" and acting_player in {"RED", "BLUE"}:
            position_key = "red_position" if acting_player == "RED" else "blue_position"
            pawn_visits.append((ply, acting_player, tuple(info[position_key])))
        root = mcts.advance_root(result.root, action)
        ply += 1

    # Finalize samples with game outcome. A resignation produces a decisive
    # (non-truncated) game so its samples are kept, with the conceding side's
    # positions labelled as losses.
    if resigned_winner is not None:
        winner = resigned_winner
        truncated = False
    else:
        winner = info.get("winner")
        if (
            config.adjudicate_repetition_draws
            and winner is None
            and not truncated
            and getattr(env.state, "is_draw", False)
        ):
            adjudicated = _adjudicate_repetition_draw(env.state)
            if adjudicated is not None:
                winner = adjudicated
    lead = info.get("lead")
    final_metadata = {
        "winner": winner,
        "lead": lead,
        "N_moves": info.get("N_moves"),
        "game_length": info.get("steps"),
        "truncated": bool(truncated),
        "resigned": resigned_winner is not None,
    }
    for sample in pending:
        sample.update(final_metadata)
        sample["value_target"] = _value_target(sample["side_to_move"], winner, truncated)
        sample["lead_target"] = float(lead) if lead is not None else 0.0
        sample["lead_mask"] = 1.0 if lead is not None else 0.0
        sample["future_map_target"] = _future_map_target(
            sample["side_to_move"], int(sample["ply"]), pawn_visits,
        )
        sample["future_map_mask"] = 1.0
        score_target, score_mask = _score_target(
            sample["side_to_move"],
            winner,
            bool(truncated),
            int(sample["ply"]),
            int(info.get("steps") or 0),
        )
        sample["score_target"] = score_target
        sample["score_mask"] = score_mask

    game_elapsed = time.time() - game_start
    game_info = {
        "winner": winner,
        "steps": info.get("steps"),
        "truncated": bool(truncated),
        "lead": lead,
        "handicap_mode": handicap.get("mode", "standard"),
        "ply": ply,
        "resigned": resigned_winner is not None,
    }
    return pending, game_elapsed, game_info


# ======================================================================
# Parallel self-play
# ======================================================================

def _clear_replay_iteration(output_dir: Path, iteration: int) -> int:
    removed = 0
    pattern = f"iter_{iteration:06d}_chunk_*.pt"
    for path in output_dir.glob(pattern):
        path.unlink()
        removed += 1
    if removed:
        print(
            f"[{_timestamp()}] [self-play] removed {removed} stale replay chunks "
            f"for iteration {iteration}",
            flush=True,
        )
    return removed


# Per-worker-process state, built once by ``_process_worker_init`` and reused for
# every game that worker runs. The model is loaded from a temp state-dict file so
# it is never pickled per task; weights are read-only during self-play.
_PROC_CTX: Dict[str, Any] = {}


def _process_worker_init(
    state_path: str,
    network_config: NetworkConfig,
    config: SelfPlayConfig,
    device: str,
    threads_per_worker: int,
) -> None:
    """Initializer run once per spawned worker: load the model and cap threads.

    Workers are spawned via forkserver/spawn (never plain fork), so they get a
    clean process and reconstruct the model from ``state_path`` rather than
    inheriting it — this avoids deadlocking on the training phase's hot OpenMP
    thread pool that a fork would carry over.
    """
    try:
        torch.set_num_threads(max(1, int(threads_per_worker)))
    except Exception:  # noqa: BLE001 - thread tuning is best-effort
        pass
    model = build_model(network_config)
    state_dict = torch.load(state_path, map_location="cpu", weights_only=False)
    _load_model_state(model, state_dict)
    model.to(torch.device(device))
    model.eval()
    _PROC_CTX.clear()
    _PROC_CTX.update(
        {
            "model": model,
            "network_config": network_config,
            "config": config,
            "device": device,
        }
    )


def _process_game_entry(
    game_index: int, iteration: int, seed: int
) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
    """Top-level entry run inside a worker process (one game, synchronous)."""
    return _run_single_game_worker(
        game_index,
        iteration,
        _PROC_CTX["model"],
        _PROC_CTX["network_config"],
        _PROC_CTX["config"],
        None,  # no shared inference server in the process path
        seed,
        device=_PROC_CTX["device"],
    )


def run_self_play(
    model: nn.Module,
    network_config: NetworkConfig,
    config: SelfPlayConfig,
    *,
    device: Optional[str] = None,
) -> List[Path]:
    """Run parallel self-play games.

    On CPU this uses true multi-process workers (each game runs synchronously in
    its own forked process, memory reclaimed per game) instead of the GPU-oriented
    thread+BatchInferenceServer design, which on CPU only made worker threads
    contend on the GIL while accumulating memory across long games.
    """
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_replay_iteration(output_dir, config.iteration)
    model.eval()

    num_workers = max(1, config.num_workers)
    if config.use_processes is None:
        use_processes = resolved_device.type == "cpu" and num_workers > 1
    else:
        use_processes = bool(config.use_processes)
    max_inference_batch = min(512, config.batch_size * num_workers * 2)
    mode = "process" if use_processes else "thread+server"

    print(
        f"[{_timestamp()}] [self-play] starting iteration {config.iteration}: "
        f"games={config.games} workers={num_workers} mode={mode} "
        f"simulations={config.base_simulations} mcts_batch={config.batch_size} "
        f"max_steps={config.max_steps} resign<={config.resign_threshold} "
        f"max_nn_batch={max_inference_batch} device={resolved_device}",
        flush=True,
    )

    all_samples: List[Dict[str, Any]] = []
    written_chunks: List[Path] = []
    chunk_index = 0
    completed_games = 0
    skipped_games = 0
    resigned_games = 0
    total_samples_generated = 0
    total_samples_collected = 0
    start_time = time.time()
    server: Optional[BatchInferenceServer] = None

    def _handle_completed(
        game_index: int,
        game_samples: List[Dict[str, Any]],
        game_elapsed: float,
        game_info: Dict[str, Any],
    ) -> None:
        nonlocal all_samples, chunk_index, completed_games, skipped_games
        nonlocal resigned_games, total_samples_generated, total_samples_collected

        total_samples_generated += len(game_samples)
        keep_game = _is_trainable_game(
            game_info.get("winner"), bool(game_info.get("truncated", False))
        )
        kept_samples = len(game_samples) if keep_game else 0
        if keep_game:
            all_samples.extend(game_samples)
            total_samples_collected += kept_samples
        else:
            skipped_games += 1
        if game_info.get("resigned"):
            resigned_games += 1
        completed_games += 1

        winner_str = game_info.get("winner") or "draw"
        steps = game_info.get("steps") or 0
        trunc = game_info.get("truncated", False)
        handicap_mode = game_info.get("handicap_mode", "?")
        resigned = bool(game_info.get("resigned", False))

        elapsed = time.time() - start_time
        games_per_sec = completed_games / max(elapsed, 0.001)
        samples_per_sec = total_samples_collected / max(elapsed, 0.001)
        mem_avail_mb, swap_used_mb = _system_mem_stats()
        mem_str = (
            f"mem_avail={mem_avail_mb:.0f}MB swap_used={swap_used_mb:.0f}MB"
            if mem_avail_mb is not None
            else "mem_avail=? swap_used=?"
        )
        nn_str = ""
        if server is not None:
            nn_stats = server.stats
            nn_str = (
                f"nn_avg_batch={nn_stats['avg_batch_size']:.1f} "
                f"nn_max_batch={nn_stats['max_batch_size']} "
            )

        print(
            f"[{_timestamp()}] [self-play] "
            f"game {completed_games}/{config.games} "
            f"(idx={game_index}) "
            f"winner={winner_str} steps={steps} "
            f"trunc={trunc} resigned={resigned} mode={handicap_mode} "
            f"samples={len(game_samples)} kept={kept_samples} "
            f"time={_format_duration(game_elapsed)} "
            f"| total_kept={total_samples_collected} "
            f"skipped_games={skipped_games} resigned_games={resigned_games} "
            f"games/s={games_per_sec:.2f} "
            f"samples/s={samples_per_sec:.1f} "
            f"{nn_str}{mem_str} "
            f"elapsed={_format_duration(elapsed)}",
            flush=True,
        )

        # Write chunks as they fill up.
        while len(all_samples) >= config.chunk_size:
            chunk_samples = all_samples[: config.chunk_size]
            all_samples = all_samples[config.chunk_size :]
            path = _write_replay_chunk(
                chunk_samples,
                output_dir,
                config.iteration,
                chunk_index,
                network_config,
                config,
            )
            written_chunks.append(path)
            chunk_index += 1

    def _drain(future_map: Dict[Any, int], done: set) -> bool:
        """Consume completed futures. Returns True if the pool broke mid-flight.

        ``done`` is updated in place with the game indices fully handled (so the
        caller can resubmit only the unfinished ones after a pool crash).
        """
        nonlocal completed_games
        for future in as_completed(future_map):
            game_index = future_map[future]
            try:
                game_samples, game_elapsed, game_info = future.result()
            except BrokenProcessPool:
                # A worker died abruptly (OOM/segfault): every in-flight future
                # now raises this. Stop draining and let the caller restart the
                # pool with the games that have not completed yet.
                return True
            except Exception as exc:  # noqa: BLE001 - log and keep other games
                print(
                    f"[{_timestamp()}] [self-play] ERROR game {game_index}: {exc}",
                    flush=True,
                )
                done.add(game_index)
                completed_games += 1
                continue
            done.add(game_index)
            _handle_completed(game_index, game_samples, game_elapsed, game_info)
        return False

    if use_processes:
        # Use forkserver/spawn — never plain ``fork``. Self-play runs right after
        # the training phase, which leaves THIS process's OpenMP/MKL thread pool
        # "hot"; forking from it makes the children deadlock on their first torch
        # op (inherited locked thread-pool state). A forkserver worker is spawned
        # from a clean server process and loads the current (just-trained) model
        # from a temp state-dict file, sidestepping the deadlock entirely.
        start_methods = mp.get_all_start_methods()
        ctx = mp.get_context(
            "forkserver" if "forkserver" in start_methods else "spawn"
        )
        state_fd, state_path = tempfile.mkstemp(
            suffix=".pt", prefix="selfplay_model_"
        )
        os.close(state_fd)
        torch.save(model.state_dict(), state_path)

        prev_threads = torch.get_num_threads()
        # Per-game attempt budget so a deterministically-crashing game cannot loop
        # forever (it would re-break each fresh pool); abandon it after N tries.
        max_attempts = 3
        attempts: Dict[int, int] = {gi: 0 for gi in range(config.games)}
        remaining = list(range(config.games))
        current_workers = num_workers
        try:
            torch.set_num_threads(1)
            while remaining:
                # Drop (once) any games that exhausted their attempt budget.
                runnable = []
                for gi in remaining:
                    if attempts[gi] >= max_attempts:
                        print(
                            f"[{_timestamp()}] [self-play] ABANDON game {gi}: "
                            f"crashed its worker {max_attempts} times",
                            flush=True,
                        )
                        completed_games += 1
                    else:
                        runnable.append(gi)
                remaining = runnable
                if not remaining:
                    break
                for gi in runnable:
                    attempts[gi] += 1

                workers = max(1, min(current_workers, len(runnable)))
                threads_per_worker = max(1, (os.cpu_count() or workers) // workers)
                executor = ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=ctx,
                    initializer=_process_worker_init,
                    initargs=(
                        state_path,
                        network_config,
                        config,
                        str(resolved_device),
                        threads_per_worker,
                    ),
                )
                done: set = set()
                try:
                    futures: Dict[Any, int] = {
                        executor.submit(
                            _process_game_entry,
                            game_index,
                            config.iteration,
                            config.seed + game_index + 1,
                        ): game_index
                        for game_index in runnable
                    }
                    broke = _drain(futures, done)
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)

                remaining = [gi for gi in remaining if gi not in done]
                if broke and remaining:
                    # Back off concurrency: a worker died (most likely OOM under
                    # peak tree memory), so retry the survivors with fewer
                    # parallel games to lower the memory high-water mark.
                    current_workers = max(1, workers // 2)
                    print(
                        f"[{_timestamp()}] [self-play] worker pool broke; "
                        f"restarting {len(remaining)} game(s) with "
                        f"{current_workers} worker(s)",
                        flush=True,
                    )
        finally:
            torch.set_num_threads(prev_threads)
            _PROC_CTX.clear()
            if state_path is not None:
                try:
                    os.unlink(state_path)
                except OSError:
                    pass
    else:
        collection_timeout = 0.001 if num_workers > 1 else 0.0
        server = BatchInferenceServer(
            model,
            resolved_device,
            max_batch_size=max_inference_batch,
            collection_timeout=collection_timeout,
        )
        server.start()
        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        _run_single_game_worker,
                        game_index,
                        config.iteration,
                        model,
                        network_config,
                        config,
                        server,
                        config.seed + game_index + 1,
                        device=str(resolved_device),
                    ): game_index
                    for game_index in range(config.games)
                }
                _drain(futures, set())
        finally:
            server.stop()

    # Write remaining samples.
    if all_samples:
        path = _write_replay_chunk(
            all_samples,
            output_dir,
            config.iteration,
            chunk_index,
            network_config,
            config,
        )
        written_chunks.append(path)

    elapsed = time.time() - start_time
    nn_tail = ""
    if server is not None:
        nn_stats = server.stats
        nn_tail = (
            f" nn_batches={nn_stats['total_batches']} "
            f"nn_inferences={nn_stats['total_inferences']} "
            f"nn_avg_batch={nn_stats['avg_batch_size']:.1f} "
            f"nn_max_batch={nn_stats['max_batch_size']}"
        )
    print(
        f"[{_timestamp()}] [self-play] DONE iteration {config.iteration}: "
        f"games={completed_games} chunks={len(written_chunks)} "
        f"skipped_games={skipped_games} resigned_games={resigned_games} "
        f"generated_samples={total_samples_generated} "
        f"kept_samples={total_samples_collected} "
        f"elapsed={_format_duration(elapsed)} "
        f"games/s={completed_games / max(elapsed, 0.001):.2f}"
        f"{nn_tail}",
        flush=True,
    )

    return written_chunks


# ======================================================================
# Training
# ======================================================================

def train_from_replay(
    model: nn.Module,
    network_config: NetworkConfig,
    config: TrainConfig,
    *,
    checkpoint_path: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> Path:
    _set_seed(config.seed)
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(resolved_device)

    if config.replay_window_size is not None and config.replay_window_size > 0:
        # Anchor the window at the explicit max iteration if given, otherwise at
        # the iteration currently being trained. The window is [anchor-N, anchor],
        # where N is the number of previous iterations to include. Chunks older
        # than the window and any stray future chunks are skipped.
        replay_window_anchor = (
            config.replay_max_iteration
            if config.replay_max_iteration is not None
            else config.iteration
        )
        replay_min_iteration: Optional[int] = max(
            1, replay_window_anchor - config.replay_window_size
        )
        replay_max_iteration: Optional[int] = replay_window_anchor
    else:
        replay_min_iteration = None
        replay_max_iteration = config.replay_max_iteration
    dataset = ReplayDataset(
        config.replay_dir,
        config.replay_limit,
        max_iteration=replay_max_iteration,
        min_iteration=replay_min_iteration,
    )
    use_pin_memory = resolved_device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=use_pin_memory,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    if checkpoint_path:
        _load_checkpoint_into(
            model, checkpoint_path, optimizer=optimizer, device=resolved_device
        )

    print(
        f"[{_timestamp()}] [train] starting iteration {config.iteration}: "
        f"samples={len(dataset)} epochs={config.epochs} "
        f"filtered_samples={dataset.filtered_samples} "
        f"filtered_policy_samples={dataset.filtered_policy_samples} "
        f"filtered_policy_games={dataset.filtered_policy_games} "
        f"skipped_future_chunks={dataset.skipped_future_chunks} "
        f"skipped_old_chunks={dataset.skipped_old_chunks} "
        f"replay_min_iteration={dataset.replay_min_iteration} "
        f"replay_max_iteration={dataset.replay_max_iteration} "
        f"batch_size={config.batch_size} lr={config.learning_rate} "
        f"device={resolved_device}",
        flush=True,
    )

    stats = []
    model.train()
    train_start = time.time()

    for epoch in range(config.epochs):
        epoch_start = time.time()
        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        lead_loss_sum = 0.0
        future_loss_sum = 0.0
        score_loss_sum = 0.0
        batches = 0
        num_batches = len(loader)
        print_interval = max(1, num_batches // 10)

        for batch_idx, (
            state_planes,
            mask,
            policy_target,
            value_target,
            lead_target,
            lead_mask,
            future_map_target,
            future_map_mask,
            score_target,
            score_mask,
        ) in enumerate(loader):
            state_planes = state_planes.to(resolved_device, non_blocking=True)
            mask = mask.to(resolved_device, non_blocking=True)
            policy_target = policy_target.to(resolved_device, non_blocking=True)
            value_target = value_target.to(resolved_device, non_blocking=True)
            lead_target = lead_target.to(resolved_device, non_blocking=True)
            lead_mask = lead_mask.to(resolved_device, non_blocking=True)
            future_map_target = future_map_target.to(resolved_device, non_blocking=True)
            future_map_mask = future_map_mask.to(resolved_device, non_blocking=True)
            score_target = score_target.to(resolved_device, non_blocking=True)
            score_mask = score_mask.to(resolved_device, non_blocking=True)

            logits, value, lead, future_logits, score = model(state_planes)
            logits = masked_logits(logits, mask)
            policy_loss = soft_target_cross_entropy(logits, policy_target)
            value_loss = F.mse_loss(value.squeeze(-1), value_target)
            lead_loss = symlog_squared_error(lead.squeeze(-1), lead_target, lead_mask)
            future_loss = masked_binary_cross_entropy(
                future_logits, future_map_target, future_map_mask,
            )
            score_loss = masked_smooth_l1_loss(
                score.squeeze(-1), symlog(score_target), score_mask,
            )
            loss = (
                policy_loss
                + config.value_loss_weight * value_loss
                + config.lead_loss_weight * lead_loss
                + config.future_loss_weight * future_loss
                + config.score_loss_weight * score_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            total_loss_sum += float(loss.item())
            policy_loss_sum += float(policy_loss.item())
            value_loss_sum += float(value_loss.item())
            lead_loss_sum += float(lead_loss.item())
            future_loss_sum += float(future_loss.item())
            score_loss_sum += float(score_loss.item())
            batches += 1

            if (batch_idx + 1) % print_interval == 0 or batch_idx + 1 == num_batches:
                epoch_elapsed = time.time() - epoch_start

        epoch_stats = {
            "epoch": epoch,
            "total_loss": total_loss_sum / max(1, batches),
            "policy_loss": policy_loss_sum / max(1, batches),
            "value_loss": value_loss_sum / max(1, batches),
            "lead_loss": lead_loss_sum / max(1, batches),
            "future_loss": future_loss_sum / max(1, batches),
            "score_loss": score_loss_sum / max(1, batches),
            "batches": batches,
            "samples": len(dataset),
        }
        stats.append(epoch_stats)
        epoch_elapsed = time.time() - epoch_start
        print(
            f"[{_timestamp()}] [train] epoch={epoch} done "
            f"avg_total={epoch_stats['total_loss']:.4f} "
            f"avg_policy={epoch_stats['policy_loss']:.4f} "
            f"avg_value={epoch_stats['value_loss']:.4f} "
            f"avg_lead={epoch_stats['lead_loss']:.4f} "
            f"avg_future={epoch_stats['future_loss']:.4f} "
            f"avg_score={epoch_stats['score_loss']:.4f} "
            f"elapsed={_format_duration(epoch_elapsed)}",
            flush=True,
        )

    train_elapsed = time.time() - train_start
    print(
        f"[{_timestamp()}] [train] DONE iteration {config.iteration}: "
        f"elapsed={_format_duration(train_elapsed)}",
        flush=True,
    )

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path = checkpoint_dir / f"model_iter_{config.iteration:06d}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "network_config": asdict(network_config),
            "train_config": asdict(config),
            "training_stats": stats,
        },
        output_path,
    )
    print(f"[{_timestamp()}] [train] checkpoint saved: {output_path}", flush=True)
    return output_path


# ======================================================================
# Evaluation
# ======================================================================

def evaluate_models(
    candidate_model: nn.Module,
    baseline_model: nn.Module,
    network_config: NetworkConfig,
    config: EvalConfig,
    *,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    rng = random.Random(config.seed)
    candidate_model.eval()
    baseline_model.eval()

    wins = 0
    losses = 0
    draws = 0
    leads: List[float] = []
    game_lengths: List[int] = []
    red_pawn_moves: List[int] = []
    blue_pawn_moves: List[int] = []

    print(
        f"[{_timestamp()}] [eval] starting: games={config.games} "
        f"simulations={config.simulations} device={device or 'auto'}",
        flush=True,
    )

    for game_index in range(config.games):
        game_start = time.time()
        candidate_player = Player.RED if game_index % 2 == 0 else Player.BLUE
        env = BarricadeEnv(max_steps=config.max_steps, invalid_action_mode="raise")
        handicap = sample_valid_handicap(config.base_walls, rng)
        _, info = env.reset(options=_env_options(handicap))
        history = []
        roots = {Player.RED: None, Player.BLUE: None}
        terminated = False
        truncated = False

        mcts_config = MCTSConfig(
            num_simulations=config.simulations,
            batch_size=config.batch_size,
            history_length=network_config.history_length,
            device=device,
            add_root_noise=False,
            action_temperature=0.0,
            # Match self-play: without this, eval games shuffle into repetition
            # draws (MCTS can't explore the winning pawn moves), so every
            # candidate scores 50% and is rejected.
            fpu_reduction=0.0,
        )

        while not terminated and not truncated:
            current = env.state.current_player
            model = candidate_model if current == candidate_player else baseline_model
            state_before = env.state.copy()
            history_before = _history_window(history, network_config.history_length)
            mcts = MCTS(
                model,
                mcts_config,
                policy_action_transform=policy_action_for_mcts,
            )
            result = mcts.search(state_before, history=history_before, root=roots[current])
            action = mcts.select_action(result, temperature=0.0)
            history.append(state_before)
            _, _, terminated, truncated, info = env.step(action)
            roots[current] = mcts.advance_root(result.root, action)
            roots[current.opposite()] = None

        winner = info.get("winner")
        if winner is None and getattr(env.state, "is_draw", False):
            adjudicated = _adjudicate_repetition_draw(env.state)
            if adjudicated is not None:
                winner = adjudicated
        if winner is None:
            draws += 1
            result_str = "draw"
        elif winner == candidate_player.name:
            wins += 1
            result_str = "win"
        else:
            losses += 1
            result_str = "loss"

        if info.get("lead") is not None:
            leads.append(float(info["lead"]))
        game_lengths.append(int(info.get("steps") or 0))
        n_moves = info.get("N_moves") or {}
        red_pawn_moves.append(int(n_moves.get("RED", 0)))
        blue_pawn_moves.append(int(n_moves.get("BLUE", 0)))

        game_elapsed = time.time() - game_start
        total = wins + losses + draws
        running_win_rate = (wins + 0.5 * draws) / max(1, total)
        print(
            f"[{_timestamp()}] [eval]   "
            f"game {game_index + 1}/{config.games} "
            f"result={result_str} steps={info.get('steps', 0)} "
            f"candidate={candidate_player.name} "
            f"time={_format_duration(game_elapsed)} "
            f"| W={wins} L={losses} D={draws} "
            f"win_rate={running_win_rate:.1%}",
            flush=True,
        )

    total = max(1, config.games)
    metrics = {
        "games": config.games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": (wins + 0.5 * draws) / total,
        "average_lead": _mean(leads),
        "average_game_length": _mean(game_lengths),
        "average_red_pawn_moves": _mean(red_pawn_moves),
        "average_blue_pawn_moves": _mean(blue_pawn_moves),
    }
    print(
        f"[{_timestamp()}] [eval] DONE: "
        f"W={wins} L={losses} D={draws} win_rate={metrics['win_rate']:.1%} "
        f"avg_lead={metrics['average_lead']:.3f} "
        f"avg_steps={metrics['average_game_length']:.1f}",
        flush=True,
    )
    return metrics


# ======================================================================
# Main loop
# ======================================================================

def run_loop(
    model: nn.Module,
    network_config: NetworkConfig,
    *,
    iterations: int,
    starting_iter: int = 1,
    self_play_config: SelfPlayConfig,
    train_config: TrainConfig,
    eval_config: EvalConfig,
    checkpoint_path: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> None:
    current_checkpoint = Path(checkpoint_path) if checkpoint_path else None
    if starting_iter < 1:
        raise ValueError("--starting-iter must be >= 1.")
    if iterations < 1:
        raise ValueError("--iterations must be >= 1.")

    final_iteration = starting_iter + iterations - 1
    for iteration in range(starting_iter, final_iteration + 1):
        iter_start = time.time()
        print(
            f"\n{'=' * 70}\n"
            f"[{_timestamp()}] [loop] iteration {iteration}/{final_iteration}\n"
            f"{'=' * 70}",
            flush=True,
        )

        # --- Self-play ---
        print(f"[{_timestamp()}] [loop] >>> self-play phase", flush=True)
        sp_config = replace(self_play_config, iteration=iteration)
        sp_start = time.time()
        run_self_play(model, network_config, sp_config, device=device)
        sp_elapsed = time.time() - sp_start
        print(
            f"[{_timestamp()}] [loop] self-play phase done "
            f"in {_format_duration(sp_elapsed)}",
            flush=True,
        )

        # --- Training ---
        print(f"[{_timestamp()}] [loop] >>> training phase", flush=True)
        replay_max_iteration = (
            train_config.replay_max_iteration
            if train_config.replay_max_iteration is not None
            else iteration
        )
        tr_config = replace(
            train_config,
            iteration=iteration,
            replay_max_iteration=replay_max_iteration,
        )
        tr_start = time.time()
        candidate_path = train_from_replay(
            model,
            network_config,
            tr_config,
            checkpoint_path=current_checkpoint,
            device=device,
        )
        tr_elapsed = time.time() - tr_start
        print(
            f"[{_timestamp()}] [loop] training phase done "
            f"in {_format_duration(tr_elapsed)}",
            flush=True,
        )

        # --- Evaluation (skip first iteration) ---
        if current_checkpoint is None:
            current_checkpoint = candidate_path
            _copy_latest(candidate_path)
            iter_elapsed = time.time() - iter_start
            print(
                f"[{_timestamp()}] [loop] iteration {iteration} complete "
                f"in {_format_duration(iter_elapsed)} (no eval — first iteration)",
                flush=True,
            )
            continue

        print(f"[{_timestamp()}] [loop] >>> evaluation phase", flush=True)
        baseline = build_model(network_config)
        _load_checkpoint_into(
            baseline, current_checkpoint, device=torch.device(device or "cpu")
        )
        eval_start = time.time()
        metrics = evaluate_models(
            model, baseline, network_config, eval_config, device=device
        )
        eval_elapsed = time.time() - eval_start
        print(
            f"[{_timestamp()}] [loop] evaluation phase done "
            f"in {_format_duration(eval_elapsed)} "
            f"win_rate={metrics['win_rate']:.1%}",
            flush=True,
        )

        if metrics["win_rate"] >= 0.55:
            current_checkpoint = candidate_path
            _copy_latest(candidate_path)
            print(
                f"[{_timestamp()}] [loop] candidate PROMOTED "
                f"(win_rate={metrics['win_rate']:.1%} >= 0.55)",
                flush=True,
            )
        else:
            _load_checkpoint_into(
                model, current_checkpoint, device=torch.device(device or "cpu")
            )
            print(
                f"[{_timestamp()}] [loop] candidate REJECTED "
                f"(win_rate={metrics['win_rate']:.1%} < 0.55), "
                f"reverting to previous checkpoint",
                flush=True,
            )

        iter_elapsed = time.time() - iter_start
        print(
            f"[{_timestamp()}] [loop] iteration {iteration} complete "
            f"in {_format_duration(iter_elapsed)}\n",
            flush=True,
        )


# ======================================================================
# Loss functions
# ======================================================================

def soft_target_cross_entropy(logits: Tensor, target: Tensor) -> Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(target * log_probs).sum(dim=1).mean()


def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def masked_mse_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    per_sample = F.mse_loss(prediction, target, reduction="none")
    return _masked_mean(per_sample, mask, prediction)


def symlog_squared_error(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    per_sample = F.mse_loss(prediction, symlog(target), reduction="none")
    return _masked_mean(per_sample, mask, prediction)


def masked_smooth_l1_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    per_sample = F.smooth_l1_loss(prediction, target, reduction="none")
    return _masked_mean(per_sample, mask, prediction)


def masked_binary_cross_entropy(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    per_cell = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    per_sample = per_cell.mean(dim=(1, 2, 3))
    return _masked_mean(per_sample, mask, logits)


def _masked_mean(per_sample: Tensor, mask: Tensor, fallback: Tensor) -> Tensor:
    mask = mask.to(device=per_sample.device, dtype=per_sample.dtype).view_as(per_sample)
    denominator = mask.sum()
    if float(denominator.item()) <= 0.0:
        return fallback.sum() * 0.0
    return (per_sample * mask).sum() / denominator


# ======================================================================
# Model helpers
# ======================================================================

def build_model(config: NetworkConfig) -> nn.Module:
    return build_network(
        history_length=config.history_length,
        conv_channels=config.conv_channels,
        residual_channels=config.residual_channels,
        num_conv_layers=config.num_conv_layers,
        num_residual_layers=config.num_residual_layers,
        value_hidden_size=config.value_hidden_size,
    )


def load_model_from_checkpoint(
    path: str | Path, device: Optional[str] = None
) -> Tuple[nn.Module, NetworkConfig]:
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    network_config = NetworkConfig(**payload.get("network_config", {}))
    model = build_model(network_config)
    _load_model_state(model, payload["model_state"])
    if device:
        model.to(torch.device(device))
    return model, network_config


def _load_checkpoint_into(
    model: nn.Module,
    path: str | Path,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    _load_model_state(model, payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        try:
            optimizer.load_state_dict(payload["optimizer_state"])
        except ValueError:
            pass
    return payload


def _load_model_state(model: nn.Module, state_dict: Dict[str, Tensor]) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    _initialize_missing_auxiliary_outputs(model, incompatible.missing_keys)


def _initialize_missing_auxiliary_outputs(
    model: nn.Module,
    missing_keys: Sequence[str],
) -> None:
    if not any(
        key.startswith(("lead_head.", "future_map_head.", "score_head."))
        for key in missing_keys
    ):
        return

    for module_name in ("lead_head.fc2", "future_map_head.conv2", "score_head.fc2"):
        module = model.get_submodule(module_name)
        if hasattr(module, "weight"):
            module.weight.data.zero_()
        if getattr(module, "bias", None) is not None:
            module.bias.data.zero_()


# ======================================================================
# Replay writing
# ======================================================================

def _write_replay_chunk(
    samples: Sequence[Dict[str, Any]],
    output_dir: Path,
    iteration: int,
    chunk_index: int,
    network_config: NetworkConfig,
    self_play_config: SelfPlayConfig,
) -> Path:
    path = output_dir / f"iter_{iteration:06d}_chunk_{chunk_index:03d}.pt"
    torch.save(
        {
            "samples": list(samples),
            "network_config": asdict(network_config),
            "self_play_config": asdict(self_play_config),
            "created_at": time.time(),
        },
        path,
    )
    print(
        f"[{_timestamp()}] [self-play] wrote chunk {chunk_index}: "
        f"{len(samples)} samples -> {path.name}",
        flush=True,
    )
    return path


# ======================================================================
# Target computation helpers
# ======================================================================

def _adjudicate_repetition_draw(state: Any) -> Optional[str]:
    """Award a threefold-repetition draw to whoever is ahead in the race.

    Self-play between two equal models tends to shuffle into repetition draws,
    whose value target (0.0) gives the value head no signal — it goes flat and
    MCTS degenerates into following the policy. Adjudicating a lopsided draw by
    race lead (shorter shortest-path, then more walls remaining) yields a
    decisive signal that teaches racing. Genuinely balanced positions (equal
    path AND equal walls) remain true draws.
    """
    red = state.shortest_path_length(Player.RED)
    blue = state.shortest_path_length(Player.BLUE)
    if red is None or blue is None:
        return None
    if red != blue:
        return Player.RED.name if red < blue else Player.BLUE.name
    red_walls = state.walls_left[Player.RED]
    blue_walls = state.walls_left[Player.BLUE]
    if red_walls != blue_walls:
        return Player.RED.name if red_walls > blue_walls else Player.BLUE.name
    return None


def _value_target(side_to_move: str, winner: Optional[str], truncated: bool) -> float:
    if winner is None:
        return 0.0
    return 1.0 if side_to_move == winner else -1.0


def _future_map_target(
    side_to_move: str,
    ply: int,
    pawn_visits: Sequence[Tuple[int, str, Tuple[int, int]]],
) -> Tensor:
    target = torch.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
    for visit_ply, player, position in pawn_visits:
        if visit_ply < ply:
            continue
        row, col = position
        channel = 0 if player == side_to_move else 1
        target[channel, int(row), int(col)] = 1.0
    return target


def _score_target(
    side_to_move: str,
    winner: Optional[str],
    truncated: bool,
    ply: int,
    terminal_steps: int,
) -> Tuple[float, float]:
    if truncated or winner is None or side_to_move != winner:
        return 0.0, 0.0
    return float(max(0, terminal_steps - ply)), 1.0


def _history_window(history: Sequence[Any], history_length: int) -> Tuple[Any, ...]:
    if history_length <= 0:
        return ()
    return tuple(history[-history_length:])


def _env_options(handicap: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "red_start": handicap["red_start"],
        "blue_start": handicap["blue_start"],
        "red_walls": handicap["red_walls"],
        "blue_walls": handicap["blue_walls"],
        "starting_player": handicap["starting_player"],
    }


def _copy_latest(checkpoint_path: Path) -> None:
    latest_path = checkpoint_path.parent / "latest.pt"
    shutil.copy2(checkpoint_path, latest_path)
    print(f"[{_timestamp()}] promoted {checkpoint_path} -> {latest_path}", flush=True)


def _mean(values: Sequence[float | int]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / len(values)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Barricade AlphaZero training pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_network_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--history-length", type=int, default=4)
        p.add_argument("--conv-channels", type=int, default=64)
        p.add_argument("--residual-channels", type=int, default=None)
        p.add_argument("--num-conv-layers", type=int, default=1)
        p.add_argument("--num-residual-layers", type=int, default=6)
        p.add_argument("--value-hidden-size", type=int, default=128)
        p.add_argument("--checkpoint", type=str, default=None)
        p.add_argument("--device", type=str, default=None)

    def add_selfplay_runtime_args(p: argparse.ArgumentParser) -> None:
        """Resignation + worker-mode knobs shared by the self-play and loop commands."""
        p.add_argument(
            "--resign-threshold",
            type=float,
            default=-0.85,
            help="Resign when root value <= this for --resign-plies turns "
            "(set to -1.0 to effectively disable).",
        )
        p.add_argument("--resign-plies", type=int, default=6)
        p.add_argument("--resign-disable-fraction", type=float, default=0.1)
        proc = p.add_mutually_exclusive_group()
        proc.add_argument(
            "--use-processes",
            dest="use_processes",
            action="store_true",
            default=None,
            help="Force multi-process self-play workers (default: auto on CPU).",
        )
        proc.add_argument(
            "--no-use-processes",
            dest="use_processes",
            action="store_false",
            help="Force the thread+inference-server path even on CPU.",
        )

    # --- self-play ---
    self_play = subparsers.add_parser("self-play")
    add_network_args(self_play)
    self_play.add_argument("--iteration", type=int, default=1)
    self_play.add_argument("--games", type=int, default=16)
    self_play.add_argument("--base-simulations", type=int, default=128)
    self_play.add_argument("--mcts-batch-size", type=int, default=16)
    self_play.add_argument("--max-steps", type=int, default=100)
    self_play.add_argument("--chunk-size", type=int, default=2048)
    self_play.add_argument("--temperature-drop-ply", type=int, default=20)
    self_play.add_argument("--seed", type=int, default=1)
    self_play.add_argument("--output-dir", type=str, default=str(SELFPLAY_DIR))
    self_play.add_argument("--walls", type=int, default=DEFAULT_WALLS_PER_PLAYER)
    self_play.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel game workers for self-play",
    )
    add_selfplay_runtime_args(self_play)

    # --- train ---
    train = subparsers.add_parser("train")
    add_network_args(train)
    train.add_argument("--iteration", type=int, default=1)
    train.add_argument("--starting-iter", type=int, default=None)
    train.add_argument("--replay-dir", type=str, default=str(SELFPLAY_DIR))
    train.add_argument("--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR))
    train.add_argument("--replay-limit", type=int, default=None)
    train.add_argument("--replay-max-iteration", type=int, default=None)
    train.add_argument(
        "--replay-window-size",
        type=int,
        default=1,
        help="Train on the current iteration plus the previous N iterations "
        "of self-play data (default: 1; 0=unbounded).",
    )
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--train-batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--value-loss-weight", type=float, default=1.0)
    train.add_argument("--lead-loss-weight", type=float, default=0.1)
    train.add_argument("--future-loss-weight", type=float, default=0.1)
    train.add_argument("--score-loss-weight", type=float, default=0.01)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=1)

    # --- eval ---
    eval_parser = subparsers.add_parser("eval")
    add_network_args(eval_parser)
    eval_parser.add_argument("--baseline-checkpoint", type=str, required=True)
    eval_parser.add_argument("--games", type=int, default=10)
    eval_parser.add_argument("--simulations", type=int, default=64)
    eval_parser.add_argument("--mcts-batch-size", type=int, default=16)
    eval_parser.add_argument("--max-steps", type=int, default=500)
    eval_parser.add_argument("--seed", type=int, default=1)
    eval_parser.add_argument("--walls", type=int, default=DEFAULT_WALLS_PER_PLAYER)

    # --- loop ---
    loop = subparsers.add_parser("loop")
    add_network_args(loop)
    loop.add_argument("--iterations", type=int, default=1)
    loop.add_argument("--starting-iter", type=int, default=1)
    loop.add_argument("--games", type=int, default=16)
    loop.add_argument("--base-simulations", type=int, default=128)
    loop.add_argument("--mcts-batch-size", type=int, default=16)
    loop.add_argument("--max-steps", type=int, default=100)
    loop.add_argument("--chunk-size", type=int, default=2048)
    loop.add_argument("--temperature-drop-ply", type=int, default=20)
    loop.add_argument("--replay-dir", type=str, default=str(SELFPLAY_DIR))
    loop.add_argument("--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR))
    loop.add_argument("--replay-limit", type=int, default=None)
    loop.add_argument("--replay-max-iteration", type=int, default=None)
    loop.add_argument(
        "--replay-window-size",
        type=int,
        default=1,
        help="Train on the current iteration plus the previous N iterations "
        "of self-play data (default: 1; 0=unbounded).",
    )
    loop.add_argument("--epochs", type=int, default=1)
    loop.add_argument("--train-batch-size", type=int, default=256)
    loop.add_argument("--learning-rate", type=float, default=1e-3)
    loop.add_argument("--value-loss-weight", type=float, default=1.0)
    loop.add_argument("--lead-loss-weight", type=float, default=0.1)
    loop.add_argument("--future-loss-weight", type=float, default=0.1)
    loop.add_argument("--score-loss-weight", type=float, default=0.01)
    loop.add_argument("--grad-clip", type=float, default=1.0)
    loop.add_argument("--weight-decay", type=float, default=1e-4)
    loop.add_argument("--eval-games", type=int, default=10)
    loop.add_argument("--eval-simulations", type=int, default=64)
    loop.add_argument(
        "--seed", type=int, default=3407
    )  # Torch.manual_seed(3407) is all you need.
    loop.add_argument("--walls", type=int, default=DEFAULT_WALLS_PER_PLAYER)
    loop.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel game workers for self-play",
    )
    add_selfplay_runtime_args(loop)

    return parser.parse_args()


def network_config_from_args(args: argparse.Namespace) -> NetworkConfig:
    return NetworkConfig(
        history_length=args.history_length,
        conv_channels=args.conv_channels,
        residual_channels=args.residual_channels,
        num_conv_layers=args.num_conv_layers,
        num_residual_layers=args.num_residual_layers,
        value_hidden_size=args.value_hidden_size,
    )


def main() -> None:
    args = parse_args()
    if args.command in {"train", "loop"}:
        _start_train_log_capture()

    network_config = network_config_from_args(args)
    _set_seed(getattr(args, "seed", 1))

    if args.command == "self-play":
        model = build_model(network_config)
        if args.checkpoint:
            _load_checkpoint_into(
                model, args.checkpoint, device=torch.device(args.device or "cpu")
            )
        config = SelfPlayConfig(
            iteration=args.iteration,
            games=args.games,
            base_simulations=args.base_simulations,
            batch_size=args.mcts_batch_size,
            max_steps=args.max_steps,
            chunk_size=args.chunk_size,
            temperature_drop_ply=args.temperature_drop_ply,
            seed=args.seed,
            output_dir=args.output_dir,
            base_walls=args.walls,
            num_workers=args.num_workers,
            resign_threshold=args.resign_threshold,
            resign_plies=args.resign_plies,
            resign_disable_fraction=args.resign_disable_fraction,
            use_processes=args.use_processes,
        )
        run_self_play(model, network_config, config, device=args.device)

    elif args.command == "train":
        model = build_model(network_config)
        train_iteration = (
            args.starting_iter if args.starting_iter is not None else args.iteration
        )
        if train_iteration < 1:
            raise ValueError("--starting-iter/--iteration must be >= 1.")
        if args.replay_window_size is not None and args.replay_window_size < 0:
            raise ValueError("--replay-window-size must be >= 0.")
        config = TrainConfig(
            iteration=train_iteration,
            replay_dir=args.replay_dir,
            checkpoint_dir=args.checkpoint_dir,
            replay_limit=args.replay_limit,
            replay_max_iteration=args.replay_max_iteration,
            replay_window_size=args.replay_window_size,
            epochs=args.epochs,
            batch_size=args.train_batch_size,
            learning_rate=args.learning_rate,
            value_loss_weight=args.value_loss_weight,
            lead_loss_weight=args.lead_loss_weight,
            future_loss_weight=args.future_loss_weight,
            score_loss_weight=args.score_loss_weight,
            grad_clip=args.grad_clip,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
        train_from_replay(
            model,
            network_config,
            config,
            checkpoint_path=args.checkpoint,
            device=args.device,
        )

    elif args.command == "eval":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for eval candidate model.")
        candidate, candidate_config = load_model_from_checkpoint(
            args.checkpoint, args.device
        )
        baseline, _ = load_model_from_checkpoint(args.baseline_checkpoint, args.device)
        config = EvalConfig(
            games=args.games,
            simulations=args.simulations,
            batch_size=args.mcts_batch_size,
            max_steps=args.max_steps,
            seed=args.seed,
            base_walls=args.walls,
        )
        metrics = evaluate_models(
            candidate, baseline, candidate_config, config, device=args.device
        )
        print(metrics)

    elif args.command == "loop":
        if args.starting_iter < 1:
            raise ValueError("--starting-iter must be >= 1.")
        if args.iterations < 1:
            raise ValueError("--iterations must be >= 1.")
        if args.replay_window_size is not None and args.replay_window_size < 0:
            raise ValueError("--replay-window-size must be >= 0.")
        model = build_model(network_config)
        if args.checkpoint:
            _load_checkpoint_into(
                model,
                args.checkpoint,
                device=torch.device(args.device or "cpu"),
            )
        self_play_config = SelfPlayConfig(
            games=args.games,
            base_simulations=args.base_simulations,
            batch_size=args.mcts_batch_size,
            max_steps=args.max_steps,
            chunk_size=args.chunk_size,
            temperature_drop_ply=args.temperature_drop_ply,
            seed=args.seed,
            output_dir=args.replay_dir,
            base_walls=args.walls,
            num_workers=args.num_workers,
            resign_threshold=args.resign_threshold,
            resign_plies=args.resign_plies,
            resign_disable_fraction=args.resign_disable_fraction,
            use_processes=args.use_processes,
        )
        train_config = TrainConfig(
            replay_dir=args.replay_dir,
            checkpoint_dir=args.checkpoint_dir,
            replay_limit=args.replay_limit,
            replay_max_iteration=args.replay_max_iteration,
            replay_window_size=args.replay_window_size,
            epochs=args.epochs,
            batch_size=args.train_batch_size,
            learning_rate=args.learning_rate,
            value_loss_weight=args.value_loss_weight,
            lead_loss_weight=args.lead_loss_weight,
            future_loss_weight=args.future_loss_weight,
            score_loss_weight=args.score_loss_weight,
            grad_clip=args.grad_clip,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
        eval_config = EvalConfig(
            games=args.eval_games,
            simulations=args.eval_simulations,
            batch_size=args.mcts_batch_size,
            max_steps=args.max_steps,
            seed=args.seed,
            base_walls=args.walls,
        )
        run_loop(
            model,
            network_config,
            iterations=args.iterations,
            starting_iter=args.starting_iter,
            self_play_config=self_play_config,
            train_config=train_config,
            eval_config=eval_config,
            checkpoint_path=args.checkpoint,
            device=args.device,
        )


if __name__ == "__main__":
    main()
