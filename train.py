"""
Self-play and training pipeline for Barricade AlphaZero experiments.

Supports parallel self-play with cross-worker batched neural-network inference
via a background BatchInferenceServer thread.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
import queue
from pathlib import Path
import random
import shutil
import sys
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
    max_steps: int = 500
    chunk_size: int = 2048
    temperature_drop_ply: int = 20
    seed: int = 1
    output_dir: str = str(SELFPLAY_DIR)
    base_walls: int = DEFAULT_WALLS_PER_PLAYER
    num_workers: int = 4


@dataclass(frozen=True)
class TrainConfig:
    iteration: int = 1
    replay_dir: str = str(SELFPLAY_DIR)
    checkpoint_dir: str = str(CHECKPOINT_DIR)
    replay_limit: Optional[int] = None
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

class ReplayDataset(Dataset):
    def __init__(self, replay_dir: str | Path, limit: Optional[int] = None) -> None:
        self.replay_dir = Path(replay_dir)
        self.samples: List[Dict[str, Any]] = []
        for path in sorted(self.replay_dir.glob("*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.samples.extend(payload.get("samples", []))
        if limit is not None and limit > 0 and len(self.samples) > limit:
            self.samples = self.samples[-limit:]
        if not self.samples:
            raise ValueError(f"No replay samples found in {self.replay_dir}")

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
    inference_server: BatchInferenceServer,
    seed: int,
) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
    """Run one complete self-play game.

    Returns ``(samples, elapsed_seconds, game_info)``.
    """
    rng = random.Random(seed)
    game_id = f"iter_{iteration:06d}_game_{game_index:06d}"
    game_start = time.time()

    handicap = sample_valid_handicap(config.base_walls, rng)
    env = BarricadeEnv(max_steps=config.max_steps)
    _, info = env.reset(options=_env_options(handicap))
    history: List[Any] = []
    pending: List[Dict[str, Any]] = []
    pawn_visits: List[Tuple[int, str, Tuple[int, int]]] = []
    root = None
    terminated = False
    truncated = False
    ply = 0

    # Create MCTS once per game and update config per-move via replace()
    mcts_config = MCTSConfig(
        num_simulations=config.base_simulations,
        batch_size=config.batch_size,
        history_length=network_config.history_length,
        device=str(inference_server.device),
        add_root_noise=True,
        action_temperature=1.0,
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
        temp = 1.0 if ply < config.temperature_drop_ply else 0.0
        mcts.config = replace(
            mcts.config,
            num_simulations=simulations,
            action_temperature=temp,
        )

        result = mcts.search(state_before, history=history_before, root=root)
        action = mcts.select_action(result, temperature=temp)

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
                "legal_actions": list(env.legal_actions()),
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

    # Finalize samples with game outcome
    winner = info.get("winner")
    lead = info.get("lead")
    final_metadata = {
        "winner": winner,
        "lead": lead,
        "N_moves": info.get("N_moves"),
        "game_length": info.get("steps"),
        "truncated": bool(truncated),
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
    }
    return pending, game_elapsed, game_info


# ======================================================================
# Parallel self-play
# ======================================================================

def run_self_play(
    model: nn.Module,
    network_config: NetworkConfig,
    config: SelfPlayConfig,
    *,
    device: Optional[str] = None,
) -> List[Path]:
    """Run parallel self-play games with cross-worker batched inference."""
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    rng = random.Random(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    num_workers = max(1, config.num_workers)
    max_inference_batch = min(512, config.batch_size * num_workers * 2)

    print(
        f"[{_timestamp()}] [self-play] starting iteration {config.iteration}: "
        f"games={config.games} workers={num_workers} "
        f"simulations={config.base_simulations} mcts_batch={config.batch_size} "
        f"max_nn_batch={max_inference_batch} device={resolved_device}",
        flush=True,
    )

    # Start the shared inference server
    collection_timeout = 0.001 if num_workers > 1 else 0.0
    server = BatchInferenceServer(
        model,
        resolved_device,
        max_batch_size=max_inference_batch,
        collection_timeout=collection_timeout,
    )
    server.start()

    all_samples: List[Dict[str, Any]] = []
    written_chunks: List[Path] = []
    chunk_index = 0
    completed_games = 0
    total_samples_collected = 0
    start_time = time.time()

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures: Dict[Any, int] = {}
            for game_index in range(config.games):
                future = executor.submit(
                    _run_single_game_worker,
                    game_index,
                    config.iteration,
                    model,
                    network_config,
                    config,
                    server,
                    config.seed + game_index + 1,
                )
                futures[future] = game_index

            for future in as_completed(futures):
                game_index = futures[future]
                try:
                    game_samples, game_elapsed, game_info = future.result()
                except Exception as e:
                    print(
                        f"[{_timestamp()}] [self-play] ERROR game {game_index}: {e}",
                        flush=True,
                    )
                    completed_games += 1
                    continue

                all_samples.extend(game_samples)
                completed_games += 1
                total_samples_collected += len(game_samples)

                winner_str = game_info.get("winner") or "draw"
                steps = game_info.get("steps") or 0
                trunc = game_info.get("truncated", False)
                handicap_mode = game_info.get("handicap_mode", "?")

                elapsed = time.time() - start_time
                games_per_sec = completed_games / max(elapsed, 0.001)
                samples_per_sec = total_samples_collected / max(elapsed, 0.001)
                nn_stats = server.stats

                print(
                    f"[{_timestamp()}] [self-play] "
                    f"game {completed_games}/{config.games} "
                    f"(idx={game_index}) "
                    f"winner={winner_str} steps={steps} "
                    f"trunc={trunc} mode={handicap_mode} "
                    f"samples={len(game_samples)} "
                    f"time={_format_duration(game_elapsed)} "
                    f"| total_samples={total_samples_collected} "
                    f"games/s={games_per_sec:.2f} "
                    f"samples/s={samples_per_sec:.1f} "
                    f"nn_avg_batch={nn_stats['avg_batch_size']:.1f} "
                    f"nn_max_batch={nn_stats['max_batch_size']} "
                    f"elapsed={_format_duration(elapsed)}",
                    flush=True,
                )

                # Write chunks as they fill up
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

        # Write remaining samples
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

    finally:
        server.stop()

    elapsed = time.time() - start_time
    nn_stats = server.stats
    print(
        f"[{_timestamp()}] [self-play] DONE iteration {config.iteration}: "
        f"games={completed_games} chunks={len(written_chunks)} "
        f"total_samples={total_samples_collected} "
        f"elapsed={_format_duration(elapsed)} "
        f"games/s={completed_games / max(elapsed, 0.001):.2f} "
        f"nn_batches={nn_stats['total_batches']} "
        f"nn_inferences={nn_stats['total_inferences']} "
        f"nn_avg_batch={nn_stats['avg_batch_size']:.1f} "
        f"nn_max_batch={nn_stats['max_batch_size']}",
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

    dataset = ReplayDataset(config.replay_dir, config.replay_limit)
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
        env = BarricadeEnv(max_steps=config.max_steps)
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
        )

        while not terminated and not truncated:
            current = env.state.current_player
            model = candidate_model if current == candidate_player else baseline_model
            state_before = env.state.copy()
            history_before = _history_window(history, network_config.history_length)
            mcts = MCTS(model, mcts_config)
            result = mcts.search(state_before, history=history_before, root=roots[current])
            action = mcts.select_action(result, temperature=0.0)
            history.append(state_before)
            _, _, terminated, truncated, info = env.step(action)
            roots[current] = mcts.advance_root(result.root, action)
            roots[current.opposite()] = None

        winner = info.get("winner")
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
    self_play_config: SelfPlayConfig,
    train_config: TrainConfig,
    eval_config: EvalConfig,
    checkpoint_path: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> None:
    current_checkpoint = Path(checkpoint_path) if checkpoint_path else None
    for iteration in range(1, iterations + 1):
        iter_start = time.time()
        print(
            f"\n{'=' * 70}\n"
            f"[{_timestamp()}] [loop] iteration {iteration}/{iterations}\n"
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
        tr_config = replace(train_config, iteration=iteration)
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

        if metrics["win_rate"] >= 0.5:
            current_checkpoint = candidate_path
            _copy_latest(candidate_path)
            print(
                f"[{_timestamp()}] [loop] candidate PROMOTED "
                f"(win_rate={metrics['win_rate']:.1%} >= 0.4)",
                flush=True,
            )
        else:
            _load_checkpoint_into(
                model, current_checkpoint, device=torch.device(device or "cpu")
            )
            print(
                f"[{_timestamp()}] [loop] candidate REJECTED "
                f"(win_rate={metrics['win_rate']:.1%} < 0.4), "
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

    # --- self-play ---
    self_play = subparsers.add_parser("self-play")
    add_network_args(self_play)
    self_play.add_argument("--iteration", type=int, default=1)
    self_play.add_argument("--games", type=int, default=16)
    self_play.add_argument("--base-simulations", type=int, default=128)
    self_play.add_argument("--mcts-batch-size", type=int, default=16)
    self_play.add_argument("--max-steps", type=int, default=500)
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

    # --- train ---
    train = subparsers.add_parser("train")
    add_network_args(train)
    train.add_argument("--iteration", type=int, default=1)
    train.add_argument("--replay-dir", type=str, default=str(SELFPLAY_DIR))
    train.add_argument("--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR))
    train.add_argument("--replay-limit", type=int, default=None)
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
    loop.add_argument("--games", type=int, default=16)
    loop.add_argument("--base-simulations", type=int, default=128)
    loop.add_argument("--mcts-batch-size", type=int, default=16)
    loop.add_argument("--max-steps", type=int, default=500)
    loop.add_argument("--chunk-size", type=int, default=2048)
    loop.add_argument("--temperature-drop-ply", type=int, default=20)
    loop.add_argument("--replay-dir", type=str, default=str(SELFPLAY_DIR))
    loop.add_argument("--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR))
    loop.add_argument("--replay-limit", type=int, default=None)
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
        )
        run_self_play(model, network_config, config, device=args.device)

    elif args.command == "train":
        model = build_model(network_config)
        config = TrainConfig(
            iteration=args.iteration,
            replay_dir=args.replay_dir,
            checkpoint_dir=args.checkpoint_dir,
            replay_limit=args.replay_limit,
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
        )
        train_config = TrainConfig(
            replay_dir=args.replay_dir,
            checkpoint_dir=args.checkpoint_dir,
            replay_limit=args.replay_limit,
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
            self_play_config=self_play_config,
            train_config=train_config,
            eval_config=eval_config,
            checkpoint_path=args.checkpoint,
            device=args.device,
        )


if __name__ == "__main__":
    main()
