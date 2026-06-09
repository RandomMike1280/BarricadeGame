"""
Self-play and training pipeline for Barricade AlphaZero experiments.

This module is intentionally single-process for v1. It provides CLI subcommands
for self-play generation, training, evaluation, and an iteration loop.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
import random
import shutil
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from barricade_env import ACTION_SIZE, BarricadeEnv, Player
from mcts import MCTS, MCTSConfig
from network import EncoderConfig, build_network, encode_state_stack


SELFPLAY_DIR = Path("data/selfplay")
CHECKPOINT_DIR = Path("checkpoints")


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


@dataclass(frozen=True)
class PipelineConfig:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


class ReplayDataset(Dataset):
    def __init__(self, replay_dir: str | Path, limit: Optional[int] = None) -> None:
        self.replay_dir = Path(replay_dir)
        self.samples: List[Dict[str, Any]] = []
        for path in sorted(self.replay_dir.glob("*.pt")):
            payload = torch.load(path, map_location="cpu")
            self.samples.extend(payload.get("samples", []))
        if limit is not None and limit > 0 and len(self.samples) > limit:
            self.samples = self.samples[-limit:]
        if not self.samples:
            raise ValueError(f"No replay samples found in {self.replay_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        sample = self.samples[index]
        state_planes = sample["state_planes"].float()
        policy_target = sample["policy_target"].float()
        value_target = torch.as_tensor(sample["value_target"], dtype=torch.float32)
        return state_planes, policy_target, value_target


def sample_playout_cap(base_simulations: int, rng: random.Random) -> int:
    roll = rng.random()
    if roll < 0.70:
        cap = base_simulations
    elif roll < 0.90:
        cap = base_simulations // 2
    else:
        cap = base_simulations * 2
    return max(1, int(cap))


def sample_handicap(rng: random.Random) -> Dict[str, Any]:
    roll = rng.random()
    config: Dict[str, Any]
    mode: str

    if roll < 0.70:
        mode = "standard"
        config = {
            "red_start": (0, 4),
            "blue_start": (8, 4),
            "red_walls": 10,
            "blue_walls": 10,
        }
    elif roll < 0.90:
        mode = "same_row_shifted"
        config = {
            "red_start": (0, rng.randint(2, 6)),
            "blue_start": (8, rng.randint(2, 6)),
            "red_walls": 10,
            "blue_walls": 10,
        }
    elif roll < 0.97:
        mode = "row_ahead"
        config = {
            "red_start": (rng.randint(0, 2), rng.randint(2, 6)),
            "blue_start": (rng.randint(6, 8), rng.randint(2, 6)),
            "red_walls": 10,
            "blue_walls": 10,
        }
    else:
        mode = "wall_handicap"
        config = {
            "red_start": (0, 4),
            "blue_start": (8, 4),
            "red_walls": rng.randint(7, 13),
            "blue_walls": rng.randint(7, 13),
        }

    if mode != "standard" and mode != "wall_handicap" and rng.random() < 0.5:
        config["red_walls"] = rng.randint(7, 13)
        config["blue_walls"] = rng.randint(7, 13)

    config["handicap_mode"] = mode
    return config


def sample_valid_handicap(rng: random.Random, max_attempts: int = 100) -> Dict[str, Any]:
    for _ in range(max_attempts):
        config = sample_handicap(rng)
        env = BarricadeEnv(max_steps=1)
        try:
            env.reset(options=_env_options(config))
            return config
        except ValueError:
            continue
    raise RuntimeError("Failed to sample a valid handicap configuration.")


def run_self_play(
    model: nn.Module,
    network_config: NetworkConfig,
    config: SelfPlayConfig,
    *,
    device: Optional[str] = None,
) -> List[Path]:
    rng = random.Random(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    samples: List[Dict[str, Any]] = []
    written_chunks: List[Path] = []
    chunk_index = 0

    for game_index in range(config.games):
        game_id = f"iter_{config.iteration:06d}_game_{game_index:06d}"
        handicap = sample_valid_handicap(rng)
        env = BarricadeEnv(max_steps=config.max_steps)
        _, info = env.reset(options=_env_options(handicap))
        history = []
        pending: List[Dict[str, Any]] = []
        root = None
        terminated = False
        truncated = False
        ply = 0

        while not terminated and not truncated:
            player_to_move = env.state.current_player
            state_before = env.state.copy()
            history_before = tuple(history[-network_config.history_length :])
            state_planes = encode_state_stack(
                state_before,
                history_before,
                history_length=network_config.history_length,
            ).cpu()

            simulations = sample_playout_cap(config.base_simulations, rng)
            mcts_config = MCTSConfig(
                num_simulations=simulations,
                batch_size=config.batch_size,
                history_length=network_config.history_length,
                device=device,
                add_root_noise=True,
                action_temperature=1.0 if ply < config.temperature_drop_ply else 0.0,
            )
            mcts = MCTS(model, mcts_config)
            result = mcts.search(state_before, history=history_before, root=root)
            action = mcts.select_action(
                result,
                temperature=1.0 if ply < config.temperature_drop_ply else 0.0,
            )

            pending.append(
                {
                    "state_planes": state_planes,
                    "policy_target": torch.as_tensor(
                        result.policy_target, dtype=torch.float32
                    ),
                    "value_target": 0.0,
                    "legal_actions": list(info["legal_actions"]),
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
            root = mcts.advance_root(result.root, action)
            ply += 1

        winner = info.get("winner")
        final_metadata = {
            "winner": winner,
            "lead": info.get("lead"),
            "N_moves": info.get("N_moves"),
            "game_length": info.get("steps"),
            "truncated": bool(truncated),
        }
        for sample in pending:
            sample.update(final_metadata)
            sample["value_target"] = _value_target(sample["side_to_move"], winner, truncated)
        samples.extend(pending)

        if len(samples) >= config.chunk_size:
            path = _write_replay_chunk(
                samples,
                output_dir,
                config.iteration,
                chunk_index,
                network_config,
                config,
            )
            written_chunks.append(path)
            samples = []
            chunk_index += 1

    if samples:
        path = _write_replay_chunk(
            samples,
            output_dir,
            config.iteration,
            chunk_index,
            network_config,
            config,
        )
        written_chunks.append(path)

    return written_chunks


def train_from_replay(
    model: nn.Module,
    network_config: NetworkConfig,
    config: TrainConfig,
    *,
    checkpoint_path: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> Path:
    _set_seed(config.seed)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(resolved_device)

    dataset = ReplayDataset(config.replay_dir, config.replay_limit)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    if checkpoint_path:
        _load_checkpoint_into(model, checkpoint_path, optimizer=optimizer, device=resolved_device)

    stats = []
    model.train()
    for epoch in range(config.epochs):
        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        batches = 0

        for state_planes, policy_target, value_target in loader:
            state_planes = state_planes.to(resolved_device)
            policy_target = policy_target.to(resolved_device)
            value_target = value_target.to(resolved_device)

            logits, value = model(state_planes)
            policy_loss = soft_target_cross_entropy(logits, policy_target)
            value_loss = F.mse_loss(value.squeeze(-1), value_target)
            loss = policy_loss + config.value_loss_weight * value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            total_loss_sum += float(loss.item())
            policy_loss_sum += float(policy_loss.item())
            value_loss_sum += float(value_loss.item())
            batches += 1

        epoch_stats = {
            "epoch": epoch,
            "total_loss": total_loss_sum / max(1, batches),
            "policy_loss": policy_loss_sum / max(1, batches),
            "value_loss": value_loss_sum / max(1, batches),
            "batches": batches,
            "samples": len(dataset),
        }
        stats.append(epoch_stats)
        print(
            f"epoch={epoch} total={epoch_stats['total_loss']:.4f} "
            f"policy={epoch_stats['policy_loss']:.4f} "
            f"value={epoch_stats['value_loss']:.4f}"
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
    return output_path


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

    for game_index in range(config.games):
        candidate_player = Player.RED if game_index % 2 == 0 else Player.BLUE
        env = BarricadeEnv(max_steps=config.max_steps)
        handicap = sample_valid_handicap(rng)
        _, info = env.reset(options=_env_options(handicap))
        history = []
        roots = {Player.RED: None, Player.BLUE: None}
        terminated = False
        truncated = False

        while not terminated and not truncated:
            current = env.state.current_player
            model = candidate_model if current == candidate_player else baseline_model
            state_before = env.state.copy()
            history_before = tuple(history[-network_config.history_length :])
            mcts_config = MCTSConfig(
                num_simulations=config.simulations,
                batch_size=config.batch_size,
                history_length=network_config.history_length,
                device=device,
                add_root_noise=False,
                action_temperature=0.0,
            )
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
        elif winner == candidate_player.name:
            wins += 1
        else:
            losses += 1

        if info.get("lead") is not None:
            leads.append(float(info["lead"]))
        game_lengths.append(int(info.get("steps") or 0))
        n_moves = info.get("N_moves") or {}
        red_pawn_moves.append(int(n_moves.get("RED", 0)))
        blue_pawn_moves.append(int(n_moves.get("BLUE", 0)))

    total = max(1, config.games)
    return {
        "games": config.games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / total,
        "average_lead": _mean(leads),
        "average_game_length": _mean(game_lengths),
        "average_red_pawn_moves": _mean(red_pawn_moves),
        "average_blue_pawn_moves": _mean(blue_pawn_moves),
    }


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
        print(f"iteration={iteration} self-play")
        sp_config = _replace_dataclass(self_play_config, iteration=iteration)
        run_self_play(model, network_config, sp_config, device=device)

        print(f"iteration={iteration} train")
        tr_config = _replace_dataclass(train_config, iteration=iteration)
        candidate_path = train_from_replay(
            model,
            network_config,
            tr_config,
            checkpoint_path=current_checkpoint,
            device=device,
        )

        if current_checkpoint is None:
            current_checkpoint = candidate_path
            _copy_latest(candidate_path)
            continue

        baseline = build_model(network_config)
        _load_checkpoint_into(baseline, current_checkpoint, device=torch.device(device or "cpu"))
        metrics = evaluate_models(model, baseline, network_config, eval_config, device=device)
        print(f"eval metrics: {metrics}")
        if metrics["win_rate"] >= 0.55:
            current_checkpoint = candidate_path
            _copy_latest(candidate_path)
        else:
            _load_checkpoint_into(model, current_checkpoint, device=torch.device(device or "cpu"))


def soft_target_cross_entropy(logits: Tensor, target: Tensor) -> Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(target * log_probs).sum(dim=1).mean()


def build_model(config: NetworkConfig) -> nn.Module:
    return build_network(
        history_length=config.history_length,
        conv_channels=config.conv_channels,
        residual_channels=config.residual_channels,
        num_conv_layers=config.num_conv_layers,
        num_residual_layers=config.num_residual_layers,
        value_hidden_size=config.value_hidden_size,
    )


def load_model_from_checkpoint(path: str | Path, device: Optional[str] = None) -> Tuple[nn.Module, NetworkConfig]:
    payload = torch.load(path, map_location=device or "cpu")
    network_config = NetworkConfig(**payload.get("network_config", {}))
    model = build_model(network_config)
    model.load_state_dict(payload["model_state"])
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
    payload = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


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
    print(f"wrote {path} samples={len(samples)}")
    return path


def _value_target(side_to_move: str, winner: Optional[str], truncated: bool) -> float:
    if winner is None:
        return 0.0
    return 1.0 if side_to_move == winner else -1.0


def _env_options(handicap: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "red_start": handicap["red_start"],
        "blue_start": handicap["blue_start"],
        "red_walls": handicap["red_walls"],
        "blue_walls": handicap["blue_walls"],
    }


def _replace_dataclass(instance, **changes):
    values = asdict(instance)
    values.update(changes)
    return type(instance)(**values)


def _copy_latest(checkpoint_path: Path) -> None:
    latest_path = checkpoint_path.parent / "latest.pt"
    shutil.copy2(checkpoint_path, latest_path)
    print(f"promoted {checkpoint_path} -> {latest_path}")


def _mean(values: Sequence[float | int]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / len(values)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Barricade AlphaZero training pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_network_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--history-length", type=int, default=0)
        p.add_argument("--conv-channels", type=int, default=128)
        p.add_argument("--residual-channels", type=int, default=None)
        p.add_argument("--num-conv-layers", type=int, default=1)
        p.add_argument("--num-residual-layers", type=int, default=10)
        p.add_argument("--value-hidden-size", type=int, default=256)
        p.add_argument("--checkpoint", type=str, default=None)
        p.add_argument("--device", type=str, default=None)

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
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=1)

    eval_parser = subparsers.add_parser("eval")
    add_network_args(eval_parser)
    eval_parser.add_argument("--baseline-checkpoint", type=str, required=True)
    eval_parser.add_argument("--games", type=int, default=10)
    eval_parser.add_argument("--simulations", type=int, default=64)
    eval_parser.add_argument("--mcts-batch-size", type=int, default=16)
    eval_parser.add_argument("--max-steps", type=int, default=500)
    eval_parser.add_argument("--seed", type=int, default=1)

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
    loop.add_argument("--grad-clip", type=float, default=1.0)
    loop.add_argument("--weight-decay", type=float, default=1e-4)
    loop.add_argument("--eval-games", type=int, default=10)
    loop.add_argument("--eval-simulations", type=int, default=64)
    loop.add_argument("--seed", type=int, default=1)

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
    network_config = network_config_from_args(args)
    _set_seed(getattr(args, "seed", 1))

    if args.command == "self-play":
        model = build_model(network_config)
        if args.checkpoint:
            _load_checkpoint_into(model, args.checkpoint, device=torch.device(args.device or "cpu"))
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
        candidate, candidate_config = load_model_from_checkpoint(args.checkpoint, args.device)
        baseline, _ = load_model_from_checkpoint(args.baseline_checkpoint, args.device)
        config = EvalConfig(
            games=args.games,
            simulations=args.simulations,
            batch_size=args.mcts_batch_size,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        metrics = evaluate_models(candidate, baseline, candidate_config, config, device=args.device)
        print(metrics)

    elif args.command == "loop":
        model = build_model(network_config)
        if args.checkpoint:
            _load_checkpoint_into(model, args.checkpoint, device=torch.device(args.device or "cpu"))
        self_play_config = SelfPlayConfig(
            games=args.games,
            base_simulations=args.base_simulations,
            batch_size=args.mcts_batch_size,
            max_steps=args.max_steps,
            chunk_size=args.chunk_size,
            temperature_drop_ply=args.temperature_drop_ply,
            seed=args.seed,
            output_dir=args.replay_dir,
        )
        train_config = TrainConfig(
            replay_dir=args.replay_dir,
            checkpoint_dir=args.checkpoint_dir,
            replay_limit=args.replay_limit,
            epochs=args.epochs,
            batch_size=args.train_batch_size,
            learning_rate=args.learning_rate,
            value_loss_weight=args.value_loss_weight,
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
