"""Fine-tune a checkpoint's value head on empty-board tactical race positions."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import random
import time
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from barricade_env import BarricadeState, Player
from network import encode_state_stack
from probe_tactical_value import scenarios
from train import (
    NetworkConfig,
    _load_checkpoint_into,
    _network_config_from_payload,
    build_model,
    tactical_value_batch,
    tactical_value_policy_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair tactical value calibration without a full self-play cycle."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.pt"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/tactical_value_finetuned.pt"),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--policy-loss-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--train-scope",
        choices=("value-head", "all"),
        default="value-head",
        help="Train only the scalar value head by default, or all network weights.",
    )
    parser.add_argument(
        "--policy-head-type",
        choices=("checkpoint", "flat", "factored"),
        default="checkpoint",
    )
    parser.add_argument(
        "--reset-policy-head",
        action="store_true",
        help="Required when changing --policy-head-type from the checkpoint type.",
    )
    return parser.parse_args()


def load_for_finetune(
    checkpoint: Path,
    *,
    device: torch.device,
    policy_head_type: str,
    reset_policy_head: bool,
) -> Tuple[torch.nn.Module, NetworkConfig]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = _network_config_from_payload(payload)
    if policy_head_type != "checkpoint":
        config = replace(config, policy_head_type=policy_head_type)

    model = build_model(config).to(device)
    _load_checkpoint_into(
        model,
        checkpoint,
        device=device,
        reset_policy_head=reset_policy_head,
    )
    return model, config


@torch.inference_mode()
def probe_summary(
    model: torch.nn.Module,
    *,
    device: torch.device,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    values: Dict[str, float] = {}
    for label, red_start, blue_start, current, _ in scenarios():
        state = BarricadeState(
            red_start=red_start,
            blue_start=blue_start,
            red_walls=10,
            blue_walls=10,
            starting_player=current,
            board_size=9,
        )
        history_length = int(getattr(model, "_finetune_history_length", 0))
        planes = encode_state_stack(
            state,
            (),
            history_length=history_length,
        ).unsqueeze(0).to(device)
        _, value, _ = model.inference(planes)
        values[label] = float(value.reshape(-1)[0].clamp(-1.0, 1.0).item())
    if was_training:
        model.train()
    return values


def print_probe(title: str, values: Dict[str, float]) -> None:
    print(title)
    for label, value in values.items():
        print(f"  {label:<36} raw_stm={value:+.3f}")


def parameters_for_scope(
    model: torch.nn.Module,
    train_scope: str,
) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = train_scope == "all"
    if train_scope == "all":
        return [parameter for parameter in model.parameters() if parameter.requires_grad]

    for parameter in model.value_head.parameters():
        parameter.requires_grad = True
    return list(model.value_head.parameters())


def set_training_mode_for_scope(model: torch.nn.Module, train_scope: str) -> None:
    if train_scope == "all":
        model.train()
        return
    model.eval()
    model.value_head.train()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.policy_loss_weight > 0.0 and args.train_scope != "all":
        raise ValueError("--policy-loss-weight requires --train-scope all.")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)

    model, config = load_for_finetune(
        args.checkpoint,
        device=device,
        policy_head_type=args.policy_head_type,
        reset_policy_head=args.reset_policy_head,
    )
    setattr(model, "_finetune_history_length", config.history_length)
    trainable_parameters = parameters_for_scope(model, args.train_scope)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(
        f"checkpoint={args.checkpoint} output={args.output} device={device} "
        f"steps={args.steps} batch_size={args.batch_size} "
        f"policy_head={config.policy_head_type} train_scope={args.train_scope} "
        f"policy_loss_weight={args.policy_loss_weight}"
    )
    print_probe("before", probe_summary(model, device=device))

    set_training_mode_for_scope(model, args.train_scope)
    started = time.time()
    print_interval = max(1, args.steps // 10)
    for step in range(1, args.steps + 1):
        if args.policy_loss_weight > 0.0:
            planes, target, policy_target, policy_mask = tactical_value_policy_batch(
                batch_size=args.batch_size,
                history_length=config.history_length,
                rng=rng,
                device=device,
            )
        else:
            planes, target = tactical_value_batch(
                batch_size=args.batch_size,
                history_length=config.history_length,
                rng=rng,
                device=device,
            )
            policy_target = torch.zeros(args.batch_size, dtype=torch.long, device=device)
            policy_mask = torch.zeros(args.batch_size, dtype=torch.float32, device=device)

        logits, value, _, _, _ = model(planes)
        loss = F.mse_loss(value.squeeze(-1), target)
        policy_loss = logits.sum() * 0.0
        if args.policy_loss_weight > 0.0 and float(policy_mask.sum().item()) > 0.0:
            per_sample_policy_loss = F.cross_entropy(
                logits,
                policy_target,
                reduction="none",
            )
            policy_loss = (per_sample_policy_loss * policy_mask).sum() / policy_mask.sum()
            loss = loss + args.policy_loss_weight * policy_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % print_interval == 0 or step == args.steps:
            print(
                f"step={step:>5}/{args.steps} loss={loss.item():.4f} "
                f"policy_loss={policy_loss.item():.4f}"
            )

    model.eval()
    print_probe("after", probe_summary(model, device=device))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "network_config": asdict(config),
            "finetune_config": {
                "source_checkpoint": str(args.checkpoint),
                "steps": args.steps,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "policy_loss_weight": args.policy_loss_weight,
                "train_scope": args.train_scope,
                "seed": args.seed,
                "reset_policy_head": args.reset_policy_head,
            },
            "elapsed_seconds": time.time() - started,
        },
        args.output,
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
