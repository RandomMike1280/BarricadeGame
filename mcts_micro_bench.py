"""
CPU-side MCTS micro-benchmark.

Measures sims/sec for ``mcts.MCTS.search`` with a trivial inference model so
the CPU tree-walking overhead dominates (mirrors how a "batched GPU"
self-play worker looks: each batch's forward cost is amortized across the
batch via the inference server, leaving the per-simulation Python work as
the hot path).

Reproducible: fixed seed, deterministic ``BarricadeEnv`` plies, fresh MCTS
per search to avoid cross-search tree reuse.

CLI flags mirror ``mcts_benchmark.py`` defaults so the two numbers are
directly comparable. Run from the repo root:

    python mcts_micro_bench.py --positions 16 --simulations 400
"""
from __future__ import annotations

import argparse
import random
import time
from typing import List

import torch
from torch import nn

from barricade_env import BarricadeEnv, BarricadeState
from mcts import MCTS, MCTSConfig


class TinyModel(nn.Module):
    """Trivial inference stub. Cheap enough that the tree-walking cost
    dominates the per-batch forward pass, which is what we want to
    optimize."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def inference(self, batch: torch.Tensor):
        batch_size = batch.shape[0]
        device = batch.device
        logits = torch.zeros((batch_size, 136), device=device)
        values = torch.zeros((batch_size, 1), device=device)
        leads = torch.zeros((batch_size, 1), device=device)
        return logits, values, leads


def _make_positions(seed: int, count: int, warmup_plies: int) -> List[BarricadeState]:
    rng = random.Random(seed)
    out: List[BarricadeState] = []
    for _ in range(count):
        env = BarricadeEnv(max_steps=max(1, warmup_plies + 1), invalid_action_mode="raise")
        env.reset(seed=rng.randrange(2**31))
        plies = rng.randint(0, warmup_plies)
        for _ in range(plies):
            legal = env.legal_actions()
            if not legal:
                break
            _, _, terminated, truncated, _ = env.step(rng.choice(legal))
            if terminated or truncated:
                break
        out.append(env.state.copy())
    return out


def bench(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cpu")
    model = TinyModel().to(device)
    cfg = MCTSConfig(
        num_simulations=args.simulations,
        batch_size=args.batch_size,
        action_temperature=0.0,
        history_length=0,
        device=str(device),
        add_root_noise=False,
    )
    positions = _make_positions(args.seed, args.positions, args.warmup_plies)

    # Warmup
    mcts = MCTS(model, cfg, device=device, rng=random.Random(args.seed))
    for pos in positions[: args.warmup_searches]:
        mcts.search(pos)

    durations: List[float] = []
    completed_total = 0
    neural_batches_total = 0
    evaluated_leaves_total = 0
    collisions_total = 0

    # Fresh MCTS per search isolates "speed of a single search" from any
    # cross-search amortisation. This is the most adversarial measurement
    # of the CPU hot path.
    started = time.perf_counter()
    for pos in positions:
        for _ in range(args.searches_per_position):
            mcts = MCTS(model, cfg, device=device, rng=random.Random(args.seed))
            t0 = time.perf_counter()
            result = mcts.search(pos)
            durations.append(time.perf_counter() - t0)
            d = result.diagnostics
            completed_total += int(d.get("completed_simulations", 0))
            neural_batches_total += int(d.get("neural_batches", 0))
            evaluated_leaves_total += int(d.get("evaluated_leaves", 0))
            collisions_total += int(d.get("collisions", 0))
    elapsed = time.perf_counter() - started

    searches = len(durations)
    sims_per_sec = completed_total / elapsed if elapsed > 0 else 0.0
    searches_per_sec = searches / elapsed if elapsed > 0 else 0.0
    mean_ms = (sum(durations) / searches * 1000.0) if searches else 0.0
    ordered = sorted(durations)
    p50 = ordered[max(0, int(0.50 * (searches - 1)))] * 1000.0
    p95 = ordered[max(0, int(0.95 * (searches - 1)))] * 1000.0
    avg_batch = evaluated_leaves_total / max(1, neural_batches_total)

    print(f"searches={searches} positions={args.positions} sims={args.simulations} batch={args.batch_size}")
    print(f"elapsed={elapsed:.3f}s")
    print(f"sims/sec={sims_per_sec:.1f}")
    print(f"searches/sec={searches_per_sec:.2f}")
    print(f"latency_ms mean={mean_ms:.2f} p50={p50:.2f} p95={p95:.2f}")
    print(f"neural_batches={neural_batches_total} avg_batch={avg_batch:.2f} collisions={collisions_total}")
    print(f"completed_sims={completed_total}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--positions", type=int, default=16)
    p.add_argument("--searches-per-position", type=int, default=1)
    p.add_argument("--warmup-plies", type=int, default=12)
    p.add_argument("--warmup-searches", type=int, default=1)
    p.add_argument("--simulations", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    bench(args)


if __name__ == "__main__":
    main()
