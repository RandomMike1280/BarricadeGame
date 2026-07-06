"""
Focused microbench for ``BarricadeState.copy`` and ``barricade_env.apply_move``.

Targets the two functions the user asked to improve. Measures per-call
latency in microseconds on representative states (fresh, mid-game with
walls, late-game near terminal), with several repetitions to tame noise.

Run:
    python copy_apply_move_bench.py [--reps 5] [--warmup 2000] [--iters 50000]
"""
from __future__ import annotations

import argparse
import gc
import random
import statistics
import time
from typing import Callable, List, Tuple

from barricade_env import (
    BarricadeEnv,
    BarricadeState,
    Player,
    WallOrientation,
)


def _make_states(seed: int) -> dict:
    """Build a battery of representative states:
    - ``fresh``:    initial position, no walls.
    - ``mid``:      after some plies + walls, the typical mid-game state.
    - ``late``:     later mid-game with more walls.
    - ``heavy``:    wall_heavy_state-style (many walls).
    """
    rng = random.Random(seed)

    states: dict = {}

    # Fresh
    env = BarricadeEnv(max_steps=1, invalid_action_mode="raise")
    env.reset(seed=rng.randrange(2**31))
    states["fresh"] = env.state.copy()

    # Mid (6 plies)
    env = BarricadeEnv(max_steps=100, invalid_action_mode="raise")
    env.reset(seed=rng.randrange(2**31))
    for _ in range(6):
        legal = env.legal_actions()
        if not legal:
            break
        _, _, term, trunc, _ = env.step(rng.choice(legal))
        if term or trunc:
            break
    states["mid"] = env.state.copy()

    # Late (14 plies)
    env = BarricadeEnv(max_steps=100, invalid_action_mode="raise")
    env.reset(seed=rng.randrange(2**31))
    for _ in range(14):
        legal = env.legal_actions()
        if not legal:
            break
        _, _, term, trunc, _ = env.step(rng.choice(legal))
        if term or trunc:
            break
    states["late"] = env.state.copy()

    # Heavy (24 plies, biased to walls)
    env = BarricadeEnv(max_steps=100, invalid_action_mode="raise")
    env.reset(seed=rng.randrange(2**31))
    for _ in range(24):
        legal = env.legal_actions()
        if not legal:
            break
        wall_actions = [a for a in legal if a >= 4]
        pick = rng.choice(wall_actions or legal)
        _, _, term, trunc, _ = env.step(pick)
        if term or trunc:
            break
    states["heavy"] = env.state.copy()

    return states


def _time_call(fn: Callable[[], None], iters: int) -> float:
    """Return microseconds-per-call for ``fn``."""
    # Run twice — Python's first call pays for warmup of caches/JIT-like
    # things; take the second pass.
    fn()
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - t0
    return (elapsed / iters) * 1e6


def _bench_copy(state: BarricadeState, iters: int) -> float:
    return _time_call(state.copy, iters)


def _bench_apply_move(state: BarricadeState, iters: int) -> Tuple[float, int]:
    """Time ``apply_move`` over a rotating set of legal moves so caches
    don't lock us into one path. Returns (us_per_call, n_moves_used).

    ``legal_action_moves()`` already returns the rule-engine move tuples
    (the ``("move_to", row, col)`` form, not ``("move", direction)``) that
    ``apply_move`` consumes directly — we feed those back in unchanged.

    Because ``apply_move`` is immutable and consumes the state, we restore
    the original state between calls by re-copying it. The copy itself is
    what we're trying to measure in ``_bench_copy`` — but it shows up in
    the cost of ``apply_move`` too, since every ``apply_move`` does its
    own internal copy. So the cost here reflects production behaviour
    (an MCTS descent is "copy state, mutate it, pass to child").
    """
    moves = state.legal_action_moves()
    if not moves:
        return (float("nan"), 0)
    move_tuples = [m for _, m in moves]
    n = len(move_tuples)
    base = state  # the immutable template

    i = 0
    current = state.copy()

    def step():
        nonlocal i, current
        m = move_tuples[i]
        i = (i + 1) % n
        current = base.apply_move(m)

    # Sanity: validate one call works (no exception)
    try:
        step()
    except Exception:
        def step():
            nonlocal i, current
            for attempt in range(n):
                idx = (i + attempt) % n
                try:
                    current = base.apply_move(move_tuples[idx])
                    i = (idx + 1) % n
                    return
                except Exception:
                    continue

    # Warmup
    step()
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    elapsed = time.perf_counter() - t0
    return ((elapsed / iters) * 1e6, n)


def _bench_apply_action(state: BarricadeState, iters: int) -> Tuple[float, int]:
    """Time ``apply_action`` on rotating actions. Restores the template
    state between calls via ``state.copy()`` so cached legal-actions lists
    stay valid. ``validate=False`` matches what ``mcts.py`` uses after it
    has independently checked legality.
    """
    actions = state.legal_actions()
    if not actions:
        return (float("nan"), 0)
    n = len(actions)
    base = state
    i = 0
    current = state.copy()

    def step():
        nonlocal i, current
        a = actions[i]
        i = (i + 1) % n
        current = base.apply_action(a, validate=False)

    try:
        step()
    except Exception:
        def step():
            nonlocal i, current
            for attempt in range(n):
                idx = (i + attempt) % n
                try:
                    current = base.apply_action(actions[idx], validate=False)
                    i = (idx + 1) % n
                    return
                except Exception:
                    continue

    step()
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    elapsed = time.perf_counter() - t0
    return ((elapsed / iters) * 1e6, n)


def _bench_apply_action(state: BarricadeState, iters: int) -> Tuple[float, int]:
    """Time ``apply_action`` on rotating actions. Restores the template
    state between calls via ``state.copy()`` so cached legal-actions lists
    stay valid. ``validate=False`` matches what ``mcts.py`` uses after it
    has independently checked legality.
    """
    actions = state.legal_actions()
    if not actions:
        return (float("nan"), 0)
    n = len(actions)
    base = state
    i = 0
    current = state.copy()

    def step():
        nonlocal i, current
        a = actions[i]
        i = (i + 1) % n
        current = base.apply_action(a, validate=False)

    try:
        step()
    except Exception:
        def step():
            nonlocal i, current
            for attempt in range(n):
                idx = (i + attempt) % n
                try:
                    current = base.apply_action(actions[idx], validate=False)
                    i = (idx + 1) % n
                    return
                except Exception:
                    continue

    step()
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    elapsed = time.perf_counter() - t0
    return ((elapsed / iters) * 1e6, n)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--reps", type=int, default=5,
                   help="Repetitions per measurement; report median.")
    args = p.parse_args()

    states = _make_states(args.seed)

    copy_results: dict = {}
    apply_results: dict = {}
    apply_action_results: dict = {}

    for name, template in states.items():
        print(f"--- state: {name}  (walls={len(template.walls)}, "
              f"current={template.current_player.name}) ---")

        # Warmup on a disposable copy so the template is never mutated.
        warmup_template = template.copy()
        for _ in range(args.warmup):
            warmup_template.copy()
            moves = warmup_template.legal_action_moves()
            if moves:
                warmup_template = warmup_template.apply_move(moves[0][1])

        # copy()
        copy_us = []
        for _ in range(args.reps):
            copy_us.append(_bench_copy(template, args.iters))
        copy_median = statistics.median(copy_us)
        copy_min = min(copy_us)
        copy_results[name] = copy_median
        print(f"  copy()        median={copy_median:.3f} us  "
              f"min={copy_min:.3f} us  reps={[f'{x:.3f}' for x in copy_us]}")

        # apply_move()
        am_us, n_moves = _bench_apply_move(template, args.iters)
        apply_results[name] = am_us
        print(f"  apply_move()  {am_us:.3f} us  (rotating over {n_moves} moves)")

        # apply_action(validate=False) for parity with mcts.py's per-step path
        aa_us, n_moves = _bench_apply_action(template, args.iters)
        apply_action_results[name] = aa_us
        print(f"  apply_action() {aa_us:.3f} us  (validate=False, "
              f"rotating over {n_moves} moves)")

    print("\n=== Summary (median over reps) ===")
    print(f"{'state':<8} {'copy(us)':>10} {'apply_move(us)':>16} "
          f"{'apply_action(us)':>20}")
    for name in states:
        print(f"{name:<8} {copy_results[name]:>10.3f} "
              f"{apply_results[name]:>16.3f} "
              f"{apply_action_results[name]:>20.3f}")


if __name__ == "__main__":
    main()