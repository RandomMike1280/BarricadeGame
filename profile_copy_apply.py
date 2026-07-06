"""Detailed profile of copy() and apply_move() on representative states.

Uses cProfile to break down per-call cost so I can target the right lines.
"""
import cProfile
import pstats
import random
import io

from barricade_env import BarricadeEnv, BarricadeState, Player


def _make_states(seed: int) -> dict:
    rng = random.Random(seed)
    states = {}

    env = BarricadeEnv(max_steps=1, invalid_action_mode="raise")
    env.reset(seed=rng.randrange(2**31))
    states["fresh"] = env.state.copy()

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


def _profile(states, n: int = 50000):
    pr = cProfile.Profile()

    print("\n=== copy() ===")
    for name, st in states.items():
        # warmup
        for _ in range(2000):
            st.copy()
        pr.enable()
        for _ in range(n):
            st.copy()
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(20)
        # Filter just copy + child calls
        lines = s.getvalue().splitlines()
        print(f"\n--- {name} ---")
        for line in lines[:30]:
            print(line)
        pr.clear()

    print("\n=== apply_move() ===")
    for name, st in states.items():
        moves = st.legal_action_moves()
        if not moves:
            continue
        move_tuples = [m for _, m in moves]
        i = 0
        # warmup
        for _ in range(2000):
            st.apply_move(move_tuples[i % len(move_tuples)])
            i += 1
        pr.enable()
        for _ in range(n):
            st.apply_move(move_tuples[i % len(move_tuples)])
            i += 1
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(20)
        lines = s.getvalue().splitlines()
        print(f"\n--- {name} ---")
        for line in lines[:30]:
            print(line)
        pr.clear()


if __name__ == "__main__":
    states = _make_states(2026)
    _profile(states)