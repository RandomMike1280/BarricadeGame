"""
Confirmation experiment: is the wall-biased PRIOR the lever behind pawn-move
starvation?

Two modes:
  probe   - single position A/B: run MCTS with each condition on the SAME states
            and report prior pawn-group mass, visit pawn-group mass, and the
            chosen move. Fast (a few searches). Proves the first-visit-order claim.
  compare - play N games vs the Alpha-Beta opponent under each condition and
            aggregate win/loss/draw, pawn-move fraction, and mean prior/visit
            pawn-group mass at the model's moves.

Conditions:
  baseline : fpu_reduction=0.33, pawn_prior_floor=0.0   (current eval config)
  fpu0     : fpu_reduction=0.00, pawn_prior_floor=0.0   (isolate FPU)
  floor    : fpu_reduction=0.33, pawn_prior_floor=0.30  (group floor on prior)
  fpu0floor: fpu_reduction=0.00, pawn_prior_floor=0.30  (both)
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from barricade_env import (
    ACTION_SIZE,
    DIAGONAL_HOP_ACTIONS,
    MOVE_ACTIONS,
    BarricadeState,
    Player,
    apply_selected_action,
)
from canonical import canonical_action
from mcts import MCTS, MCTSConfig
from mini_bench import (
    MODEL_HISTORY_LENGTH_ATTR,
    AlphaBetaAI,
    adjudicated_winner,
    evaluate_value_head_blue_pov,
    load_model,
    model_for_mcts,
    probe_value_perspective,
    raw_value_head,
)


def is_pawn(action: int) -> bool:
    return action < MOVE_ACTIONS or action >= ACTION_SIZE - DIAGONAL_HOP_ACTIONS


CONDITIONS: Dict[str, Tuple[float, float]] = {
    # name      : (fpu_reduction, pawn_prior_floor)
    "baseline":  (0.33, 0.00),
    "fpu0":      (0.00, 0.00),
    "floor":     (0.33, 0.30),
    "fpu0floor": (0.00, 0.30),
}


def build_mcts(model, *, fpu: float, floor: float, sims: int, batch: int,
               history_length: int, device, rng) -> MCTS:
    config = MCTSConfig(
        num_simulations=sims,
        batch_size=batch,
        cpuct_init=1.5,
        policy_target_temperature=1.0,
        action_temperature=0.0,
        history_length=history_length,
        add_root_noise=False,
        lead_weight=0.02,
        lead_scale=5.0,
        fpu_reduction=fpu,
        pawn_prior_floor=floor,
    )
    return MCTS(
        model_for_mcts(model),
        config=config,
        device=device,
        rng=rng,
        policy_action_transform=lambda action, state: canonical_action(
            action, state.current_player
        ),
    )


def measure_search(result) -> Tuple[float, float, bool, List[Tuple[int, int]]]:
    """Return (prior_pawn_mass, visit_pawn_mass, chosen_is_pawn, top_visits)."""
    edges = result.root.edges
    prior_pawn = sum(e.prior for a, e in edges.items() if is_pawn(a))
    visit_pawn = sum(result.policy_target[i] for i in range(ACTION_SIZE) if is_pawn(i))
    top = sorted(((e.visits, a) for a, e in edges.items()), reverse=True)[:5]
    top_visits = [(a, v) for v, a in top]
    return prior_pawn, visit_pawn, is_pawn(int(result.action)), top_visits


def initial_state(board_size: int, walls: int, start_col: int,
                  starting_player: Player) -> BarricadeState:
    return BarricadeState(
        red_start=(0, start_col),
        blue_start=(board_size - 1, start_col),
        red_walls=walls,
        blue_walls=walls,
        starting_player=starting_player,
        board_size=board_size,
    )


def play_one(model, *, board_size, walls, sims, batch, max_steps, ab_depth,
             device, rng, history_length, game_idx, fpu, floor,
             starting_player=Player.RED):
    """Play one game; return (winner, adj_winner, steps, model_move_records)."""
    model_role = Player.RED if (game_idx // 2) % 2 == 0 else Player.BLUE
    ab = AlphaBetaAI(model_role.opposite(), max_depth=ab_depth)
    start_col = rng.randrange(board_size)
    state = initial_state(board_size, walls, start_col, starting_player)
    history: List[BarricadeState] = []
    mcts = build_mcts(model, fpu=fpu, floor=floor, sims=sims, batch=batch,
                      history_length=history_length, device=device, rng=rng)
    records = []  # (prior_pawn, visit_pawn, chosen_is_pawn, value_blue, legal_pawn)
    steps = 0
    for ply in range(max_steps):
        if state.winner is not None or getattr(state, "is_draw", False):
            break
        legal = state.legal_actions()
        if not legal:
            state.winner = state.current_player.opposite()
            break
        state_before = state.copy()
        if state.current_player == model_role:
            hist = history[-history_length:] if history_length > 0 else ()
            result = mcts.search(state_before, history=hist)
            action = int(result.action)
            if action not in legal:
                action = legal[0]
            prior_pawn, visit_pawn, chosen_pawn, _ = measure_search(result)
            value_blue = evaluate_value_head_blue_pov(
                model, state_before, board_size=board_size, device=device,
                history=history,
            )
            n_pawn_legal = sum(1 for a in legal if is_pawn(a))
            records.append((prior_pawn, visit_pawn, chosen_pawn, value_blue, n_pawn_legal))
        else:
            action = ab.get_best_move(state_before)
        state = apply_selected_action(state, action, legal)
        history.append(state_before)
        steps = ply + 1
    winner = state.winner
    adj = winner if winner is not None else adjudicated_winner(state)
    return winner, adj, steps, model_role, records


def run_compare(model, args, device, history_length):
    rng_seed = args.seed
    print(f"\n{'cond':<10} {'W':>3} {'L':>3} {'D':>3} | "
          f"{'adjW':>4} {'adjL':>4} | {'pawn%play':>9} {'prior_pawn':>11} "
          f"{'visit_pawn':>11} {'steps':>6}")
    print("-" * 86)
    summary = {}
    for name, (fpu, floor) in CONDITIONS.items():
        if args.conditions and name not in args.conditions:
            continue
        rng = random.Random(rng_seed)  # same seed -> same start cols/roles across conds
        W = L = D = adjW = adjL = 0
        all_recs: List = []
        total_steps = 0
        t0 = time.time()
        for g in range(args.games):
            winner, adj, steps, model_role, recs = play_one(
                model, board_size=args.board_size, walls=args.walls,
                sims=args.simulations, batch=args.batch_size, max_steps=args.max_steps,
                ab_depth=args.depth, device=device, rng=rng, history_length=history_length,
                game_idx=g, fpu=fpu, floor=floor,
            )
            if winner == model_role:
                W += 1
            elif winner == model_role.opposite():
                L += 1
            else:
                D += 1
            if adj == model_role:
                adjW += 1
            elif adj == model_role.opposite():
                adjL += 1
            total_steps += steps
            all_recs.extend(recs)
        dt = time.time() - t0
        n = max(1, len(all_recs))
        pawn_play = sum(1 for r in all_recs if r[2]) / n
        mean_prior = sum(r[0] for r in all_recs) / n
        mean_visit = sum(r[1] for r in all_recs) / n
        avg_steps = total_steps / max(1, args.games)
        summary[name] = (W, L, D, pawn_play, mean_prior, mean_visit)
        print(f"{name:<10} {W:>3} {L:>3} {D:>3} | {adjW:>4} {adjL:>4} | "
              f"{pawn_play*100:>8.1f}% {mean_prior:>11.4f} {mean_visit:>11.4f} "
              f"{avg_steps:>6.1f}  ({dt:.0f}s)")
    return summary


def run_probe(model, args, device, history_length):
    """Single-position A/B on a handful of states drawn from a baseline game."""
    rng = random.Random(args.seed)
    # Generate a baseline trajectory; capture states where the model (RED) is
    # to move and is LOSING per the value head, plus the opening.
    board_size, walls = args.board_size, args.walls
    start_col = rng.randrange(board_size)
    state = initial_state(board_size, walls, start_col, Player.RED)
    history: List[BarricadeState] = []
    ab = AlphaBetaAI(Player.BLUE, max_depth=args.depth)
    base_mcts = build_mcts(model, fpu=0.33, floor=0.0, sims=args.simulations,
                           batch=args.batch_size, history_length=history_length,
                           device=device, rng=rng)
    captured: List[Tuple[int, BarricadeState, List[BarricadeState], float]] = []
    for ply in range(args.max_steps):
        if state.winner is not None or getattr(state, "is_draw", False):
            break
        legal = state.legal_actions()
        if not legal:
            break
        sb = state.copy()
        if state.current_player == Player.RED:
            value_blue = evaluate_value_head_blue_pov(
                model, sb, board_size=board_size, device=device, history=history)
            # RED losing  <=> value_blue > 0 (blue winning)
            if ply == 0 or value_blue > 0.25:
                captured.append((ply, sb, list(history), value_blue))
            hist = history[-history_length:] if history_length > 0 else ()
            action = int(base_mcts.search(sb, history=hist).action)
            if action not in legal:
                action = legal[0]
        else:
            action = ab.get_best_move(sb)
        state = apply_selected_action(state, action, legal)
        history.append(sb)
        if len(captured) >= args.probe_states + 1:
            break

    print(f"\nCaptured {len(captured)} RED-to-move states (opening + losing). "
          f"sims={args.simulations}\n")
    header = (f"{'ply':>3} {'valueB':>7} {'#pawnLegal':>10} | "
              f"{'cond':<10} {'prior_pawn':>11} {'visit_pawn':>11} "
              f"{'chosen':>7}  top-visited(action:visits)")
    for ply, st, hist, vblue in captured:
        print("=" * 100)
        n_pawn_legal = sum(1 for a in st.legal_actions() if is_pawn(a))
        print(f"ply={ply} valueB={vblue:+.3f} (RED {'LOSING' if vblue>0 else 'ok'}) "
              f"pawnLegal={n_pawn_legal} totalLegal={len(st.legal_actions())}")
        print(header)
        for name, (fpu, floor) in CONDITIONS.items():
            if args.conditions and name not in args.conditions:
                continue
            rng2 = random.Random(args.seed + 1)
            mcts = build_mcts(model, fpu=fpu, floor=floor, sims=args.simulations,
                              batch=args.batch_size, history_length=history_length,
                              device=device, rng=rng2)
            h = hist[-history_length:] if history_length > 0 else ()
            res = mcts.search(st.copy(), history=h)
            pp, vp, cp, top = measure_search(res)
            top_str = " ".join(f"{a}{'(P)' if is_pawn(a) else ''}:{v}" for a, v in top)
            print(f"{'':>3} {'':>7} {'':>10} | {name:<10} {pp:>11.4f} {vp:>11.4f} "
                  f"{('PAWN' if cp else 'wall'):>7}  {top_str}")


def run_value_sanity(model, args, device, history_length):
    """Is the value head calibrated? Raw value is side-to-move POV (+1 == the
    player to move is winning). Check obviously-decided positions."""
    bs, walls = args.board_size, args.walls

    def st(red, blue, mover):
        return BarricadeState(
            red_start=red, blue_start=blue, red_walls=walls, blue_walls=walls,
            starting_player=mover, board_size=bs,
        )

    cases = [
        ("start symmetric (RED to move)      expect ~0",
         st((0, 4), (bs - 1, 4), Player.RED)),
        ("RED 1 step from goal, RED to move  expect ~+1",
         st((bs - 2, 2), (bs - 1, 6), Player.RED)),
        ("BLUE 1 step from goal, BLUE to move expect ~+1",
         st((0, 6), (1, 2), Player.BLUE)),
        ("BLUE about to win, RED to move     expect ~-1",
         st((bs - 2, 6), (1, 2), Player.RED)),
        ("RED about to win, BLUE to move     expect ~-1",
         st((bs - 2, 2), (1, 6), Player.BLUE)),
    ]
    print("\n=== VALUE HEAD CALIBRATION (raw, side-to-move POV) ===")
    for label, state in cases:
        v = raw_value_head(model, state, board_size=bs, device=device)
        flag = ""
        print(f"  raw_value={v:+.3f}   {label}")
    mean_probe = probe_value_perspective(model, board_size=bs, device=device)
    print(f"  probe mean (near-win positions, expect >0): {mean_probe:+.3f}")


def run_strength(model, args, device, history_length):
    """Can the model beat a WEAKER opponent? Sweep Alpha-Beta depth."""
    print(f"\n=== ALPHA-BETA DEPTH SWEEP (baseline cond, sims={args.simulations}, "
          f"{args.games} games each) ===")
    print(f"{'ab_depth':>8} {'W':>3} {'L':>3} {'D':>3} | {'adjW':>4} {'adjL':>4} "
          f"| {'pawn%play':>9} {'steps':>6}")
    print("-" * 60)
    for depth in args.depths:
        rng = random.Random(args.seed)
        W = L = D = adjW = adjL = 0
        recs = []
        total_steps = 0
        t0 = time.time()
        for g in range(args.games):
            winner, adj, steps, role, r = play_one(
                model, board_size=args.board_size, walls=args.walls,
                sims=args.simulations, batch=args.batch_size, max_steps=args.max_steps,
                ab_depth=depth, device=device, rng=rng, history_length=history_length,
                game_idx=g, fpu=0.33, floor=0.0,
            )
            if winner == role:
                W += 1
            elif winner == role.opposite():
                L += 1
            else:
                D += 1
            if adj == role:
                adjW += 1
            elif adj == role.opposite():
                adjL += 1
            total_steps += steps
            recs.extend(r)
        dt = time.time() - t0
        n = max(1, len(recs))
        pawn_play = sum(1 for x in recs if x[2]) / n
        print(f"{depth:>8} {W:>3} {L:>3} {D:>3} | {adjW:>4} {adjL:>4} "
              f"| {pawn_play*100:>8.1f}% {total_steps/max(1,args.games):>6.1f}  ({dt:.0f}s)")


def run_simsweep(model, args, device, history_length):
    """Does real search help? Sweep MCTS simulations vs Alpha-Beta depth 3."""
    print(f"\n=== SIMULATION SWEEP (baseline cond, ab_depth={args.depth}, "
          f"{args.games} games each) ===")
    print(f"{'sims':>6} {'W':>3} {'L':>3} {'D':>3} | {'adjW':>4} {'adjL':>4} "
          f"| {'pawn%play':>9} {'steps':>6}")
    print("-" * 60)
    for sims in args.simlist:
        rng = random.Random(args.seed)
        W = L = D = adjW = adjL = 0
        recs = []
        total_steps = 0
        t0 = time.time()
        for g in range(args.games):
            winner, adj, steps, role, r = play_one(
                model, board_size=args.board_size, walls=args.walls,
                sims=sims, batch=args.batch_size, max_steps=args.max_steps,
                ab_depth=args.depth, device=device, rng=rng, history_length=history_length,
                game_idx=g, fpu=0.33, floor=0.0,
            )
            if winner == role:
                W += 1
            elif winner == role.opposite():
                L += 1
            else:
                D += 1
            if adj == role:
                adjW += 1
            elif adj == role.opposite():
                adjL += 1
            total_steps += steps
            recs.extend(r)
        dt = time.time() - t0
        n = max(1, len(recs))
        pawn_play = sum(1 for x in recs if x[2]) / n
        print(f"{sims:>6} {W:>3} {L:>3} {D:>3} | {adjW:>4} {adjL:>4} "
              f"| {pawn_play*100:>8.1f}% {total_steps/max(1,args.games):>6.1f}  ({dt:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model_iter_000008.pt"))
    ap.add_argument("--mode",
                    choices=("probe", "compare", "valsanity", "strength", "simsweep"),
                    default="probe")
    ap.add_argument("--depths", type=int, nargs="*", default=[1, 2, 3],
                    help="Alpha-Beta depths for --mode strength")
    ap.add_argument("--simlist", type=int, nargs="*", default=[128, 384, 768],
                    help="MCTS simulation counts for --mode simsweep")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--simulations", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--board-size", type=int, default=9)
    ap.add_argument("--walls", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe-states", type=int, default=3)
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="subset of: baseline fpu0 floor fpu0floor")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    print(f"Loading {args.checkpoint} on {device}")
    model = load_model(args.checkpoint, board_size=args.board_size, device=device)
    history_length = int(getattr(model, MODEL_HISTORY_LENGTH_ATTR, 0))
    print(f"history_length={history_length} board={args.board_size}x{args.board_size} "
          f"walls={args.walls} sims={args.simulations}")

    if args.mode == "probe":
        run_probe(model, args, device, history_length)
    elif args.mode == "compare":
        run_compare(model, args, device, history_length)
    elif args.mode == "valsanity":
        run_value_sanity(model, args, device, history_length)
    elif args.mode == "strength":
        run_value_sanity(model, args, device, history_length)
        run_strength(model, args, device, history_length)
    elif args.mode == "simsweep":
        run_simsweep(model, args, device, history_length)


if __name__ == "__main__":
    main()
