"""Throwaway probe: does the value head see the pawn race? (read-only diagnosis)"""
import torch
from barricade_env import BarricadeState, Player
from mini_bench import load_model, raw_value_head, MODEL_HISTORY_LENGTH_ATTR
from network import encode_state_stack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(__import__("pathlib").Path("checkpoints/latest.pt"), board_size=9, device=device)
hist_len = int(getattr(model, MODEL_HISTORY_LENGTH_ATTR, 0))
print(f"history_length={hist_len}, device={device}")


def raw_val_and_lead(state):
    sp = encode_state_stack(state, (), history_length=hist_len).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits, value, lead = model.inference(sp)
    return float(value.reshape(-1)[0]), float(lead.reshape(-1)[0])


def mk(red_rc, blue_rc, stm, rw=10, bw=10):
    return BarricadeState(red_start=red_rc, blue_start=blue_rc, red_walls=rw,
                          blue_walls=bw, starting_player=stm, board_size=9)


print("\n=== RED-to-move, walls 10/10, BLUE fixed far at (6,0) [BLUE 6 from its goal] ===")
print("RED goal=row8. RED distance shrinks as row grows. side-to-move=RED.")
print(f"{'red_row':>7} {'dist_to_goal':>12} {'raw_value(RED POV)':>18} {'lead':>8}")
for r in range(1, 8):
    s = mk((r, 4), (6, 0), Player.RED)
    v, ld = raw_val_and_lead(s)
    print(f"{r:>7} {8 - r:>12} {v:>18.3f} {ld:>8.3f}")

print("\n=== BLUE-to-move, walls 10/10, RED fixed far at (2,4) [RED 6 from its goal] ===")
print("BLUE goal=row0. BLUE distance shrinks as row shrinks. side-to-move=BLUE.")
print(f"{'blue_row':>8} {'dist_to_goal':>12} {'raw_value(BLUE POV)':>19} {'lead':>8}")
for r in range(7, 0, -1):
    s = mk((2, 4), (r, 0), Player.BLUE)
    v, ld = raw_val_and_lead(s)
    print(f"{r:>8} {r:>12} {v:>19.3f} {ld:>8.3f}")

print("\n=== User scenario 3: RED=(0,1) BLUE=(1,0), RED to move, walls 10/10 ===")
print("BLUE is 1 from its goal (row0). RED is 8 from its goal. RED is LOST.")
s = mk((0, 1), (1, 0), Player.RED)
v, ld = raw_val_and_lead(s)
print(f"raw_value(RED side-to-move POV) = {v:+.3f}  lead={ld:+.3f}   (should be ~ -1)")

print("\n=== Clean test: RED 1-from-goal, RED to move, walls 10/10 (RED wins next move) ===")
s = mk((7, 4), (2, 0), Player.RED)
v, ld = raw_val_and_lead(s)
print(f"raw_value(RED side-to-move POV) = {v:+.3f}  lead={ld:+.3f}   (should be ~ +1)")

print("\n=== Sanity: vary ONLY walls_left for side-to-move, pawns fixed mid-board ===")
print("RED=(4,4) BLUE=(4,0), RED to move. Sweep RED walls_left 0..10 (BLUE fixed 10).")
print(f"{'red_walls':>9} {'raw_value(RED POV)':>18} {'lead':>8}")
for w in (0, 2, 5, 8, 10):
    s = mk((4, 4), (4, 0), Player.RED, rw=w, bw=10)
    v, ld = raw_val_and_lead(s)
    print(f"{w:>9} {v:>18.3f} {ld:>8.3f}")
