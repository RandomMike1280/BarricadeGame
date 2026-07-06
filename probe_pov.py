import random
import torch
from barricade_env import BarricadeState, Player
from mcts import MCTS, MCTSConfig
from mini_bench import MODEL_VALUE_MULTIPLIER_ATTR, raw_value_head, evaluate_value_head_blue_pov
from test_mcts import ConstantValueModel, ConstantValueLeadModel

def run(label, model, state, num_sims=64, seed=13):
    print(f"=== {label} ===")
    print(f"  state.winner={state.winner!r} current_player={state.current_player.name}")
    raw = raw_value_head(model, state, board_size=9, device=torch.device("cpu"))
    print(f"  raw_value_head={raw:.6f}")
    mcts = MCTS(model, MCTSConfig(num_simulations=num_sims, batch_size=8, device="cpu", add_root_noise=False), rng=random.Random(seed))
    result = mcts.search(state)
    print(f"  _root_value={result.root_value:.6f}")
    print(f"  blue_pov={evaluate_value_head_blue_pov(model, state, board_size=9, device=torch.device('cpu')):.6f}")
    print()

model = ConstantValueModel(0.6)
setattr(model, MODEL_VALUE_MULTIPLIER_ATTR, 1.0)

run("open board RED-to-move", model, BarricadeState(red_start=(0, 4), blue_start=(8, 4), starting_player=Player.RED))
run("open board BLUE-to-move", model, BarricadeState(red_start=(0, 4), blue_start=(8, 4), starting_player=Player.BLUE))
run("terminal RED wins, RED-to-move", model, BarricadeState(red_start=(8, 4), blue_start=(0, 4), starting_player=Player.RED))
run("terminal RED wins, BLUE-to-move", model, BarricadeState(red_start=(8, 4), blue_start=(0, 4), starting_player=Player.BLUE))
run("forcing-win RED-to-move (red row 7)", model, BarricadeState(red_start=(7, 4), blue_start=(0, 4), starting_player=Player.RED))
run("forcing-win RED-to-move (red row 7) BLUE-to-move (blue row 1)",
    model, BarricadeState(red_start=(7, 4), blue_start=(1, 4), starting_player=Player.BLUE))

print("=== forcing-win with value=1.0, RED-to-move, various sims ===")
model_hi = ConstantValueModel(1.0)
setattr(model_hi, MODEL_VALUE_MULTIPLIER_ATTR, 1.0)
state_f = BarricadeState(red_start=(7, 4), blue_start=(0, 4), starting_player=Player.RED)
for n in (64, 256, 1024, 4096):
    mcts = MCTS(model_hi, MCTSConfig(num_simulations=n, batch_size=8, device="cpu", add_root_noise=False), rng=random.Random(13))
    result = mcts.search(state_f)
    print(f"  n={n}: _root_value={result.root_value:.6f}")

print()
print("=== lead test (value=0, lead=0.7), forcing-win RED-to-move ===")
model_l = ConstantValueLeadModel(0.0, 0.7)
setattr(model_l, MODEL_VALUE_MULTIPLIER_ATTR, 1.0)
for n in (64, 256, 1024, 4096):
    mcts = MCTS(model_l, MCTSConfig(num_simulations=n, batch_size=8, device="cpu", add_root_noise=False), rng=random.Random(19))
    result = mcts.search(state_f)
    print(f"  n={n}: _root_lead={result.root_lead:.6f}")