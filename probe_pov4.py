import random
import torch
from barricade_env import BarricadeState, Player
from mcts import MCTS, MCTSConfig
from mini_bench import MODEL_VALUE_MULTIPLIER_ATTR, raw_value_head, evaluate_value_head_blue_pov
from test_mcts import ConstantValueModel, ConstantValueLeadModel

model = ConstantValueLeadModel(0.0, 0.7)
setattr(model, MODEL_VALUE_MULTIPLIER_ATTR, 1.0)

# Forcing-win RED-to-move
state = BarricadeState(red_start=(7, 4), blue_start=(0, 4), starting_player=Player.RED)
print("forcing-win RED-to-move, various num_simulations")
for n in (64, 256, 1024, 4096, 16384):
    mcts = MCTS(model, MCTSConfig(num_simulations=n, batch_size=8, device="cpu", add_root_noise=False), rng=random.Random(19))
    result = mcts.search(state)
    print(f"  n={n}: _root_lead={result.root_lead:.6f}")

print()
# Symmetric: BLUE-to-move, BLUE wins
state2 = BarricadeState(red_start=(0, 4), blue_start=(1, 4), starting_player=Player.BLUE)
print("forcing-win BLUE-to-move (mirror), various num_simulations")
for n in (64, 256, 1024, 4096, 16384):
    mcts = MCTS(model, MCTSConfig(num_simulations=n, batch_size=8, device="cpu", add_root_noise=False), rng=random.Random(19))
    result = mcts.search(state2)
    print(f"  n={n}: _root_lead={result.root_lead:.6f}")