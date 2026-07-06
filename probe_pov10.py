import random
import torch
from barricade_env import BarricadeState, Player, MoveDirection
from mcts import MCTS, MCTSConfig
from mini_bench import MODEL_VALUE_MULTIPLIER_ATTR
from test_mcts import ConstantValueLeadModel

model = ConstantValueLeadModel(0.0, 0.7)
setattr(model, MODEL_VALUE_MULTIPLIER_ATTR, 1.0)

# Open board, RED-to-move vs BLUE-to-move (mirror)
state_red = BarricadeState(red_start=(0, 4), blue_start=(8, 4), starting_player=Player.RED)
state_blue = BarricadeState(red_start=(8, 4), blue_start=(0, 4), starting_player=Player.BLUE)

mcts_red = MCTS(model, MCTSConfig(num_simulations=64, batch_size=8, device="cpu", add_root_noise=False), rng=random.Random(13))
r_red = mcts_red.search(state_red)
print(f"open board, RED-to-move (red=0,4 / blue=8,4): _root_lead={r_red.root_lead:.10f}")

mcts_blue = MCTS(model, MCTSConfig(num_simulations=64, batch_size=8, device="cpu", add_root_noise=False), rng=random.Random(13))
r_blue = mcts_blue.search(state_blue)
print(f"open board, BLUE-to-move (mirror: red=8,4 / blue=0,4): _root_lead={r_blue.root_lead:.10f}")

print()
# Forcing-win RED-to-move vs BLUE-to-move (mirror)
# RED-to-move, RED wins (red=7,4 / blue=0,4)
state_red_win = BarricadeState(red_start=(7, 4), blue_start=(0, 4), starting_player=Player.RED)
# BLUE-to-move, BLUE wins (mirror: red=0,4 / blue=1,4)
state_blue_win = BarricadeState(red_start=(0, 4), blue_start=(1, 4), starting_player=Player.BLUE)

mcts_red = MCTS(model, MCTSConfig(num_simulations=64, batch_size=8, device="cpu", add_root_noise=False), rng=random.Random(13))
r_red = mcts_red.search(state_red_win)
print(f"forcing-win, RED-to-move (red=7,4 / blue=0,4): _root_lead={r_red.root_lead:.10f}")

mcts_blue = MCTS(model, MCTSConfig(num_simulations=64, batch_size=8, device="cpu", add_root_noise=False), rng=random.Random(13))
r_blue = mcts_blue.search(state_blue_win)
print(f"forcing-win, BLUE-to-move (mirror: red=0,4 / blue=1,4): _root_lead={r_blue.root_lead:.10f}")