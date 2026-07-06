import random
import torch
from barricade_env import BarricadeState, Player, MoveDirection
from mcts import MCTS, MCTSConfig
from mini_bench import MODEL_VALUE_MULTIPLIER_ATTR, raw_value_head, evaluate_value_head_blue_pov
from test_mcts import ConstantValueModel, ConstantValueLeadModel

# Empirical values for various states with constant models
# Goal: pick states where _root_value and _root_lead are deterministic and sign-sensitive.

print("=== OPEN BOARD RED-to-move (mirror of BLUE-to-move) ===")
model = ConstantValueModel(0.6)
setattr(model, MODEL_VALUE_MULTIPLIER_ATTR, 1.0)

state_red = BarricadeState(red_start=(0, 4), blue_start=(8, 4), starting_player=Player.RED)
mcts = MCTS(model, MCTSConfig(num_simulations=64, batch_size=8, device="cpu", add_root_noise=False), rng=random.Random(13))
r = mcts.search(state_red)
print(f"  RED-to-move: _root_value={r.root_value:.10f}")

state_blue = BarricadeState(red_start=(0, 4), blue_start=(8, 4), starting_player=Player.BLUE)
mcts = MCTS(model, MCTSConfig(num_simulations=64, batch_size=8, device="cpu", add_root_noise=False), rng=random.Random(13))
r = mcts.search(state_blue)
print(f"  BLUE-to-move: _root_value={r.root_value:.10f}")

print()
print("=== terminal state RED-wins (BLUE-to-move after RED wins) ===")
state_red_wins = BarricadeState(red_start=(7, 4), blue_start=(0, 4), red_walls=0, blue_walls=0, starting_player=Player.RED).apply_action(MoveDirection.DOWN.value)
mcts = MCTS(model, MCTSConfig(num_simulations=64, batch_size=8, device="cpu", add_root_noise=False), rng=random.Random(13))
r = mcts.search(state_red_wins)
print(f"  RED-wins, BLUE-to-move: _root_value={r.root_value:.10f}")
print(f"  evaluate_value_head_blue_pov={evaluate_value_head_blue_pov(model, state_red_wins, board_size=9, device=torch.device('cpu')):.10f}")

print()
print("=== terminal state BLUE-wins (RED-to-move after BLUE wins) ===")
state_blue_wins = BarricadeState(red_start=(8, 4), blue_start=(1, 4), red_walls=0, blue_walls=0, starting_player=Player.BLUE).apply_action(MoveDirection.UP.value)
mcts = MCTS(model, MCTSConfig(num_simulations=64, batch_size=8, device="cpu", add_root_noise=False), rng=random.Random(13))
r = mcts.search(state_blue_wins)
print(f"  BLUE-wins, RED-to-move: _root_value={r.root_value:.10f}")
print(f"  evaluate_value_head_blue_pov={evaluate_value_head_blue_pov(model, state_blue_wins, board_size=9, device=torch.device('cpu')):.10f}")