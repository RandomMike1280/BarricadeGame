import random
import torch
from barricade_env import BarricadeState, Player, MoveDirection
from mcts import MCTS, MCTSConfig
from mini_bench import MODEL_VALUE_MULTIPLIER_ATTR, raw_value_head
from test_mcts import ConstantValueLeadModel

model = ConstantValueLeadModel(0.0, 0.7)
setattr(model, MODEL_VALUE_MULTIPLIER_ATTR, 1.0)

state_red_wins = BarricadeState(red_start=(7, 4), blue_start=(0, 4), red_walls=0, blue_walls=0, starting_player=Player.RED).apply_action(MoveDirection.DOWN.value)
print(f"terminal RED-wins (BLUE-to-move): _state_lead={state_red_wins._state_lead if hasattr(state_red_wins, '_state_lead') else 'N/A'}")
mcts = MCTS(model, MCTSConfig(num_simulations=64, batch_size=8, device="cpu", add_root_noise=False), rng=random.Random(13))
r = mcts.search(state_red_wins)
print(f"  _root_value={r.root_value:.4f}, _root_lead={r.root_lead:.4f}")

state_blue_wins = BarricadeState(red_start=(8, 4), blue_start=(1, 4), red_walls=0, blue_walls=0, starting_player=Player.BLUE).apply_action(MoveDirection.UP.value)
mcts = MCTS(model, MCTSConfig(num_simulations=64, batch_size=8, device="cpu", add_root_noise=False), rng=random.Random(13))
r = mcts.search(state_blue_wins)
print(f"terminal BLUE-wins (RED-to-move): _root_value={r.root_value:.4f}, _root_lead={r.root_lead:.4f}")