import random
import torch
from barricade_env import BarricadeState, Player, MoveDirection
from mcts import MCTS, MCTSConfig
from mini_bench import MODEL_VALUE_MULTIPLIER_ATTR, raw_value_head, evaluate_value_head_blue_pov
from test_mcts import ConstantValueModel

# Use no walls so the path is open
state = BarricadeState(red_start=(8, 4), blue_start=(1, 4), red_walls=0, blue_walls=0, starting_player=Player.BLUE)
print(f"BLUE dist: {state.shortest_path_length(Player.BLUE)}, RED dist: {state.shortest_path_length(Player.RED)}")
state_term = state.apply_action(MoveDirection.UP.value)
print(f"After BLUE UP: winner={state_term.winner!r}, current_player={state_term.current_player.name}")

model = ConstantValueModel(0.6)
setattr(model, MODEL_VALUE_MULTIPLIER_ATTR, 1.0)
mcts = MCTS(model, MCTSConfig(num_simulations=64, batch_size=8, device="cpu", add_root_noise=False), rng=random.Random(13))
result = mcts.search(state_term)
print(f"  _root_value={result.root_value:.4f}")
print(f"  blue_pov={evaluate_value_head_blue_pov(model, state_term, board_size=9, device=torch.device('cpu')):.4f}")
print(f"  raw_value_head={raw_value_head(model, state_term, board_size=9, device=torch.device('cpu')):.4f}")