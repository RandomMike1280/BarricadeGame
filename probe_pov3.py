import random
import torch
from barricade_env import BarricadeState, Player, MoveDirection
from mcts import MCTS, MCTSConfig
from mini_bench import MODEL_VALUE_MULTIPLIER_ATTR, raw_value_head, evaluate_value_head_blue_pov
from test_mcts import ConstantValueModel

# BLUE starts at (1, 4), moves UP to (0, 4) = BLUE wins
state_blue_won = BarricadeState(red_start=(8, 4), blue_start=(1, 4), starting_player=Player.BLUE).apply_action(MoveDirection.UP.value)
print(f"After BLUE UP: winner={state_blue_won.winner!r}, current_player={state_blue_won.current_player.name}")
print(f"  is_terminal={state_blue_won.is_terminal()}, legal_actions count={len(state_blue_won.legal_actions())}")

model = ConstantValueModel(0.6)
setattr(model, MODEL_VALUE_MULTIPLIER_ATTR, 1.0)
mcts = MCTS(model, MCTSConfig(num_simulations=64, batch_size=8, device="cpu", add_root_noise=False), rng=random.Random(13))
result = mcts.search(state_blue_won)
print(f"  _root_value={result.root_value:.4f}")
print(f"  blue_pov={evaluate_value_head_blue_pov(model, state_blue_won, board_size=9, device=torch.device('cpu')):.4f}")

# Now: test root_value_explicit_blue_pov_flip
print()
print("=== Test 3: terminal RED-wins, BLUE-to-move ===")
state_terminal = BarricadeState(red_start=(7, 4), blue_start=(0, 4), starting_player=Player.RED).apply_action(MoveDirection.DOWN.value)
print(f"winner={state_terminal.winner!r}, current_player={state_terminal.current_player.name}")
result_terminal = mcts.search(state_terminal)
print(f"  _root_value={result_terminal.root_value:.4f}")
print(f"  blue_pov={evaluate_value_head_blue_pov(model, state_terminal, board_size=9, device=torch.device('cpu')):.4f}")