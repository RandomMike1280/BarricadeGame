import random
import torch
from barricade_env import BarricadeState, Player, MoveDirection
from mcts import MCTS, MCTSConfig
from mini_bench import MODEL_VALUE_MULTIPLIER_ATTR, raw_value_head, evaluate_value_head_blue_pov
from test_mcts import ConstantValueModel, ConstantValueLeadModel

# Build a forcing-win terminal state by applying a winning move.
# RED starts at (7,4), moves DOWN to (8,4) = RED wins.
state_start = BarricadeState(red_start=(7, 4), blue_start=(0, 4), starting_player=Player.RED)
state_terminal = state_start.apply_action(MoveDirection.DOWN.value)
print(f"After RED DOWN: winner={state_terminal.winner!r}, current_player={state_terminal.current_player.name}")
print(f"  is_terminal={state_terminal.is_terminal()}, legal_actions count={len(state_terminal.legal_actions())}")

model = ConstantValueModel(0.6)
setattr(model, MODEL_VALUE_MULTIPLIER_ATTR, 1.0)

print(f"raw_value_head on terminal RED-wins, RED-to-move: {raw_value_head(model, state_terminal, board_size=9, device=torch.device('cpu')):.4f}")
mcts = MCTS(model, MCTSConfig(num_simulations=64, batch_size=8, device="cpu", add_root_noise=False), rng=random.Random(13))
result = mcts.search(state_terminal)
print(f"  _root_value={result.root_value:.4f}")
print(f"  blue_pov={evaluate_value_head_blue_pov(model, state_terminal, board_size=9, device=torch.device('cpu')):.4f}")

print()
print("=== Same but BLUE-to-move (we need to flip turn) ===")
# The terminal state above has current_player still RED (since RED just moved). For BLUE-to-move, we need to construct a position where RED already won and it's BLUE's turn (which only makes sense if a search simulates an opponent move after RED's win, but in reality that's a terminal state).
# Use a different approach: BLUE's terminal win position, BLUE-to-move.
state_blue_won = BarricadeState(red_start=(0, 4), blue_start=(7, 4), starting_player=Player.BLUE).apply_action(MoveDirection.UP.value)
print(f"After BLUE UP: winner={state_blue_won.winner!r}, current_player={state_blue_won.current_player.name}")
print(f"  is_terminal={state_blue_won.is_terminal()}, legal_actions count={len(state_blue_won.legal_actions())}")
print(f"  _root_value={MCTS(model, MCTSConfig(num_simulations=64, batch_size=8, device='cpu', add_root_noise=False), rng=random.Random(13)).search(state_blue_won).root_value:.4f}")
print(f"  blue_pov={evaluate_value_head_blue_pov(model, state_blue_won, board_size=9, device=torch.device('cpu')):.4f}")