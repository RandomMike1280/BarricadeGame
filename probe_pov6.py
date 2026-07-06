import random
import torch
from barricade_env import BarricadeState, Player, MoveDirection
from mcts import MCTS, MCTSConfig
from mini_bench import MODEL_VALUE_MULTIPLIER_ATTR, raw_value_head, evaluate_value_head_blue_pov
from test_mcts import ConstantValueModel

# Find a position where BLUE can move UP to (0, 4) — BLUE wins
state = BarricadeState(red_start=(8, 4), blue_start=(1, 4), starting_player=Player.BLUE)
print(f"BLUE legal actions: {state.legal_actions()}")
print(f"BLUE dist: {state.shortest_path_length(Player.BLUE)}, RED dist: {state.shortest_path_length(Player.RED)}")

# Try a position where blue has more space
state2 = BarricadeState(red_start=(8, 4), blue_start=(1, 4), red_walls=0, blue_walls=0, starting_player=Player.BLUE)
print(f"BLUE legal actions (no walls): {state2.legal_actions()}")
for a in state2.legal_actions()[:10]:
    decoded = state2.decode_action(a)
    print(f"  action {a} -> {decoded}")