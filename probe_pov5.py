import random
import torch
from barricade_env import BarricadeState, Player, MoveDirection
from mcts import MCTS, MCTSConfig
from mini_bench import MODEL_VALUE_MULTIPLIER_ATTR, raw_value_head
from test_mcts import ConstantValueLeadModel

# Probe in detail
model = ConstantValueLeadModel(0.0, 0.7)
setattr(model, MODEL_VALUE_MULTIPLIER_ATTR, 1.0)

# BLUE-to-move forcing-win (BLUE at row 1, RED at row 0)
state = BarricadeState(red_start=(0, 4), blue_start=(1, 4), starting_player=Player.BLUE)
print(f"BLUE-to-move, blue dist: {state.shortest_path_length(Player.BLUE)}, red dist: {state.shortest_path_length(Player.RED)}")
# Move BLUE UP to (0, 4)
state_term = state.apply_action(MoveDirection.DOWN.value)
print(f"After BLUE UP: winner={state_term.winner!r}, current_player={state_term.current_player.name}")

mcts = MCTS(model, MCTSConfig(num_simulations=64, batch_size=8, device="cpu", add_root_noise=False), rng=random.Random(19))
result = mcts.search(state)
print(f"_root_lead={result.root_lead:.6f}")
print(f"_root_value={result.root_value:.6f}")
print(f"# root edges:")
for action, edge in sorted(result.root.edges.items()):
    print(f"  action={action}: visits={edge.visits}, value_sum={edge.value_sum:.4f}, lead_sum={edge.lead_sum:.4f}")
print(f"total visits: {sum(e.visits for e in result.root.edges.values())}")
print(f"total value_sum: {sum(e.value_sum for e in result.root.edges.values()):.4f}")
print(f"total lead_sum: {sum(e.lead_sum for e in result.root.edges.values()):.4f}")
print(f"computed root_lead: {sum(e.lead_sum for e in result.root.edges.values()) / sum(e.visits for e in result.root.edges.values()):.4f}")