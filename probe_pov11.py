import random
import torch
from barricade_env import BarricadeState, Player, MoveDirection
from mcts import MCTS, MCTSConfig
from mini_bench import MODEL_VALUE_MULTIPLIER_ATTR
from test_mcts import ConstantValueModel, ConstantValueLeadModel

# Probe with various seeds to find a stable test value
for seed in [13, 19, 23, 42]:
    model = ConstantValueModel(0.6)
    setattr(model, MODEL_VALUE_MULTIPLIER_ATTR, 1.0)
    state = BarricadeState(red_start=(0, 4), blue_start=(8, 4), starting_player=Player.RED)
    mcts = MCTS(model, MCTSConfig(num_simulations=64, batch_size=8, device="cpu", add_root_noise=False), rng=random.Random(seed))
    r = mcts.search(state)
    print(f"seed={seed}: _root_value={r.root_value:.10f}")