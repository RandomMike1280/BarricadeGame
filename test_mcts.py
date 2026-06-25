import random
import unittest

import torch
from torch import nn

from barricade_env import ACTION_SIZE, BarricadeEnv, HORIZONTAL_WALL_OFFSET
from mcts import MCTS, MCTSConfig


class TinyMCTSModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def inference(self, batch: torch.Tensor):
        batch_size = batch.shape[0]
        device = batch.device
        logits = torch.zeros((batch_size, ACTION_SIZE), device=device)
        values = torch.zeros((batch_size, 1), device=device)
        leads = torch.zeros((batch_size, 1), device=device)
        return logits, values, leads


def wall_heavy_state(plies: int = 20):
    rng = random.Random(321)
    env = BarricadeEnv(max_steps=100, invalid_action_mode="raise")
    env.reset()
    for _ in range(plies):
        legal_actions = env.legal_actions()
        wall_actions = [
            action for action in legal_actions if action >= HORIZONTAL_WALL_OFFSET
        ]
        if not legal_actions:
            break
        _, _, terminated, truncated, _ = env.step(
            rng.choice(wall_actions or legal_actions)
        )
        if terminated or truncated:
            break
    return env.state.copy()


class MCTSTests(unittest.TestCase):
    def test_large_batch_flushes_on_collision(self) -> None:
        model = TinyMCTSModel()
        state = wall_heavy_state()
        mcts = MCTS(
            model,
            MCTSConfig(
                num_simulations=128,
                batch_size=128,
                device="cpu",
                add_root_noise=False,
            ),
            rng=random.Random(1),
        )

        result = mcts.search(state)

        self.assertEqual(result.diagnostics["completed_simulations"], 128)
        self.assertGreater(result.diagnostics["collision_flushes"], 0)
        self.assertLess(result.diagnostics["collisions"], 256)


if __name__ == "__main__":
    unittest.main()
