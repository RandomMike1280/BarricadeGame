import random
import unittest

import torch
from torch import nn

from barricade_env import (
    ACTION_SIZE,
    BarricadeEnv,
    BarricadeState,
    DIAGONAL_HOP_OFFSET,
    HORIZONTAL_WALL_OFFSET,
    MoveDirection,
    Player,
    WallOrientation,
)
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
    def test_adjacent_pawn_can_jump_straight_when_unblocked(self) -> None:
        state = BarricadeState(
            red_start=(4, 4),
            blue_start=(5, 4),
            red_walls=0,
            blue_walls=0,
            starting_player=Player.RED,
        )

        pawn_action_moves = [
            (action, move) for action, move in state.legal_action_moves() if action < DIAGONAL_HOP_OFFSET
        ]

        self.assertIn((MoveDirection.DOWN.value, ("move_to", 6, 4)), pawn_action_moves)
        next_state = state.apply_action(MoveDirection.DOWN.value)
        self.assertEqual(next_state.pawns[Player.RED], (6, 4))

    def test_adjacent_pawn_can_side_hop_when_straight_jump_blocked(self) -> None:
        state = BarricadeState(
            red_start=(4, 4),
            blue_start=(5, 4),
            red_walls=0,
            blue_walls=0,
            starting_player=Player.RED,
        )
        state.walls.add((WallOrientation.HORIZONTAL, 5, 4))
        state._walls_frozenset = frozenset(state.walls)

        pawn_action_moves = [
            (action, move)
            for action, move in state.legal_action_moves()
            if action < 4 or action >= DIAGONAL_HOP_OFFSET
        ]

        self.assertIn((DIAGONAL_HOP_OFFSET + 2, ("move_to", 5, 3)), pawn_action_moves)
        self.assertIn((DIAGONAL_HOP_OFFSET + 3, ("move_to", 5, 5)), pawn_action_moves)
        self.assertNotIn(MoveDirection.DOWN.value, state.legal_actions())
        self.assertEqual(
            state.apply_action(DIAGONAL_HOP_OFFSET + 2).pawns[Player.RED],
            (5, 3),
        )

    def test_side_hop_requires_unblocked_side_edge(self) -> None:
        state = BarricadeState(
            red_start=(4, 4),
            blue_start=(5, 4),
            red_walls=0,
            blue_walls=0,
            starting_player=Player.RED,
        )
        state.walls.update(
            {
                (WallOrientation.HORIZONTAL, 5, 4),
                (WallOrientation.VERTICAL, 5, 3),
            }
        )
        state._walls_frozenset = frozenset(state.walls)

        self.assertNotIn(DIAGONAL_HOP_OFFSET + 2, state.legal_actions())
        self.assertIn(DIAGONAL_HOP_OFFSET + 3, state.legal_actions())

    def test_threefold_repetition_is_draw_terminal(self) -> None:
        state = BarricadeState(red_start=(0, 0), blue_start=(8, 8))
        actions = [
            MoveDirection.DOWN.value,
            MoveDirection.UP.value,
            MoveDirection.UP.value,
            MoveDirection.DOWN.value,
            MoveDirection.DOWN.value,
            MoveDirection.UP.value,
            MoveDirection.UP.value,
            MoveDirection.DOWN.value,
        ]

        for action in actions:
            self.assertFalse(state.is_draw)
            state = state.apply_action(action)

        self.assertTrue(state.is_draw)
        self.assertEqual(state.draw_reason, "threefold_repetition")
        self.assertIsNone(state.winner)
        self.assertEqual(state.legal_actions(), [])

    def test_env_step_reports_repetition_draw(self) -> None:
        env = BarricadeEnv(
            red_start=(0, 0),
            blue_start=(8, 8),
            max_steps=100,
            invalid_action_mode="raise",
        )
        env.reset()
        actions = [
            MoveDirection.DOWN.value,
            MoveDirection.UP.value,
            MoveDirection.UP.value,
            MoveDirection.DOWN.value,
            MoveDirection.DOWN.value,
            MoveDirection.UP.value,
            MoveDirection.UP.value,
            MoveDirection.DOWN.value,
        ]

        for action in actions[:-1]:
            _, _, terminated, truncated, _ = env.step(action)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
        _, reward, terminated, truncated, info = env.step(actions[-1])

        self.assertEqual(reward, 0.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertIsNone(info["winner"])
        self.assertTrue(info["draw"])
        self.assertEqual(info["draw_reason"], "threefold_repetition")

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
