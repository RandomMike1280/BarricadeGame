import random
import tempfile
import unittest
from pathlib import Path

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
    decode_action,
)
from mcts import MCTS, MCTSConfig
from mini_bench import (
    MODEL_VALUE_MULTIPLIER_ATTR,
    evaluate_value_head_blue_pov,
    load_model as load_bench_model,
)
from finetune_tactical_value import (
    parameters_for_scope,
    probe_summary,
    set_training_mode_for_scope,
)
from train import (
    NetworkConfig,
    _load_model_state,
    build_model,
    tactical_value_batch,
    tactical_value_policy_batch,
)


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


class ConstantValueModel(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.value = float(value)

    def inference(self, batch: torch.Tensor):
        batch_size = batch.shape[0]
        device = batch.device
        logits = torch.zeros((batch_size, ACTION_SIZE), device=device)
        values = torch.full((batch_size, 1), self.value, device=device)
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
    def test_blue_pov_conversion_from_side_to_move_value(self) -> None:
        model = ConstantValueModel(0.75)
        setattr(model, MODEL_VALUE_MULTIPLIER_ATTR, 1.0)
        red_turn = BarricadeState(
            red_start=(4, 4),
            blue_start=(8, 4),
            starting_player=Player.RED,
        )
        blue_turn = BarricadeState(
            red_start=(0, 4),
            blue_start=(4, 4),
            starting_player=Player.BLUE,
        )

        self.assertAlmostEqual(
            evaluate_value_head_blue_pov(
                model, red_turn, board_size=9, device=torch.device("cpu")
            ),
            -0.75,
        )
        self.assertAlmostEqual(
            evaluate_value_head_blue_pov(
                model, blue_turn, board_size=9, device=torch.device("cpu")
            ),
            0.75,
        )

    def test_mini_bench_loads_flat_policy_checkpoint_without_random_head(self) -> None:
        model = build_model(
            NetworkConfig(
                history_length=0,
                conv_channels=8,
                num_residual_layers=0,
                value_hidden_size=8,
                policy_head_type="flat",
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flat.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "network_config": {
                        "history_length": 0,
                        "conv_channels": 8,
                        "residual_channels": None,
                        "num_conv_layers": 1,
                        "num_residual_layers": 0,
                        "value_hidden_size": 8,
                    },
                },
                path,
            )

            loaded = load_bench_model(path, board_size=9, device=torch.device("cpu"))

        self.assertEqual(getattr(loaded, "policy_head_type"), "flat")

    def test_policy_head_mismatch_requires_explicit_reset(self) -> None:
        flat = build_model(
            NetworkConfig(
                history_length=0,
                conv_channels=8,
                num_residual_layers=0,
                value_hidden_size=8,
                policy_head_type="flat",
            )
        )
        factored = build_model(
            NetworkConfig(
                history_length=0,
                conv_channels=8,
                num_residual_layers=0,
                value_hidden_size=8,
                policy_head_type="factored",
            )
        )

        with self.assertRaises(RuntimeError):
            _load_model_state(factored, flat.state_dict())
        _load_model_state(factored, flat.state_dict(), reset_policy_head=True)
        self.assertTrue(
            torch.equal(
                factored.policy_head.move_fc.weight,
                flat.policy_head.fc.weight[factored.policy_head.move_index],
            )
        )
        self.assertTrue(
            torch.equal(
                factored.policy_head.wall_fc.bias,
                flat.policy_head.fc.bias[factored.policy_head.wall_index],
            )
        )
        self.assertTrue(torch.equal(factored.policy_head.type_fc.weight, torch.zeros_like(factored.policy_head.type_fc.weight)))

    def test_tactical_value_batch_targets_match_shortest_race(self) -> None:
        rng = random.Random(7)
        planes, targets = tactical_value_batch(
            batch_size=32,
            history_length=2,
            rng=rng,
            device=torch.device("cpu"),
        )

        self.assertEqual(planes.shape, (32, 27, 9, 9))
        self.assertEqual(targets.shape, (32,))
        self.assertTrue(torch.all((targets == 1.0) | (targets == -1.0)))
        for planes_i, target in zip(planes, targets):
            own = torch.nonzero(planes_i[1] > 0.5)[0]
            opp = torch.nonzero(planes_i[2] > 0.5)[0]
            own_distance = 8 - int(own[0])
            opp_distance = int(opp[0])
            expected = 1.0 if own_distance < opp_distance else -1.0
            self.assertEqual(float(target.item()), expected)

    def test_tactical_policy_batch_targets_forward_move_when_winning(self) -> None:
        rng = random.Random(11)
        planes, value_targets, policy_targets, policy_mask = tactical_value_policy_batch(
            batch_size=64,
            history_length=0,
            rng=rng,
            device=torch.device("cpu"),
        )

        self.assertEqual(planes.shape, (64, 9, 9, 9))
        self.assertTrue(torch.all(policy_targets == MoveDirection.DOWN.value))
        self.assertTrue(torch.all(policy_mask[value_targets < 0.0] == 0.0))
        self.assertGreater(float(policy_mask.sum().item()), 0.0)

    def test_probe_summary_uses_eval_mode_without_leaving_it_changed(self) -> None:
        model = build_model(
            NetworkConfig(
                history_length=0,
                conv_channels=8,
                num_residual_layers=0,
                value_hidden_size=8,
                policy_head_type="flat",
            )
        )
        model.train()

        values = probe_summary(model, device=torch.device("cpu"))

        self.assertTrue(model.training)
        self.assertIn("RED wins next move", values)

    def test_value_head_train_scope_freezes_policy_and_trunk(self) -> None:
        model = build_model(
            NetworkConfig(
                history_length=0,
                conv_channels=8,
                num_residual_layers=0,
                value_hidden_size=8,
                policy_head_type="flat",
            )
        )

        trainable = parameters_for_scope(model, "value-head")

        self.assertEqual(set(trainable), set(model.value_head.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.value_head.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.policy_head.parameters()))

    def test_value_head_train_scope_keeps_frozen_batchnorm_in_eval(self) -> None:
        model = build_model(
            NetworkConfig(
                history_length=0,
                conv_channels=8,
                num_residual_layers=0,
                value_hidden_size=8,
                policy_head_type="flat",
            )
        )

        parameters_for_scope(model, "value-head")
        set_training_mode_for_scope(model, "value-head")

        self.assertFalse(model.conv_tower.training)
        self.assertFalse(model.policy_head.training)
        self.assertTrue(model.value_head.training)

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

    def test_apply_move_rejects_move_onto_opponent(self) -> None:
        state = BarricadeState(
            red_start=(4, 4),
            blue_start=(5, 4),
            red_walls=0,
            blue_walls=0,
            starting_player=Player.RED,
        )

        self.assertEqual(decode_action(MoveDirection.DOWN.value), ("move", MoveDirection.DOWN))

        with self.assertRaises(ValueError):
            state.apply_move(decode_action(MoveDirection.DOWN.value))

        next_state = state.apply_action(MoveDirection.DOWN.value)
        self.assertEqual(next_state.pawns[Player.RED], (6, 4))
        self.assertEqual(next_state.pawns[Player.BLUE], (5, 4))

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
