import importlib
import multiprocessing as mp
import tempfile
import unittest
from pathlib import Path

import train


_METRICS = {
    "games": 8,
    "wins_when_red_starts": 3,
    "wins_when_blue_starts": 2,
    "draws": 1,
    "games_when_red_starts": 4,
    "games_when_blue_starts": 4,
    "wins": 5,
    "losses": 2,
    "win_rate": 0.5625,
    "average_lead": 0.5,
    "average_game_length": 40.0,
}


def _child_append(path_str: str, iteration: int) -> None:
    """Worker entry point for the cross-process test.

    Runs inside a fresh interpreter (spawn context) so it has its own
    module state and lock objects; the shared state is only the on-disk
    file at ``path_str``.
    """
    importlib.reload(train)
    train.TRAIN_HISTORY_PATH = Path(path_str)
    train._append_train_history(iteration=iteration, metrics=_METRICS)


class TestAppendTrainHistory(unittest.TestCase):
    def test_writes_valid_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_path = train.TRAIN_HISTORY_PATH
            train.TRAIN_HISTORY_PATH = Path(tmp) / "train_log_history.md"
            try:
                train._append_train_history(
                    iteration=1,
                    metrics=_METRICS,
                )
                self.assertTrue(train.TRAIN_HISTORY_PATH.exists())
                contents = train.TRAIN_HISTORY_PATH.read_text(encoding="utf-8")
                self.assertIn("wins_when_red_starts", contents)
                self.assertIn("| 1 |", contents)
            finally:
                train.TRAIN_HISTORY_PATH = old_path


class TestAppendTrainHistoryConcurrent(unittest.TestCase):
    def test_two_processes_produce_well_formed_file(self):
        """Two simultaneous writers must yield one header and both rows."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_log_history.md"
            ctx = mp.get_context("spawn")
            procs = [
                ctx.Process(target=_child_append, args=(str(path), i + 1))
                for i in range(2)
            ]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout=30)
                self.assertEqual(p.exitcode, 0)
            contents = path.read_text(encoding="utf-8")
            header_count = sum(
                1
                for line in contents.splitlines()
                if line.startswith("| iteration")
            )
            self.assertEqual(
                header_count,
                1,
                f"expected exactly one header row, got {header_count}:\n{contents}",
            )
            self.assertIn("| 1 |", contents)
            self.assertIn("| 2 |", contents)


if __name__ == "__main__":
    unittest.main()
