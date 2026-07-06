"""Record the pre-PR baseline fixture for ``test_mcts.py``.

The fixture is recorded against the pre-PR (HEAD~1) ``mcts.py`` so the
``test_pre_pr_baseline_regression`` test can compare the current
``mcts.py`` (which adds ``virtual_loss``, AMAF, and the per-node
``node_path`` machinery) against an apples-to-apples reference.

Steps:
1.  Materialize the pre-PR ``mcts.py`` snapshot at
    ``tests/fixtures/mcts_clean_pre_pr.py``. Prefer ``git show
    HEAD~1:mcts.py`` from the live repo so the recorder stays correct
    across MCTS refactors. Fall back to a cached on-disk copy when git
    history is unavailable (shallow clone, offline sandbox).
2.  Load that snapshot into a SEPARATE module name
    (``mcts_pr_baseline``) so it doesn't replace the live ``mcts``
    module that's currently imported by ``test_mcts``.
3.  Build a single-threaded MCTS using the legacy config (no virtual
    loss, no AMAF), run one search, serialize the per-edge arrays to
    ``tests/fixtures/mcts_pr_baseline.npz``.
4.  The script is intentionally idempotent; re-running it produces a
    byte-identical file given the same seed.
"""
from __future__ import annotations

import importlib.util
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

# Make the repository root importable so ``mcts_pr_baseline`` can import
# ``barricade_env`` / ``network`` (the live ``mcts`` module). The recorder
# is meant to be invoked from anywhere in the repo.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_SNAPSHOT_FILENAME = "mcts_clean_pre_pr.py"
_PRE_PR_REF = "HEAD~1"
_PRE_PR_PATH_IN_TREE = "mcts.py"


def _try_git_show_snapshot() -> bytes | None:
    """Return ``git show HEAD~1:mcts.py`` bytes, or ``None`` if unavailable.

    Returns ``None`` (not raising) when git is missing, the working
    tree isn't a repo, or the ref can't be resolved. Callers decide
    whether that's fatal.
    """
    try:
        completed = subprocess.run(
            ["git", "show", f"{_PRE_PR_REF}:{_PRE_PR_PATH_IN_TREE}"],
            cwd=str(_REPO_ROOT),
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout


def _ensure_clean_snapshot() -> Path:
    """Materialize the pre-PR MCTS snapshot on disk.

    Resolution order:
    1. ``SKIP_REGEN=1`` env var + existing file on disk: use as-is
       (offline re-runs and explicit "trust the cache" mode).
    2. ``git show HEAD~1:mcts.py``: regenerate from history and write
       the bytes to ``mcts_clean_pre_pr.py``. Logs the resolved SHA
       so the recording is auditable.
    3. Disk-copy fallback: if a cached ``mcts_clean_pre_pr.py`` exists
       from a prior run, use it (covers shallow clones / detached CI).
    4. Otherwise raise with a pointer to the README.

    Returns the path to the materialized snapshot.
    """
    snapshot = Path(__file__).parent / _SNAPSHOT_FILENAME
    skip_regen = os.environ.get("SKIP_REGEN") == "1"

    if skip_regen and snapshot.exists():
        print(f"[snapshot] SKIP_REGEN=1; reusing {snapshot}")
        return snapshot

    blob = _try_git_show_snapshot()
    if blob is not None:
        # ``git show <ref>:path`` may return text or binary depending on
        # whether git detected the blob as text. ``mcts.py`` is text;
        # write bytes verbatim so line endings round-trip on Windows.
        snapshot.write_bytes(blob)
        sha_proc = subprocess.run(
            ["git", "rev-parse", _PRE_PR_REF],
            cwd=str(_REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        sha = sha_proc.stdout.strip()
        print(
            f"[snapshot] regenerated {snapshot} from "
            f"{_PRE_PR_REF} ({sha})"
        )
        return snapshot

    if snapshot.exists():
        print(
            f"[snapshot] git unavailable; using cached {snapshot}. "
            "Set SKIP_REGEN=1 to silence this message."
        )
        return snapshot

    raise RuntimeError(
        f"Cannot materialize {_SNAPSHOT_FILENAME}: git history is "
        "unavailable and no cached snapshot exists on disk. See "
        "tests/fixtures/README.md for the offline procedure."
    )


def _load_clean_mcts():
    """Load the pre-PR ``mcts.py`` from disk into a fresh module slot.

    Loading into ``mcts_pr_baseline`` instead of the live ``mcts`` module
    means the recorder sees the pre-PR MCTS API (no ``virtual_loss``,
    no ``amaf_*``, no ``node_path``) without touching the working tree.
    The live ``mcts`` module stays untouched, so the recorder can also
    import ``test_mcts.TinyMCTSModel`` from the same Python process.
    """
    clean_path = Path(__file__).parent / "mcts_clean_pre_pr.py"
    spec = importlib.util.spec_from_file_location(
        "mcts_pr_baseline", str(clean_path)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mcts_pr_baseline"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    _ensure_clean_snapshot()
    mcts_mod = _load_clean_mcts()
    MCTS = mcts_mod.MCTS
    MCTSConfig = mcts_mod.MCTSConfig

    # Import the live ``test_mcts`` helpers so the fixture state is bit-
    # identical to what ``test_mcts.py::test_pre_pr_baseline_regression``
    # will reconstruct at replay time.
    from test_mcts import TinyMCTSModel, wall_heavy_state

    seed = 42
    state = wall_heavy_state(plies=20)
    model = TinyMCTSModel()

    # Legacy single-threaded config: no virtual_loss, no AMAF. Mirrors
    # ``MCTSConfig(num_simulations=64, batch_size=64, device="cpu",
    # add_root_noise=False, virtual_loss=0.0, amaf_weight=0.0,
    # amaf_rollout_depth=0)`` against the CURRENT ``mcts.py`` (which
    # defaults ``virtual_loss=0.0`` and ``amaf_weight=0.0``).
    cfg = MCTSConfig(
        num_simulations=64,
        batch_size=64,
        device="cpu",
        add_root_noise=False,
    )
    mcts = MCTS(model, cfg, rng=random.Random(seed))
    root = mcts.search(state).root

    edges = sorted(root.edges.items())
    action_indices = np.array([a for a, _ in edges], dtype=np.int32)
    visits = np.array([e.visits for _, e in edges], dtype=np.int32)
    value_sum = np.array([e.value_sum for _, e in edges], dtype=np.float64)
    virtual_visits = np.array([e.virtual_visits for _, e in edges], dtype=np.int32)

    out = Path(__file__).parent / "mcts_pr_baseline.npz"
    np.savez(
        out,
        action_indices=action_indices,
        visits=visits,
        value_sum=value_sum,
        virtual_visits=virtual_visits,
        seed=np.int32(seed),
        num_simulations=np.int32(cfg.num_simulations),
        batch_size=np.int32(cfg.batch_size),
    )
    print(f"Wrote {out} ({len(edges)} edges)")


if __name__ == "__main__":
    main()