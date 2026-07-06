# Pre-PR MCTS Baseline Fixture

This directory holds the regression-test fixture used by
[`test_mcts.py::MCTSTests::test_pre_pr_baseline_regression`](../../test_mcts.py)
to verify the current `mcts.py` reproduces the pre-PR behavior on the
default-off virtual-loss / AMAF path.

## What's in the fixture

`mcts_pr_baseline.npz` contains per-edge arrays recorded from a
single-threaded `MCTS.search` run on a known seed and config:

| Array            | dtype    | Description                                      |
|------------------|----------|--------------------------------------------------|
| `action_indices` | `int32`  | Legal-action IDs at the root, ascending order    |
| `visits`         | `int32`  | Per-edge real-visit counts after the search      |
| `value_sum`      | `float64`| Per-edge accumulated values (zero for this test) |
| `virtual_visits` | `int32`  | Per-edge virtual-visit counts (zero pre-PR)      |
| `seed`           | `int32`  | RNG seed used for the recording (currently 42)   |
| `num_simulations`| `int32`  | `MCTSConfig.num_simulations` used                |
| `batch_size`     | `int32`  | `MCTSConfig.batch_size` used                      |

The arrays are recorded against a `wall_heavy_state(plies=20)` built
from `test_mcts.wall_heavy_state`, a `TinyMCTSModel` (constant zero
logits + zero value), and `MCTSConfig(num_simulations=64, batch_size=64,
device="cpu", add_root_noise=False)`. The test re-runs this exact
config against the **current** `mcts.py` (which defaults
`virtual_loss=0.0` and `amaf_weight=0.0`) and asserts the per-edge
arrays match bit-for-bit (`1e-6` slack on `value_sum`).

## How to regenerate the fixture

The recorder loads the **pre-PR** `mcts.py` (the parent commit of the
current Task 1 / Task 2 work) into a SEPARATE module name so the
working-tree `mcts.py` is never replaced. This makes the recorder
safe to run from the current branch without `git stash` / `git stash
pop` dance.

### Pre-PR snapshot lifecycle

`mcts_clean_pre_pr.py` is **not committed** — it lives in
`.gitignore`. The recorder regenerates it from git history at run time
via `git show HEAD~1:mcts.py`, which is the parent of the current
branch tip. That ref is the recording-time definition of "pre-PR
state"; if a future refactor renames or splits `mcts.py`, the recorder
must be updated alongside.

Re-record on a checkout with full git history:

```bash
python tests/fixtures/_record_pr_baseline.py
```

The script writes `tests/fixtures/mcts_clean_pre_pr.py` (gitignored)
and `tests/fixtures/mcts_pr_baseline.npz` (committed) next to itself,
then prints the resolved pre-PR SHA and the edge count.

### Offline / shallow-clone fallback

If git history is unavailable (shallow clone, sandboxed CI without
`.git`), do one of:

- Pre-populate `tests/fixtures/mcts_clean_pre_pr.py` from a known-good
  SHA, then run with `SKIP_REGEN=1`:

  ```bash
  git show <sha>:mcts.py > tests/fixtures/mcts_clean_pre_pr.py
  SKIP_REGEN=1 python tests/fixtures/_record_pr_baseline.py
  ```

- Or leave a cached `mcts_clean_pre_pr.py` from a prior run in place
  and let the recorder auto-detect the fallback (it logs a notice).

### Commit

```bash
git add tests/fixtures/mcts_pr_baseline.npz tests/fixtures/_record_pr_baseline.py tests/fixtures/README.md
git commit -m "Update pre-PR MCTS baseline regression fixture"
```

Do **not** add `tests/fixtures/mcts_clean_pre_pr.py`; it stays
gitignored and is regenerated from history on demand. The `.npz`
fixture is the only binary that needs to ship with the test suite.

## Why a pre-PR fixture instead of asserting against computed values

`mcts.py` changes its default behavior across commits (root noise
alpha, exploration fraction, PUCT constants, etc.). A frozen fixture
recorded from the last known-good commit lets the test catch even
benign-looking drift on the default-off path — the trainer relies on
bit-identical legacy behavior when `virtual_loss=0.0` to keep
self-play and replay buffers consistent across model reloads.

If you intentionally change the default-off behavior:

1. Update the fixture by re-running the recorder against the new
   pre-PR state (`HEAD~1` after your change lands).
2. Re-record by re-running `_record_pr_baseline.py`.
3. Commit the new fixture alongside the behavior change.

## Files in this directory

| File                          | Purpose                                            |
|-------------------------------|----------------------------------------------------|
| `mcts_pr_baseline.npz`        | The fixture (loaded by the regression test)        |
| `mcts_clean_pre_pr.py`        | Pre-PR `mcts.py` snapshot used by the recorder     |
| `_record_pr_baseline.py`      | Stand-alone recorder script                        |
| `README.md`                   | This file                                          |