# MCTS virtual-loss audit & fix notes

## Before this PR

`mcts.py` already implements **Lc0-style unscored virtual visits** for the
intra-search batching path: `SearchEdge.virtual_visits` is incremented in
`_select_leaf` (line 442 in the post-PR file), decremented in
`_backpropagate` and `_release_virtual_visits`, and is consumed only as
`effective_visits = visits + virtual_visits` in PUCT exploration (lines 489,
522). `value_sum` / `Q` is left untouched until backprop — exactly Lc0
semantics, not ELF OpenGo.

The legacy `node.in_flight` marker is set only on the **leaf** returned by
`_select_leaf` (line 488). It prevents a single search from re-selecting the
same unexpanded node twice within one batched forward pass (the
"collision" path), but it does NOT prevent two concurrent threads that
share a search tree from passing through the same *internal* node on
different edges during the GPU-forward release window. This is the §4.1
collapse mode: simultaneous visits average to ~0 and the value estimate
collapses.

Worse, the `edge.virtual_visits += 1` and `node.in_flight` mutations are
non-atomic under CPython's GIL (LOAD_ATTR / BINARY_ADD / STORE_ATTR are
three separate bytecodes), so concurrent updates can lose counts even
when only one `MCTS` instance is involved — manifesting as
non-deterministic `Q`/`N` aggregates.

## What changed

- New `MCTSConfig.virtual_loss: float = 0.0`. Backward-compatible default;
  opt-in.
- New per-`MCTS` `threading.RLock` (`self._select_lock`), acquired in
  `_select_leaf` for the entire descent-and-mark phase when the flag is
  enabled. `RLock` so the same thread can re-enter from backprop / release
  helpers.
- `_select_leaf` now returns a parallel `node_path` so release/backprop can
  walk the parent nodes and clear their `in_flight` markers.
- `_backpropagate`, `_release_virtual_visits`, `_clear_virtual_visits`
  acquire the lock when `virtual_loss > 0`, serializing per-tree mutation
  windows.
- `MCTSResult.diagnostics["virtual_loss_selections"]` reports how many
  descents actually entered the locked path (debugging aid, only nonzero
  when the flag is on).
- `SelfPlayConfig.virtual_loss` mirrors the MCTS flag and is plumbed into
  the per-game `MCTSConfig` in `_run_single_game_worker`.
- New CLI flag `--virtual-loss` in `add_selfplay_runtime_args`, default
  `0.0`. Wired into both the `self-play` and `loop` subparsers.
- New test `TestVirtualLossConcurrency` in `test_mcts.py`:
  (a) Spawns `NUM_THREADS=8` threads each owning its OWN `MCTS` instance
      (i.e., its own per-instance `_select_lock`), all calling
      `mcts.search(..., root=shared_root)` against the same `SearchNode`.
      The test therefore verifies the **per-instance lock's** intra-instance
      guarantees inside `_select_leaf` / `_backpropagate` /
      `_release_virtual_visits` — NOT a guarantee against races between
      distinct `MCTS` instances sharing a tree. (A genuinely-shared-tree
      test would pass a single `MCTS` instance to multiple worker threads.)
      Asserts (i) `max_edge_virtual_visits <= 2 * NUM_THREADS + 1` observed
      inside `_ProbingModel.inference`'s snapshot read (the lock takes the
      reading thread's own `_select_lock`, so the bound is the burst-
      scheduling slack for batched evaluation, not a per-edge hard cap),
      and (ii) no leaked `in_flight` / `virtual_visits` markers after all
      searches completed.
  (b) Asserts single-threaded `virtual_loss=0.0` and `virtual_loss=1.0`
      produce bit-equal `(visits, value_sum, virtual_visits)` per edge —
      the lock is uncontended in single-threaded execution so the only
      difference between modes is the lock acquisition itself.

## Why

- **§4.1 collapse**: without virtual loss, two threads picking the same
  internal node on different edges during the GPU forward release window
  both see value estimates that converge on average to 0, and the value
  head saturates at 0. This is the failure mode already commented in
  `train.py` (lines 100-103, 1116-1120, 2832-2835).
- **Lc0 vs ELF**: Lc0's "unscored virtual visits" (increment `N`, leave
  `Q` unchanged) is preferred over ELF OpenGo's `virtual_loss = 1.0`
  (decrement `Q` by 1) because the existing accounting already matches
  Lc0 semantics and ELF's `Q` decrement requires touching
  `edge.value_sum` on every concurrent visit, doubling the write
  amplification.
- **Opt-in default**: keeping `virtual_loss=0.0` as the default preserves
  the existing self-play byte-for-byte until the fix is empirically
  validated in production. The CPU path (`use_processes=True`, default on
  CPU) doesn't share trees across threads, so the flag has no effect on
  the current CPU pipeline; it future-proofs the GPU
  thread+`BatchInferenceServer` path.

## Knobs

| `virtual_loss` | Behavior |
| --- | --- |
| `0.0` (default) | Legacy path: no lock, no per-node `in_flight` increments, intra-search collision detection only. |
| `> 0.0` (e.g. `1.0`) | Per-MCTS `RLock` acquired in `_select_leaf`, `_backpropagate`, `_release_virtual_visits`, `_clear_virtual_visits`. When `virtual_loss > 0`, the `in_flight` marker persists on a node across multiple descent attempts within a single `search()` call. It is cleared in bulk at end-of-search by `_clear_virtual_visits(root)`. Within a search, concurrent descenters observe `in_flight == 1` and trigger the collision branch. |

## Interactions

- **`BatchInferenceServer`** (`train.py` line 276): unchanged. The server
  already serializes inference calls via its own `Condition`; the MCTS
  lock serializes per-tree mutations. The two locks are independent and
  non-overlapping.
- **GIL**: under CPython, the lock is technically only needed for
  read-modify-write sequences that span multiple bytecodes
  (`virtual_visits += 1`, `in_flight = 1`). The lock is cheap and
  uncontended in single-threaded execution, so the cost is negligible.
- **`num_simulations`, `batch_size`**: unchanged. The lock is released
  before `_evaluate_and_expand` enqueues to the inference server, so
  batched forward calls still overlap across threads.

## Files touched

- `mcts.py` — `MCTSConfig.virtual_loss`, `RLock`, descend-and-mark
  locking, parallel `node_path` plumbing, diagnostics count.
- `train.py` — `SelfPlayConfig.virtual_loss`, two pass-through
  assignments to `MCTSConfig`, `--virtual-loss` CLI flag.
- `test_mcts.py` — new `TestVirtualLossConcurrency` class + `_ProbingModel`
  fixture + `_count_non_zero_markers` helper.
- This file.

## Out of scope

- Sharing a single `SearchNode` tree across multiple `MCTS` instances
  simultaneously. `_select_lock` is per-instance (allocated in
  `MCTS.__init__` at line 248), so distinct instances competing for the
  same tree's edges can race against each other without serialization.
  The concurrency test (`TestVirtualLossConcurrency` (a)) verifies the
  per-instance lock's intra-instance guarantees; it is NOT a proof of
  shared-tree safety across instances. To use a single `MCTS` across
  multiple worker threads safely today, ensure each thread owns its own
  subtree under the shared root (per-edge disjoint descent paths) until
  a tree-level lock is added.
- Adaptive per-batch virtual-loss scheduling (Lc0's "virtual loss per
  active search"). Currently a constant flag; future work could read
  the active-search count from `BatchInferenceServer` and scale the
  per-edge decrement accordingly.