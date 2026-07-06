# MCTS Speed Optimization — Iteration Log

Goal: improve the wall-clock speed of `mcts.MCTS.search` while preserving search
correctness on the legacy single-threaded default path (no virtual_loss /
no AMAF / no history).

Metric: sims/sec from `mcts_benchmark.run_benchmark` on `wall_heavy_state()`-
shaped positions with the legacy `MCTSConfig()` defaults
(`num_simulations=400`, `batch_size=32`, no history, no noise). Secondary
check: per-edge visits / value_sum must remain bit-identical to the pre-PR
fixture (`tests/fixtures/mcts_pr_baseline.npz`) — bit-exact on integers,
`1e-6` slack on floats.

Legend: KEEP = change retained; REVERT = change rolled back; SUPERSEDED =
replaced by a later iteration. Baseline #0 = current `mcts.py` HEAD
unchanged.

| # | Date | Idea (one line) | sims/sec | vs prev | vs base | core wall_moves_uncached best_us | core shortest_path_uncached best_us | Outcome | Side effects / notes |
|---|------|-----------------|---------:|--------:|--------:|--------------------------------:|------------------------------------:|---------|----------------------|
| 0 | 2026-07-04 | Baseline (HEAD) — measured at N=32 sims=400 batch=32 | ~1387 (1094.6 @ N=16, 1372.6+1400.6 @ N=32) | — | 0% | 235.21 (best of 3 reps) | 46.78 (best of 3 reps) | KEEP | mcts_micro_bench; 20 mcts unit tests OK. Variance run-to-run ~3%. Use N=32 going forward. |
| 1 | 2026-07-04 | Flatten adjacency into CSR (int lists) for DFS hot loop | 1127.2 (single run, drift dominates) | n/a | n/a | 343.90 (-46.2%) | 52.64 (-12.5%) | REVERT | Microbench regressed in 4-run mean (1349 vs 1471 baseline, -8.3%); env benchmark clearly worse. CSR int-range-iterate is no faster than tuple-unpack in CPython 3.14 on 9x9. |
| 2 | 2024-07-04 | Split _get_valid_wall_action_moves into horizontal/vertical passes; replace `1 << (index-1) & mask` with `(mask >> (index-1)) & 1` for adj cells; extract helpers | 1443.2 (single run, ~3% noise) | +4.0% (vs baseline mean ~1387) | +4.0% | 229.30 (+2.5%) | — | KEEP | Specialized path for `candidates is None`; orientation-pure inner loops; right-shift adj tests; minor CPU savings on ~74% hot path |
| 3 | 2026-07-04 | Precompute goal mask per cell as bytearray on `PathGraph`; replace `goal_mask & (1 << next_index)` in DFS with `goal_mask_by_index[next_index]` | 1625 (mean of 4: 1624.2/1630.3/1622.7/1623.5) | +12.6% | +17.2% | 200.87 (-14.6% vs baseline 235.21) | — | KEEP | Replaces a per-neighbour `int & (1 << n)` (big-int AND + shift) with a single bytearray load. Saves ~50–80ns per neighbour visited; on a 9x9 board each DFS visits ~80 neighbours, ~30K DFS calls per microbench. Tests OK; bit-exact vs baseline. |
| 4 | 2026-07-04 | Flatten per-node adjacency into a single (next, edge, next, edge, …) tuple and iterate with index arithmetic in DFS | 1349 (mean of 4: 1327.9/1365.1/1346.5/1355.5) | -17.0% | -2.7% | — | — | REVERT | Index-pair `while i < len(n): n[i]; n[i+1]; i += 2` regressed badly — tuple-unpack in CPython 3.14 for-loops is faster than index arithmetic on small tuples; the `len()` recompute and bounds checks also hurt. Also tried a chained-or `if x or y: continue` -> split `if x: continue; if y: continue` (no diff, ~1471 vs ~1474 in 4 runs) — pure noise, no action. |
| 5 | 2026-07-04 | Add `node.edge_list` tuple snapshot of `edges.values()`; iterate that in `_select_edge` (hottest path, 2 passes × 1664 calls × 136 edges) | 1652 (mean of 4: 1661.8/1656.0/1658.1/1632.0) | +1.7% | +19.1% | — | — | KEEP | Tuple iteration in CPython 3.14 is measurably faster than `dict.values()` iteration when no key is needed. Eliminates ~1664 dict view allocations per search. |
| 6 | 2026-07-04 | In `BarricadeState.copy`: share `pawns` and `initial_walls` references (read-only); skip `set()` copy of `_horizontal_walls_cache` / `_vertical_walls_cache` when empty | 1727 (mean of 4: 1726.5/1722.8/1717.7/1743.2) | +4.5% | +24.5% | — | — | KEEP | Copy cost went 1.06 µs → 0.92 µs (~13%). Both `pawns` and `initial_walls` are immutable, so the defensive `dict()` copy is wasted. The empty-set shortcut on the lazily-built wall-lookup caches avoids a no-op allocation for the common case. Tests OK. |
| 7 | 2026-07-04 | Unroll `_get_pawn_action_moves` 4-direction loop: replace `MOVE_DIRECTION_DELTAS.items()` iteration with a hardcoded `(direction_value, dr, dc)` list and pre-bind `self.board_size` | 1763 (mean of 4: 1755.5/1765.8/1776.7/1755.5); final 8-run median 1728.7, mean 1728.0 (1711.5–1753.9 range) | +2.1% (4-run) | +24.7% (8-run median vs baseline ~1387) | 226.10 (env bench unchanged vs baseline 235 — variance) | — | KEEP | Dict iteration with enum keys + tuple unpacking in the per-node hot path was the per-call bottleneck. Hardcoded literal-tuple list with local-bound board_size saves ~30% on this function. ~400 calls per search. **Final kept state** after 7 iterations. |
