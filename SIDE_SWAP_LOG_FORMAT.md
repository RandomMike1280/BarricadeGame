# Arena Side-Swap + Per-Side Win-Rate Logging

This document records the schema change introduced by the arena side-swap task. It applies to two artifacts produced by `train.py`:

1. **`train_log_history.md`** — a new markdown table, one row per arena batch.
2. **`train_log_<DD_MM>.txt`** — the existing tee'd stdout file; per-side fields are appended to the per-game and per-batch summary lines.

Both files now expose the same per-side counters; the markdown file is the canonical structured consumer.

## Motivation

Asymmetric boards can let the value head quietly saturate toward one side, which an aggregate `wins / losses / draws` counter will hide. Tracking wins separately for games where the new net played RED versus BLUE catches first-player bias early.

The arena also now enforces strict side swap: the new net plays as the side it starts as, alternating per game.

## Convention (pinned)

For every arena batch in `train.py::evaluate_models`:

- `game_index` even (0, 2, 4, ...) — the new net plays **RED** and **starts as RED**.
- `game_index` odd (1, 3, 5, ...) — the new net plays **BLUE** and **starts as BLUE**.

Because the new net always starts as the side it plays, the labels `wins_when_red_starts` and `wins_when_blue_starts` are equivalent to "wins as RED" and "wins as BLUE" respectively.

The invariant `candidate starts as the side it plays` is enforced by an explicit override `handicap['starting_player'] = candidate_player` in `train.py`; the even/odd alternation alone does not produce this invariant.

The first iteration is skipped (no baseline exists yet) and does **not** produce a row.

## Schema — `train_log_history.md`

One header row, one separator, then one row per arena batch.

| Column                       | Type    | Units | Meaning                                                                 |
|-----------------------------|---------|-------|-------------------------------------------------------------------------|
| `iteration`                  | int     | —     | 1-indexed training iteration number.                                    |
| `games`                      | int     | —     | Total arena games played this batch (config-driven).                    |
| `wins_when_red_starts`       | int     | —     | Candidate wins on even-indexed games (new net = RED, starts).           |
| `wins_when_blue_starts`      | int     | —     | Candidate wins on odd-indexed games (new net = BLUE, starts).           |
| `draws`                      | int     | —     | Games ending in draw.                                                   |
| `games_when_red_starts`      | int     | —     | Number of arena games where candidate was RED (≈ `games // 2`).         |
| `games_when_blue_starts`     | int     | —     | Number of arena games where candidate was BLUE (≈ `games // 2`).        |
| `wins`                       | int     | —     | **Derived** aggregate: `wins_when_red_starts + wins_when_blue_starts`.  |
| `losses`                     | int     | —     | Candidate losses (winner = opponent).                                   |
| `win_rate`                   | float   | 0..1  | `(wins + 0.5 * draws) / games`.                                         |
| `average_lead`               | float   | plies | Mean lead over all arena games.                                        |
| `average_game_length`        | float   | steps | Mean number of env steps per arena game.                                |

### Invariants

These should hold for every row:

- `games_when_red_starts + games_when_blue_starts == games`
- `wins_when_red_starts + wins_when_blue_starts + losses + draws == games`
- `wins_when_red_starts + wins_when_blue_starts == wins`
- `0 <= wins_when_red_starts <= games_when_red_starts`
- `0 <= wins_when_blue_starts <= games_when_blue_starts`

### Header

The header row is written exactly once on first append:

```
| iteration | games | wins_when_red_starts | wins_when_blue_starts | draws | games_when_red_starts | games_when_blue_starts | wins | losses | win_rate | average_lead | average_game_length |
|---|---|---|---|---|---|---|---|---|---|---|---|
```

Floats are formatted with four decimal places; integers are emitted as-is.

## Schema — `train_log_<DD_MM>.txt`

Per-game line (one per arena game) now ends with:

```
red_starts=<wins>/<games_when_red_starts> blue_starts=<wins>/<games_when_blue_starts>
```

Per-batch summary (the `[eval] DONE` line) ends with the same suffix.

## Implementation pointers

- `evaluate_models` — `train.py`, around line 1940 onward. The override
  `handicap["starting_player"] = candidate_player` is at the top of each
  per-game iteration. Per-side counters are incremented in the
  winner-classification branch immediately below.
- `_append_train_history` — `train.py`, around line 313. Called from
  `run_loop` immediately after `[loop] evaluation phase done`.
- `TRAIN_HISTORY_PATH` — `train.py`, around line 284. Defaults to
  `train_log_history.md` in the cwd.

## Out-of-scope

- Self-play starting-player logic is **not** modified. Self-play remains
  on whatever convention it uses today; parity between train and eval is
  preserved because both the arena side-swap and self-play starting-side
  are independent of the candidate-side rotation in the arena.
- `EvalConfig.games` (the arena count) and the 0.55 promotion threshold
  are unchanged.