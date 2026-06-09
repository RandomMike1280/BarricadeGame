# BarricadeGame

A Pygame Barricade/Quoridor-style game with a minimax AI opponent, plus a
separate headless reinforcement-learning environment.

## Run the Pygame game

```powershell
python barricade_pygame.py
```

## Use the RL environment

The RL environment is in `barricade_env.py` and does not import Pygame.
It follows the Gymnasium `reset`/`step` API. Gymnasium and NumPy are optional:
if installed, the environment exposes Gymnasium spaces and NumPy observations;
otherwise it falls back to plain Python lists.

```python
from barricade_env import BarricadeEnv

env = BarricadeEnv()
obs, info = env.reset(seed=1)

action = env.sample_legal_action()
obs, reward, terminated, truncated, info = env.step(action)

print(reward, terminated, truncated)
print(info["current_player"], info["winner"])
```

You can change the initial pawn positions and wall counts when creating the
environment:

```python
env = BarricadeEnv(
    red_start=(0, 2),
    blue_start=(8, 6),
    red_walls=5,
    blue_walls=5,
)
```

You can also override them per episode through `reset(options=...)`:

```python
obs, info = env.reset(
    options={
        "red_start": (2, 4),
        "blue_start": (6, 4),
        "red_walls": 3,
        "blue_walls": 7,
    }
)
```

## Action space

The environment uses a fixed discrete action space with 132 actions:

- `0`: move pawn one square up
- `1`: move pawn one square down
- `2`: move pawn one square left
- `3`: move pawn one square right
- `4..67`: place a horizontal wall, encoded as `4 + row * 8 + col`
- `68..131`: place a vertical wall, encoded as `68 + row * 8 + col`

Use `env.legal_action_mask()` or `info["action_mask"]` to mask illegal moves.
Use `env.decode_action(action)` and `env.encode_move(move)` to convert between
action ids and rule-engine move tuples.

## Observation

`obs` is a dictionary:

- `board`: 4 planes shaped `(4, 9, 9)`
  - plane 0: Red pawn
  - plane 1: Blue pawn
  - plane 2: horizontal wall anchors in the top-left 8x8 area
  - plane 3: vertical wall anchors in the top-left 8x8 area
- `features`: `[current_player, red_walls_left, blue_walls_left, red_greedy_path, blue_greedy_path]`
- `action_mask`: length-132 legal-action mask

Rewards are from the perspective of the player who just acted. By default,
legal non-terminal moves receive `0`, winning moves receive `1`, and invalid
actions terminate the episode with `-1`.
