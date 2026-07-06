# Log of training history for improvements

### 12-06-2026
 - Model name 
 - 1000 games
post_training_vs_random: games=80 wins=66 losses=13 draws=1 win_rate=0.825 red=0.750 blue=0.900 avg_steps=38.6

### 13-06-2026
 - 1000 games
 - Modle name test.pt
post_training_vs_random: games=80 wins=44 losses=30 draws=6 win_rate=0.550 red=0.525 blue=0.575 avg_steps=66.7

### 14-06-2026
 - 1000 games
 - 2000 cumulative games
 - continued from 13-06 checkpoint
 - Model name 7x7_mcts_2000it.pt
post_training_vs_random: games=80 wins=77 losses=2 draws=1 win_rate=0.963 red=1.000 blue=0.925 avg_steps=28.5

### 15-06-2026
 - 2000 games
 - 4000 cumulative games
 - continued from 14-06 checkpoint
 - Model name best_1506.pt
post_training_vs_random: games=80 wins=77 losses=2 draws=1 win_rate=0.963 red=1.000 blue=0.925 avg_steps=28.5
 * Benchmark against previous checkpoint
```
device=cpu games=256 simulations=128 batch_size=16 walls=5 max_steps=96
model_a=best_1506 path=checkpoint_copies/best_1506.pt hidden=64 residual_blocks=4
model_b=7x7_mcts_2000it path=checkpoint_copies/7x7_mcts_2000it.pt hidden=64 residual_blocks=4
score best_1506=256 7x7_mcts_2000it=0 draws=0 best_1506_win_rate=1.000
best_1506_as_red=128/128 best_1506_as_blue=128/128 first_player_wins=128/256
avg_steps=27.5 elapsed=1674.474s games_per_second=0.15
```

 * Benchmark against self
```
benchmark_models
device=cpu games=256 simulations=128 batch_size=16 walls=5 max_steps=96
model_a=best_1506 path=checkpoint_copies/best_1506.pt hidden=64 residual_blocks=4
model_b=best_1506 path=checkpoint_copies/best_1506.pt hidden=64 residual_blocks=4
score best_1506=130 best_1506=126 draws=0 best_1506_win_rate=0.508
best_1506_as_red=65/128 best_1506_as_blue=65/128 first_player_wins=158/256
avg_steps=38.3 elapsed=2801.567s games_per_second=0.09
```

### 16-06-2026
 - 1000 games
 - 5000 cumulative games
 - continued from 15-06 checkpoint

 * Benchmark against previous checkpoint
```
device=cpu games=256 simulations=128 batch_size=16 walls=5 max_steps=96
model_a=model_1606 path=checkpoint_copies/model_1606.pt hidden=64 residual_blocks=4
model_b=best_1506 path=checkpoint_copies/best_1506.pt hidden=64 residual_blocks=4
score model_1606=182 best_1506=74 draws=0 model_1606_win_rate=0.711
model_1606_as_red=90/128 model_1606_as_blue=92/128 first_player_wins=180/256
avg_steps=41.0 elapsed=2432.830s games_per_second=0.11
```

### 24-06-2026
 - Official 9x9 training

### 25-06-2026
 - 1400 accumulated games
 - Tabula rasa

### 25-06-2026 7:44
 - 2000 accumulated games

### 25-05-2026 19:17
 - 5000 accumulated games

### 26-06-2026
 - model_iter_000010.pt
 - base line elo

### 27-06-2026
 - model_iter_000033.pt
 - +52 elo (256 simulations per move)

### 28-06-2026
 - latest_2806h1203.pt
 - +259 elo (256 sims per move)| 4 | 10 | 1 | 3 | 2 | 5 | 5 | 4 | 4 | 0.5000 | -0.2125 | 35.4000 |
| 5 | 10 | 1 | 2 | 0 | 5 | 5 | 3 | 7 | 0.3000 | -0.1875 | 40.3000 |
