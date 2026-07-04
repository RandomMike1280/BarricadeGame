"""
Barricade Play — Web UI for playing against the AlphaZero-trained model.

Run:
    python play.py [--port PORT] [--checkpoint CHECKPOINT]

Then open http://localhost:5000 in your browser.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from flask import Flask, jsonify, request, send_from_directory

from barricade_env import (
    ACTION_SIZE,
    BOARD_SIZE,
    BarricadeState,
    Move,
    Player,
    WallOrientation,
    encode_move,
    decode_action,
)
from canonical import canonical_action
from mcts import MCTS, MCTSConfig, SearchNode
from network import (
    AlphaZeroNetwork,
    EncoderConfig,
    build_network,
    encode_state_stack,
    infer_policy_head_type_from_state_dict,
    symexp,
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Barricade Play Web UI")
parser.add_argument("--port", type=int, default=5000, help="Server port")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="checkpoints/latest.pt",
    help="Path to model checkpoint (.pt file)",
)
parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address")
parser.add_argument("--device", type=str, default=None, help="Device (auto if omitted)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Device & model loading
# ---------------------------------------------------------------------------

device = torch.device(
    args.device or ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"[play] using device: {device}")

checkpoint_path = Path(args.checkpoint)
if not checkpoint_path.exists():
    print(f"[play] ERROR: checkpoint not found: {checkpoint_path}", file=sys.stderr)
    sys.exit(1)

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
state_dict = checkpoint.get("model_state", checkpoint)
raw_config = dict(checkpoint.get("network_config", {}))
if "policy_head_type" not in raw_config:
    raw_config["policy_head_type"] = infer_policy_head_type_from_state_dict(state_dict)

history_length = int(raw_config.get("history_length", 0))
conv_channels = int(raw_config.get("conv_channels", 128))
residual_channels = raw_config.get("residual_channels", None)
num_conv_layers = int(raw_config.get("num_conv_layers", 1))
num_residual_layers = int(raw_config.get("num_residual_layers", 10))
value_hidden_size = int(raw_config.get("value_hidden_size", 256))
policy_head_type = str(raw_config.get("policy_head_type", "factored"))

model = build_network(
    history_length=history_length,
    conv_channels=conv_channels,
    residual_channels=residual_channels,
    num_conv_layers=num_conv_layers,
    num_residual_layers=num_residual_layers,
    value_hidden_size=value_hidden_size,
    policy_head_type=policy_head_type,
)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()
print(f"[play] model loaded from {checkpoint_path}")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder=None)

# ---------------------------------------------------------------------------
# Game state (per-connection — single-player for now)
# ---------------------------------------------------------------------------

game_state = None  # type: Optional[BarricadeState]
mcts = None        # type: Optional[MCTS]
mcts_root = None   # type: Optional[SearchNode]
game_history = []  # type: List[BarricadeState]
# Pawn positions immediately before the last applied action (None until
# at least one move has been played in the current game). Used by the
# frontend to draw the last-move arrow.
previous_pawns = None  # type: Optional[Dict[str, List[int]]]

# Default MCTS config — can be overridden via /settings
current_mcts_config = MCTSConfig(
    num_simulations=400,
    batch_size=16,
    cpuct_init=1.75,
    cpuct_base=38739.0,
    cpuct_factor=3.89,
    fpu_reduction=0.33,
    pawn_prior_floor=0.0,
    policy_temperature=1.0,
    root_dirichlet_alpha=0.3,
    root_exploration_fraction=0.25,
    action_temperature=1.0,
    policy_target_temperature=None,
    policy_target_floor=0.0,
    history_length=history_length,
    device=str(device),
    add_root_noise=True,
    lead_weight=0.01,
    lead_scale=10.0,
)


def _build_mcts() -> MCTS:
    """Create a fresh MCTS instance with the current config."""
    return MCTS(
        model,
        current_mcts_config,
        policy_action_transform=canonical_action,
    )


def _serialize_state(state: BarricadeState) -> Dict[str, Any]:
    """Serialize game state for the frontend."""
    red_pl = state.shortest_path_length(Player.RED)
    blue_pl = state.shortest_path_length(Player.BLUE)
    return {
        "board_size": state.board_size,
        "pawns": {
            "RED": list(state.pawns[Player.RED]),
            "BLUE": list(state.pawns[Player.BLUE]),
        },
        "walls": [
            {
                "orientation": w[0].name if isinstance(w[0], WallOrientation) else str(w[0]),
                "row": int(w[1]),
                "col": int(w[2]),
            }
            for w in state.walls
        ],
        "current_player": state.current_player.name,
        "winner": state.winner.name if state.winner is not None else None,
        "is_draw": state.is_draw,
        "draw_reason": state.draw_reason,
        "walls_left": {
            "RED": state.walls_left[Player.RED],
            "BLUE": state.walls_left[Player.BLUE],
        },
        "is_terminal": state.is_terminal(),
        "previous_pawns": previous_pawns,
        "red_path_length": red_pl,
        "blue_path_length": blue_pl,
        "path_gap": (
            (blue_pl - red_pl) if (red_pl is not None and blue_pl is not None) else None
        ),
    }


def _serialize_valid_moves(state: BarricadeState) -> List[Dict[str, Any]]:
    """Serialize valid moves for the frontend."""
    moves = []
    for action, move in state.legal_action_moves():
        move_type = move[0]
        if move_type == "move_to":
            moves.append({
                "type": "move",
                "action": int(action),
                "row": int(move[1]),
                "col": int(move[2]),
            })
        elif move_type == "wall":
            orient = move[1]
            orient_name = orient.name if isinstance(orient, WallOrientation) else str(orient)
            moves.append({
                "type": "wall",
                "action": int(action),
                "orientation": orient_name,
                "row": int(move[2]),
                "col": int(move[3]),
            })
    return moves


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return send_from_directory(".", "play.html")


@app.route("/api/new_game", methods=["POST"])
def new_game():
    global game_state, mcts, mcts_root, game_history, previous_pawns
    data = request.get_json(silent=True) or {}
    red_start = tuple(data.get("red_start", (0, 4)))
    blue_start = tuple(data.get("blue_start", (8, 4)))
    red_walls = int(data.get("red_walls", 10))
    blue_walls = int(data.get("blue_walls", 10))
    # player_side: which side the human wants to play ("RED" or "BLUE")
    player_side = data.get("player_side", "RED")
    # starting_player: who moves first. If human plays RED they move first;
    # if human plays BLUE the AI (RED) moves first.
    starting_player = "RED" if player_side == "BLUE" else "RED"

    game_state = BarricadeState(
        red_start=red_start,
        blue_start=blue_start,
        red_walls=red_walls,
        blue_walls=blue_walls,
        starting_player=starting_player,
    )
    mcts = _build_mcts()
    mcts_root = None
    game_history = []
    previous_pawns = None

    # If the AI (RED) is to move first, run MCTS now and return the AI's move
    ai_first_move = None
    if starting_player == "RED" and player_side == "BLUE":
        result = mcts.search(game_state)
        ai_action = mcts.select_action(result, temperature=current_mcts_config.action_temperature)
        pre_ai_pawns = {
            Player.RED.name: list(game_state.pawns[Player.RED]),
            Player.BLUE.name: list(game_state.pawns[Player.BLUE]),
        }
        game_history.append(game_state.copy())
        game_state = game_state.apply_action(ai_action, validate=False)
        mcts_root = mcts.advance_root(result.root, ai_action)
        previous_pawns = pre_ai_pawns
        ai_first_move = {
            "action": int(ai_action),
            "value": round(float(result.root_value), 4),
            "lead": round(float(result.root_lead), 4),
        }

    return jsonify({
        "state": _serialize_state(game_state),
        "valid_moves": _serialize_valid_moves(game_state),
        "ai_first_move": ai_first_move,
    })


@app.route("/api/state", methods=["GET"])
def get_state():
    global game_state
    if game_state is None:
        return jsonify({"error": "No game in progress"}), 400
    return jsonify({
        "state": _serialize_state(game_state),
        "valid_moves": _serialize_valid_moves(game_state),
    })


@app.route("/api/move", methods=["POST"])
def make_move():
    global game_state, mcts, mcts_root, game_history, previous_pawns
    if game_state is None:
        return jsonify({"error": "No game in progress"}), 400
    if game_state.is_terminal():
        return jsonify({"error": "Game is over"}), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400

    action = int(data.get("action", -1))
    legal_actions = set(game_state.legal_actions())
    if action not in legal_actions:
        return jsonify({"error": f"Illegal action {action}"}), 400

    # Apply human move
    pre_human_pawns = {
        Player.RED.name: list(game_state.pawns[Player.RED]),
        Player.BLUE.name: list(game_state.pawns[Player.BLUE]),
    }
    game_history.append(game_state.copy())
    game_state = game_state.apply_action(action, validate=False)
    previous_pawns = pre_human_pawns

    if game_state.is_terminal():
        mcts_root = None
        return jsonify({
            "state": _serialize_state(game_state),
            "valid_moves": [],
            "ai_move": None,
            "game_over": True,
        })

    # Run MCTS for AI response
    if mcts is None:
        mcts = _build_mcts()

    # Advance the MCTS root by the human's action so the search tree is reused
    if mcts_root is not None:
        mcts_root = mcts.advance_root(mcts_root, action)

    search_t0 = time.perf_counter()
    result = mcts.search(
        game_state,
        history=game_history[-current_mcts_config.history_length:] if current_mcts_config.history_length > 0 else None,
        root=mcts_root,
    )
    search_dt_ms = (time.perf_counter() - search_t0) * 1000.0
    ai_action = mcts.select_action(result, temperature=current_mcts_config.action_temperature)

    # Apply AI move
    pre_ai_pawns = {
        Player.RED.name: list(game_state.pawns[Player.RED]),
        Player.BLUE.name: list(game_state.pawns[Player.BLUE]),
    }
    game_history.append(game_state.copy())
    game_state = game_state.apply_action(ai_action, validate=False)
    previous_pawns = pre_ai_pawns

    # Advance MCTS root for the AI move (root is now at the state after AI's move)
    mcts_root = mcts.advance_root(result.root, ai_action)

    # Build stats for the frontend
    stats = []
    for s in result.stats:
        stats.append({
            "action": s.action,
            "prior": round(s.prior, 4),
            "visits": s.visits,
            "q": round(s.q, 4),
            "lead_q": round(s.lead_q, 4),
        })

    # Flip root value/lead to BLUE perspective: +1 = BLUE wins, -1 = RED wins
    current_player = game_state.current_player
    blue_perspective_value = float(result.root_value)
    blue_perspective_lead = float(result.root_lead)
    if current_player == Player.RED:
        blue_perspective_value = -blue_perspective_value
        blue_perspective_lead = -blue_perspective_lead

    completed_sims = int(result.diagnostics.get("completed_simulations", 0))
    sims_per_sec = (
        completed_sims / (search_dt_ms / 1000.0)
        if search_dt_ms > 0 else 0.0
    )

    return jsonify({
        "state": _serialize_state(game_state),
        "valid_moves": _serialize_valid_moves(game_state),
        "ai_move": {
            "action": int(ai_action),
            "value": round(float(result.root_value), 4),
            "lead": round(float(result.root_lead), 4),
        },
        "blue_value": round(blue_perspective_value, 4),
        "blue_lead": round(blue_perspective_lead, 4),
        "stats": stats,
        "diagnostics": {
            "simulations": completed_sims,
            "neural_batches": int(result.diagnostics.get("neural_batches", 0)),
            "collisions": int(result.diagnostics.get("collisions", 0)),
            "evaluated_leaves": int(result.diagnostics.get("evaluated_leaves", 0)),
            "collision_flushes": int(result.diagnostics.get("collision_flushes", 0)),
            "search_time_ms": round(search_dt_ms, 2),
            "sims_per_sec": round(sims_per_sec, 1),
        },
        "game_over": game_state.is_terminal(),
    })


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """Run a full forward pass on the current state and return all head outputs."""
    global game_state, game_history
    if game_state is None:
        return jsonify({"error": "No game in progress"}), 400

    data = request.get_json(silent=True) or {}
    # Optionally evaluate a specific action's resulting state
    action = data.get("action")
    if action is not None:
        action = int(action)
        if action not in set(game_state.legal_actions()):
            return jsonify({"error": f"Illegal action {action}"}), 400
        eval_state = game_state.apply_action(action, validate=False)
        eval_history = game_history + [game_state]
    else:
        eval_state = game_state
        eval_history = game_history

    # Encode state
    state_tensor = encode_state_stack(
        eval_state,
        eval_history[-history_length:] if history_length > 0 else None,
        history_length=history_length,
    ).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits, value, lead, future_logits, score = model(state_tensor)

    # Value: tanh output in [-1, 1] from current-player perspective
    raw_value = float(value.view(-1).cpu().item())
    # Lead: symexp to undo the symlog applied during training
    raw_lead = float(symexp(lead).view(-1).cpu().item())
    # Score: softplus, positive
    raw_score = float(score.view(-1).cpu().item())

    # Flip to BLUE perspective: +1 = BLUE wins, -1 = RED wins
    current = eval_state.current_player
    if current == Player.RED:
        blue_value = -raw_value
        blue_lead = -raw_lead
    else:
        blue_value = raw_value
        blue_lead = raw_lead

    # Future map: 2 x 9 x 9 logits -> softmax per cell over the 2 channels
    future_logits_np = future_logits.squeeze(0).cpu().numpy()  # (2, 9, 9)
    # Softmax over channel dim for each cell
    import numpy as np
    future_probs = np.exp(future_logits_np) / np.sum(np.exp(future_logits_np), axis=0, keepdims=True)
    future_map = {
        "side_to_move": future_probs[0].tolist(),  # 9x9
        "opponent": future_probs[1].tolist(),       # 9x9
    }

    # Top-8 policy actions
    legal_actions = eval_state.legal_actions()
    action_mask = torch.zeros(ACTION_SIZE, dtype=torch.bool)
    action_mask[legal_actions] = True
    masked_logits = logits.clone()
    masked_logits[0, ~action_mask] = -1e9
    probs = torch.softmax(masked_logits, dim=1).squeeze(0)
    top_probs, top_indices = torch.topk(probs, min(8, len(legal_actions)))
    top_policy = [
        {"action": int(idx), "prob": round(float(p), 4)}
        for idx, p in zip(top_indices.cpu().tolist(), top_probs.cpu().tolist())
    ]

    return jsonify({
        "value": round(raw_value, 4),
        "lead": round(raw_lead, 4),
        "score": round(raw_score, 4),
        "blue_value": round(blue_value, 4),
        "blue_lead": round(blue_lead, 4),
        "future_map": future_map,
        "top_policy": top_policy,
        "current_player": current.name,
    })


@app.route("/api/settings", methods=["GET", "POST"])
def settings():
    global current_mcts_config, mcts
    if request.method == "GET":
        return jsonify({
            "num_simulations": current_mcts_config.num_simulations,
            "batch_size": current_mcts_config.batch_size,
            "cpuct_init": current_mcts_config.cpuct_init,
            "cpuct_base": current_mcts_config.cpuct_base,
            "cpuct_factor": current_mcts_config.cpuct_factor,
            "fpu_reduction": current_mcts_config.fpu_reduction,
            "pawn_prior_floor": current_mcts_config.pawn_prior_floor,
            "policy_temperature": current_mcts_config.policy_temperature,
            "root_dirichlet_alpha": current_mcts_config.root_dirichlet_alpha,
            "root_exploration_fraction": current_mcts_config.root_exploration_fraction,
            "action_temperature": current_mcts_config.action_temperature,
            "lead_weight": current_mcts_config.lead_weight,
            "lead_scale": current_mcts_config.lead_scale,
            "add_root_noise": current_mcts_config.add_root_noise,
        })

    data = request.get_json(silent=True) or {}
    cfg = current_mcts_config
    current_mcts_config = MCTSConfig(
        num_simulations=int(data.get("num_simulations", cfg.num_simulations)),
        batch_size=int(data.get("batch_size", cfg.batch_size)),
        cpuct_init=float(data.get("cpuct_init", cfg.cpuct_init)),
        cpuct_base=float(data.get("cpuct_base", cfg.cpuct_base)),
        cpuct_factor=float(data.get("cpuct_factor", cfg.cpuct_factor)),
        fpu_reduction=float(data.get("fpu_reduction", cfg.fpu_reduction)),
        pawn_prior_floor=float(data.get("pawn_prior_floor", cfg.pawn_prior_floor)),
        policy_temperature=float(data.get("policy_temperature", cfg.policy_temperature)),
        root_dirichlet_alpha=float(data.get("root_dirichlet_alpha", cfg.root_dirichlet_alpha)),
        root_exploration_fraction=float(data.get("root_exploration_fraction", cfg.root_exploration_fraction)),
        action_temperature=float(data.get("action_temperature", cfg.action_temperature)),
        lead_weight=float(data.get("lead_weight", cfg.lead_weight)),
        lead_scale=float(data.get("lead_scale", cfg.lead_scale)),
        add_root_noise=bool(data.get("add_root_noise", cfg.add_root_noise)),
        history_length=cfg.history_length,
        device=str(device),
        policy_target_temperature=cfg.policy_target_temperature,
        policy_target_floor=cfg.policy_target_floor,
    )
    mcts = _build_mcts()
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Initialize a default game at module level
    _game_state = BarricadeState()
    _mcts = _build_mcts()
    _mcts_root = None
    _game_history: list[BarricadeState] = []
    
    # Assign to the module-level globals (already global — no `global` keyword needed)
    game_state = _game_state
    mcts = _mcts
    mcts_root = _mcts_root
    game_history = _game_history
    print(f"[play] server starting on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)