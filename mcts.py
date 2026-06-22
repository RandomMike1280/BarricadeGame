"""
Lc0-inspired batched MCTS for Barricade.

This module implements an AlphaZero-style search with PUCT, FPU, masked neural
priors, root Dirichlet noise, unscored virtual visits, and synchronous batched
neural-network evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from barricade_env import ACTION_SIZE, BarricadeState, Move, Player, encode_move
from network import encode_state_stack


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    num_simulations: int = 800
    batch_size: int = 32
    cpuct_init: float = 1.75
    cpuct_base: float = 38739.0
    cpuct_factor: float = 3.89
    fpu_reduction: float = 0.33
    policy_temperature: float = 1.0
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    action_temperature: float = 1.0
    policy_target_temperature: Optional[float] = None
    history_length: int = 0
    device: Optional[str] = None
    add_root_noise: bool = True
    lead_weight: float = 0.01
    lead_scale: float = 10.0


@dataclass(frozen=True, slots=True)
class SearchEvaluation:
    value: float
    lead: float


@dataclass(slots=True)
class SearchEdge:
    action: int
    move: Move
    prior: float
    child: Optional["SearchNode"] = None
    visits: int = 0
    virtual_visits: int = 0
    value_sum: float = 0.0
    lead_sum: float = 0.0

    @property
    def effective_visits(self) -> int:
        return self.visits + self.virtual_visits

    @property
    def q(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    @property
    def lead_q(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.lead_sum / self.visits


@dataclass(slots=True)
class SearchNode:
    state: BarricadeState
    history: Tuple[BarricadeState, ...] = ()
    edges: Dict[int, SearchEdge] = field(default_factory=dict)
    is_expanded: bool = False
    in_flight: int = 0
    value_estimate: Optional[float] = None
    lead_estimate: Optional[float] = None
    root_noise_applied: bool = False
    terminal_eval: Optional[SearchEvaluation] = None

    @property
    def effective_visits(self) -> int:
        return sum(edge.effective_visits for edge in self.edges.values())

    @property
    def real_visits(self) -> int:
        return sum(edge.visits for edge in self.edges.values())


@dataclass(frozen=True, slots=True)
class ActionStats:
    action: int
    move: Move
    prior: float
    visits: int
    virtual_visits: int
    q: float
    lead_q: float
    value_sum: float
    lead_sum: float
    policy: float


@dataclass(frozen=True, slots=True)
class MCTSResult:
    action: int
    policy_target: List[float]
    root_value: float
    root_lead: float
    root: SearchNode
    stats: List[ActionStats]
    diagnostics: Dict[str, int] = field(default_factory=dict)


class MCTS:
    """AlphaZero-style MCTS with batched neural-network evaluation.

    When ``inference_server`` is provided, all neural-network evaluations are
    delegated to the server, which batches requests across multiple parallel
    game workers for dramatically improved GPU utilization.
    """

    def __init__(
        self,
        model,
        config: Optional[MCTSConfig] = None,
        *,
        device: Optional[str | torch.device] = None,
        rng: Optional[random.Random] = None,
        state_encoder: Optional[
            Callable[[BarricadeState, Sequence[BarricadeState], int], Tensor]
        ] = None,
        action_size: Optional[int] = None,
        encode_move_fn: Optional[Callable[[Move], int]] = None,
        policy_action_transform: Optional[Callable[[int, BarricadeState], int]] = None,
        lead_transform: Optional[Callable[[float, BarricadeState], float]] = None,
        inference_server: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.config = config or MCTSConfig()
        self._inference_server = inference_server
        self.device = self._resolve_device(device or self.config.device)
        # Only move model to device when NOT using an external server,
        # since the server manages the model's device.
        if inference_server is None:
            self.model.to(self.device)
        self.rng = rng or random.Random()
        self.state_encoder = state_encoder or self._default_state_encoder
        self.action_size = int(action_size or ACTION_SIZE)
        self.encode_move_fn = encode_move_fn
        self.policy_action_transform = policy_action_transform
        self.lead_transform = lead_transform

        # Pre-compute frequently used config values
        cfg = self.config
        self._history_length = cfg.history_length
        self._use_history = cfg.history_length > 0
        self._lead_weight = cfg.lead_weight
        self._use_lead_bonus = cfg.lead_weight > 0
        self._lead_scale = max(float(cfg.lead_scale), 1e-6)
        self._inv_lead_scale = 1.0 / self._lead_scale
        self._fpu_reduction = cfg.fpu_reduction
        self._cpuct_init = cfg.cpuct_init
        self._cpuct_base = max(cfg.cpuct_base, 1e-6)
        self._cpuct_factor = cfg.cpuct_factor
        self._policy_temperature = cfg.policy_temperature
        self._add_root_noise = cfg.add_root_noise
        self._root_dirichlet_alpha = cfg.root_dirichlet_alpha
        self._root_exploration_fraction = cfg.root_exploration_fraction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        state: BarricadeState,
        history: Optional[Sequence[BarricadeState]] = None,
        root: Optional[SearchNode] = None,
    ) -> MCTSResult:
        root = self._prepare_root(state, history, root)

        if root.terminal_eval is None and not root.is_expanded:
            self._evaluate_and_expand([root])

        if (
            self._add_root_noise
            and root.is_expanded
            and not root.root_noise_applied
        ):
            self._add_root_dirichlet_noise(root)

        completed = 0
        collision_streak = 0
        collisions = 0
        neural_batches = 0
        evaluated_leaves = 0
        num_simulations = self.config.num_simulations
        batch_size = self.config.batch_size

        while completed < num_simulations:
            pending: List[Tuple[SearchNode, List[SearchEdge]]] = []
            terminal_paths: List[Tuple[List[SearchEdge], SearchEvaluation]] = []
            target_batch = min(max(1, batch_size), num_simulations - completed)
            collision_limit = max(16, target_batch * 4)

            while len(pending) + len(terminal_paths) < target_batch:
                kind, leaf, path, evaluation = self._select_leaf(root)

                if kind == "collision":
                    self._release_virtual_visits(path)
                    collision_streak += 1
                    collisions += 1
                    if collision_streak > collision_limit:
                        break
                    continue

                collision_streak = 0
                if kind == "terminal":
                    terminal_paths.append((path, evaluation))
                elif kind == "leaf":
                    pending.append((leaf, path))

                if completed + len(pending) + len(terminal_paths) >= num_simulations:
                    break

            if not pending and not terminal_paths:
                break

            for path, evaluation in terminal_paths:
                self._backpropagate(path, evaluation)
            completed += len(terminal_paths)

            if pending:
                neural_batches += 1
                evaluated_leaves += len(pending)
                leaves = [leaf for leaf, _ in pending]
                evaluations = self._evaluate_and_expand(leaves)
                for (leaf, path), evaluation in zip(pending, evaluations):
                    if leaf.in_flight > 0:
                        leaf.in_flight -= 1
                    self._backpropagate(path, evaluation)
                completed += len(pending)

        self._clear_virtual_visits(root)

        policy_target_temperature = (
            self.config.action_temperature
            if self.config.policy_target_temperature is None
            else self.config.policy_target_temperature
        )
        policy_target = self._policy_target(root, policy_target_temperature)
        action_policy = self._policy_target(root, self.config.action_temperature)
        action = self._sample_policy(action_policy)
        stats = self._action_stats(root, policy_target)
        return MCTSResult(
            action=action,
            policy_target=policy_target,
            root_value=self._root_value(root),
            root_lead=self._root_lead(root),
            root=root,
            stats=stats,
            diagnostics={
                "completed_simulations": completed,
                "neural_batches": neural_batches,
                "evaluated_leaves": evaluated_leaves,
                "collisions": collisions,
            },
        )

    def select_action(self, result: MCTSResult, temperature: Optional[float] = None) -> int:
        if temperature is None:
            return result.action
        policy = self._policy_from_stats(
            result.stats,
            temperature,
            len(result.policy_target),
        )
        return self._sample_policy(policy)

    def advance_root(self, root: SearchNode, action: int) -> Optional[SearchNode]:
        edge = root.edges.get(int(action))
        if edge is None:
            return None
        return edge.child

    # ------------------------------------------------------------------
    # Root preparation
    # ------------------------------------------------------------------

    def _prepare_root(
        self,
        state: BarricadeState,
        history: Optional[Sequence[BarricadeState]],
        root: Optional[SearchNode],
    ) -> SearchNode:
        trimmed_history = self._trim_history(tuple(history or ()))
        if root is not None and root.state.state_cache_key() == state.state_cache_key():
            root.history = trimmed_history
            return root
        new_root = SearchNode(state.copy(), trimmed_history)
        winner_eval = self._winner_evaluation(new_root)
        if winner_eval is not None:
            new_root.terminal_eval = winner_eval
        return new_root

    # ------------------------------------------------------------------
    # Selection (hot path)
    # ------------------------------------------------------------------

    def _select_leaf(
        self, root: SearchNode
    ) -> Tuple[str, Optional[SearchNode], List[SearchEdge], Optional[SearchEvaluation]]:
        node = root
        path: List[SearchEdge] = []
        use_history = self._use_history
        trim_history = self._trim_history

        while True:
            te = node.terminal_eval
            if te is not None:
                return "terminal", node, path, te

            if not node.is_expanded:
                if node.in_flight > 0:
                    return "collision", node, path, None
                node.in_flight = 1
                return "leaf", node, path, None

            edges = node.edges
            if not edges:
                eval_ = SearchEvaluation(
                    value=-1.0,
                    lead=self._state_lead(node.state),
                )
                node.terminal_eval = eval_
                return "terminal", node, path, eval_

            edge = self._select_edge(node, edges)
            edge.virtual_visits += 1
            path.append(edge)

            child = edge.child
            if child is None:
                new_state = node.state.apply_move(edge.move)
                if use_history:
                    history = trim_history((*node.history, node.state))
                else:
                    history = ()
                child = SearchNode(state=new_state, history=history)
                child_eval = self._winner_evaluation(child)
                if child_eval is not None:
                    child.terminal_eval = child_eval
                edge.child = child
            node = child

    def _select_edge(self, node: SearchNode, edges: Dict[int, SearchEdge]) -> SearchEdge:
        # Materialize the edge list once and reuse it for both passes.
        edge_list = list(edges.values())

        real_visits = 0
        parent_effective_visits = 0
        parent_value_sum = 0.0
        parent_lead_sum = 0.0
        explored_prior = 0.0

        for edge in edge_list:
            v = edge.visits
            ev = v + edge.virtual_visits
            real_visits += v
            parent_effective_visits += ev
            parent_value_sum += edge.value_sum
            parent_lead_sum += edge.lead_sum
            if ev > 0:
                explored_prior += edge.prior

        pev = parent_effective_visits if parent_effective_visits > 0 else 1
        cpuct = self._cpuct_fast(pev)

        if real_visits > 0:
            parent_q = parent_value_sum / real_visits
            parent_lead_q = parent_lead_sum / real_visits
        else:
            parent_q = 0.0
            parent_lead_q = 0.0

        fpu_q = parent_q - self._fpu_reduction * math.sqrt(
            explored_prior if explored_prior > 0.0 else 0.0
        )
        sqrt_parent = math.sqrt(pev)
        cpuct_sqrt_parent = cpuct * sqrt_parent

        use_lead = self._use_lead_bonus
        if use_lead:
            lead_weight = self._lead_weight
            inv_lead_scale = self._inv_lead_scale
            # Unvisited edges all share parent_lead_q, so their lead bonus is
            # constant within this call: compute it once.
            fpu_lead_bonus = lead_weight * math.tanh(parent_lead_q * inv_lead_scale)
            # Score contribution of the FPU branch (q + lead) is also constant.
            fpu_q_plus_lead = fpu_q + fpu_lead_bonus
        else:
            fpu_q_plus_lead = fpu_q

        best_score = -math.inf
        best_edge: Optional[SearchEdge] = None

        for edge in edge_list:
            v = edge.visits
            ev = v + edge.virtual_visits

            u = cpuct_sqrt_parent * edge.prior / (1.0 + ev)

            if v > 0:
                score = edge.value_sum / v + u
                if use_lead:
                    score += lead_weight * math.tanh(
                        (edge.lead_sum / v) * inv_lead_scale
                    )
            else:
                score = fpu_q_plus_lead + u

            if score > best_score:
                best_score = score
                best_edge = edge

        if best_edge is None:
            raise RuntimeError("Cannot select from a node with no edges.")
        return best_edge

    # ------------------------------------------------------------------
    # Evaluation & expansion (batched neural network)
    # ------------------------------------------------------------------

    def _evaluate_and_expand(self, nodes: Sequence[SearchNode]) -> List[SearchEvaluation]:
        if not nodes:
            return []

        n = len(nodes)
        evaluations: List[Optional[SearchEvaluation]] = [None] * n
        expandable: List[Tuple[int, SearchNode, List[Tuple[int, Move]]]] = []

        for index in range(n):
            node = nodes[index]
            legal_action_moves = self._legal_action_moves(node.state)
            if legal_action_moves:
                expandable.append((index, node, legal_action_moves))
            else:
                eval_ = SearchEvaluation(
                    value=-1.0,
                    lead=self._state_lead(node.state),
                )
                node.edges = {}
                node.value_estimate = eval_.value
                node.lead_estimate = eval_.lead
                node.is_expanded = True
                node.terminal_eval = eval_
                evaluations[index] = eval_

        if not expandable:
            return [e for e in evaluations if e is not None]

        # Build batch tensor on CPU — the server (or .to(device)) handles GPU transfer
        batch = torch.stack(
            [
                self.state_encoder(
                    node.state,
                    node.history,
                    self._history_length,
                )
                for _, node, _ in expandable
            ],
            dim=0,
        )

        if self._inference_server is not None:
            # Delegate to the batched inference server for cross-worker batching.
            # The server collects requests from all parallel game workers and
            # runs them as a single large batched forward pass.
            logits_list, values_list, leads_list = self._inference_server.infer(batch)
        else:
            batch = batch.to(self.device)
            was_training = self.model.training
            self.model.eval()

            with torch.inference_mode():
                logits_t, values_t, leads_t = self._model_inference(batch)
                logits_list = logits_t.tolist()
                values_list = values_t.view(-1).tolist()
                leads_list = leads_t.view(-1).tolist()

            if was_training:
                self.model.train()

        policy_temperature = self._policy_temperature

        for idx in range(len(expandable)):
            index, node, legal_action_moves = expandable[idx]
            value = values_list[idx]
            lead = self._model_lead(leads_list[idx], node.state)
            eval_ = SearchEvaluation(value=value, lead=lead)
            self._expand_node(
                node,
                logits_list[idx],
                eval_,
                legal_action_moves,
                policy_temperature,
            )
            evaluations[index] = eval_

        return [e for e in evaluations if e is not None]

    def _expand_node(
        self,
        node: SearchNode,
        logits,
        evaluation: SearchEvaluation,
        legal_action_moves: Optional[Sequence[Tuple[int, Move]]] = None,
        policy_temperature: Optional[float] = None,
    ) -> None:
        if legal_action_moves is None:
            legal_action_moves = self._legal_action_moves(node.state)
        if policy_temperature is None:
            policy_temperature = self._policy_temperature

        legal_actions = [action for action, _ in legal_action_moves]
        priors = self._masked_priors(logits, legal_actions, node.state, policy_temperature)

        # legal_action_moves is already in ascending action order (pawn actions
        # 0-3, then walls at strictly increasing offsets), so build the edge
        # dict directly in that order without re-sorting.
        node.edges = {
            action: SearchEdge(action=action, move=move, prior=prior)
            for (action, move), prior in zip(legal_action_moves, priors)
        }
        node.value_estimate = evaluation.value
        node.lead_estimate = evaluation.lead
        node.is_expanded = True

    def _masked_priors(
        self,
        logits,
        legal_actions: Sequence[int],
        state: BarricadeState,
        temperature: Optional[float] = None,
    ) -> List[float]:
        """Return masked priors as a list aligned to ``legal_actions`` order."""
        if not legal_actions:
            return []

        if temperature is None:
            temperature = self._policy_temperature

        if isinstance(logits, Tensor):
            logits = logits.tolist()

        # Map legal actions to network indices. Skip the per-action function
        # call entirely when there is no transform (the common case).
        transform = self.policy_action_transform
        if transform is None:
            network_actions = legal_actions
        else:
            network_actions = [
                int(transform(int(action), state)) for action in legal_actions
            ]

        n = len(network_actions)
        legal_logits = [0.0] * n
        for i in range(n):
            val = logits[network_actions[i]]
            if val != val:  # NaN
                legal_logits[i] = 0.0
            elif val == float('inf'):
                legal_logits[i] = 80.0
            elif val == float('-inf'):
                legal_logits[i] = -80.0
            else:
                legal_logits[i] = val

        if temperature <= 0:
            best_idx = 0
            best_val = legal_logits[0]
            for i in range(1, n):
                if legal_logits[i] > best_val:
                    best_val = legal_logits[i]
                    best_idx = i
            priors = [0.0] * n
            priors[best_idx] = 1.0
            return priors

        temp = temperature if temperature > 1e-6 else 1e-6
        max_logit = max(legal_logits)

        exp_values = [0.0] * n
        total = 0.0
        all_finite = True
        for i in range(n):
            e = math.exp((legal_logits[i] - max_logit) / temp)
            if not math.isfinite(e):
                all_finite = False
                break
            exp_values[i] = e
            total += e

        if total <= 0 or not all_finite:
            uniform = 1.0 / n
            return [uniform] * n

        inv_total = 1.0 / total
        return [e * inv_total for e in exp_values]

    # ------------------------------------------------------------------
    # Root Dirichlet noise
    # ------------------------------------------------------------------

    def _add_root_dirichlet_noise(self, root: SearchNode) -> None:
        if not root.edges:
            return

        alpha = self._root_dirichlet_alpha
        fraction = self._root_exploration_fraction
        if alpha <= 0 or fraction <= 0:
            root.root_noise_applied = True
            return

        actions = list(root.edges.keys())
        n = len(actions)

        gammavariate = self.rng.gammavariate
        noise = [gammavariate(alpha, 1.0) for _ in range(n)]
        total_noise = sum(noise)
        if total_noise > 0:
            inv_total_noise = 1.0 / total_noise
            noise = [x * inv_total_noise for x in noise]
        else:
            uniform = 1.0 / n
            noise = [uniform] * n

        one_minus_frac = 1.0 - fraction
        edges = root.edges

        for i in range(n):
            edge = edges[actions[i]]
            edge.prior = one_minus_frac * edge.prior + fraction * noise[i]

        total_prior = 0.0
        for edge in edges.values():
            total_prior += edge.prior
        if total_prior > 0:
            inv_total_prior = 1.0 / total_prior
            for edge in edges.values():
                edge.prior *= inv_total_prior
        root.root_noise_applied = True

    # ------------------------------------------------------------------
    # Backpropagation (hot path)
    # ------------------------------------------------------------------

    def _backpropagate(
        self,
        path: Sequence[SearchEdge],
        leaf_evaluation: SearchEvaluation,
    ) -> None:
        value = leaf_evaluation.value
        lead = leaf_evaluation.lead
        for i in range(len(path) - 1, -1, -1):
            edge = path[i]
            if edge.virtual_visits > 0:
                edge.virtual_visits -= 1
            value = -value
            lead = -lead
            edge.value_sum += value
            edge.lead_sum += lead
            edge.visits += 1

    def _release_virtual_visits(self, path: Sequence[SearchEdge]) -> None:
        for i in range(len(path)):
            edge = path[i]
            if edge.virtual_visits > 0:
                edge.virtual_visits -= 1

    def _clear_virtual_visits(self, root: Optional[SearchNode]) -> None:
        if root is None:
            return
        stack: List[SearchNode] = [root]
        while stack:
            node = stack.pop()
            node.in_flight = 0
            for edge in node.edges.values():
                if edge.virtual_visits:
                    edge.virtual_visits = 0
                child = edge.child
                if child is not None:
                    stack.append(child)

    # ------------------------------------------------------------------
    # Terminal evaluation
    # ------------------------------------------------------------------

    def _winner_evaluation(self, node: SearchNode) -> Optional[SearchEvaluation]:
        winner = node.state.winner
        if winner is not None:
            value = 1.0 if winner == node.state.current_player else -1.0
            return SearchEvaluation(value=value, lead=self._state_lead(node.state))
        return None

    def _terminal_evaluation(self, node: SearchNode) -> Optional[SearchEvaluation]:
        if node.terminal_eval is not None:
            return node.terminal_eval
        winner_eval = self._winner_evaluation(node)
        if winner_eval is not None:
            return winner_eval
        if node.is_expanded:
            if not node.edges:
                return SearchEvaluation(value=-1.0, lead=self._state_lead(node.state))
            return None
        if not node.state.legal_moves():
            return SearchEvaluation(value=-1.0, lead=self._state_lead(node.state))
        return None

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def _child_history(self, node: SearchNode) -> Tuple[BarricadeState, ...]:
        if not self._use_history:
            return ()
        return self._trim_history((*node.history, node.state))

    def _trim_history(
        self, history: Sequence[BarricadeState]
    ) -> Tuple[BarricadeState, ...]:
        if not self._use_history:
            return ()
        return tuple(history[-self._history_length:])

    # ------------------------------------------------------------------
    # CPUCT
    # ------------------------------------------------------------------

    def _cpuct(self, parent_effective_visits: int) -> float:
        return self._cpuct_fast(parent_effective_visits)

    def _cpuct_fast(self, parent_effective_visits: int) -> float:
        base = self._cpuct_base
        return self._cpuct_init + math.log(
            (parent_effective_visits + base + 1.0) / base
        ) * self._cpuct_factor

    # ------------------------------------------------------------------
    # Policy extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _parent_q(node: SearchNode) -> float:
        visits = sum(edge.visits for edge in node.edges.values())
        if visits == 0:
            return 0.0
        return sum(edge.value_sum for edge in node.edges.values()) / visits

    @staticmethod
    def _parent_lead_q(node: SearchNode) -> float:
        visits = sum(edge.visits for edge in node.edges.values())
        if visits == 0:
            return 0.0
        return sum(edge.lead_sum for edge in node.edges.values()) / visits

    def _lead_bonus(self, lead: float) -> float:
        if not self._use_lead_bonus:
            return 0.0
        return self._lead_weight * math.tanh(float(lead) * self._inv_lead_scale)

    def _policy_target(self, root: SearchNode, temperature: float) -> List[float]:
        stats = [
            (action, edge.visits)
            for action, edge in root.edges.items()
            if edge.visits > 0
        ]
        policy = [0.0] * self.action_size
        if not stats:
            legal_actions = list(root.edges.keys())
            if not legal_actions:
                return policy
            uniform = 1.0 / len(legal_actions)
            for action in legal_actions:
                policy[action] = uniform
            return policy

        if temperature <= 0:
            best_action = max(
                stats,
                key=lambda item: (
                    item[1],
                    root.edges[item[0]].q,
                    root.edges[item[0]].lead_q,
                ),
            )[0]
            policy[best_action] = 1.0
            return policy

        inv_temp = 1.0 / max(temperature, 1e-6)
        weights = [
            (action, float(visits) ** inv_temp)
            for action, visits in stats
        ]
        total = sum(w for _, w in weights)
        if total <= 0:
            uniform = 1.0 / len(stats)
            for action, _ in stats:
                policy[action] = uniform
            return policy

        inv_total = 1.0 / total
        for action, weight in weights:
            policy[action] = weight * inv_total
        return policy

    def _sample_policy(self, policy: Sequence[float]) -> int:
        total = sum(policy)
        if total <= 0:
            return -1
        return self.rng.choices(range(len(policy)), weights=policy, k=1)[0]

    @staticmethod
    def _policy_from_stats(
        stats: Sequence[ActionStats],
        temperature: float,
        action_size: int,
    ) -> List[float]:
        policy = [0.0] * int(action_size)
        if not stats:
            return policy

        if temperature <= 0:
            best = max(stats, key=lambda item: (item.visits, item.q, item.lead_q))
            policy[best.action] = 1.0
            return policy

        inv_temp = 1.0 / max(temperature, 1e-6)
        weights = [
            (stat.action, float(stat.visits) ** inv_temp)
            for stat in stats
            if stat.visits > 0
        ]
        if not weights:
            uniform = 1.0 / len(stats)
            for stat in stats:
                policy[stat.action] = uniform
            return policy

        total = sum(w for _, w in weights)
        inv_total = 1.0 / total
        for action, weight in weights:
            policy[action] = weight * inv_total
        return policy

    def _action_stats(self, root: SearchNode, policy_target: Sequence[float]) -> List[ActionStats]:
        return [
            ActionStats(
                action=edge.action,
                move=edge.move,
                prior=edge.prior,
                visits=edge.visits,
                virtual_visits=edge.virtual_visits,
                q=edge.q,
                lead_q=edge.lead_q,
                value_sum=edge.value_sum,
                lead_sum=edge.lead_sum,
                policy=policy_target[edge.action],
            )
            for edge in sorted(root.edges.values(), key=lambda item: item.action)
        ]

    def _root_value(self, root: SearchNode) -> float:
        visits = root.real_visits
        if visits > 0:
            return sum(edge.value_sum for edge in root.edges.values()) / visits
        terminal_evaluation = self._terminal_evaluation(root)
        if terminal_evaluation is not None:
            return terminal_evaluation.value
        return float(root.value_estimate or 0.0)

    def _root_lead(self, root: SearchNode) -> float:
        visits = root.real_visits
        if visits > 0:
            return sum(edge.lead_sum for edge in root.edges.values()) / visits
        terminal_evaluation = self._terminal_evaluation(root)
        if terminal_evaluation is not None:
            return terminal_evaluation.lead
        return float(root.lead_estimate or 0.0)

    # ------------------------------------------------------------------
    # Encoding / move helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_state_encoder(
        state: BarricadeState,
        history: Sequence[BarricadeState],
        history_length: int,
    ) -> Tensor:
        return encode_state_stack(
            state,
            history,
            history_length=history_length,
        )

    def _encode_move(self, move: Move, state: BarricadeState) -> int:
        if self.encode_move_fn is not None:
            return int(self.encode_move_fn(move))
        if hasattr(state, "encode_move"):
            return int(state.encode_move(move))
        return int(encode_move(move))

    def _legal_action_moves(
        self, state: BarricadeState
    ) -> Sequence[Tuple[int, Move]]:
        legal_action_moves = getattr(state, "legal_action_moves", None)
        if legal_action_moves is not None:
            # The env already returns a cached, ascending-order tuple of
            # (int_action, move) pairs. It is consumed read-only downstream, so
            # reuse it directly instead of rebuilding the list every call.
            return legal_action_moves()
        legal_moves = state.legal_moves()
        return [
            (self._encode_move(move, state), move)
            for move in legal_moves
        ]

    def _network_action(self, action: int, state: BarricadeState) -> int:
        if self.policy_action_transform is None:
            return int(action)
        return int(self.policy_action_transform(int(action), state))

    def _model_lead(self, lead: float, state: BarricadeState) -> float:
        if self.lead_transform is not None:
            return float(self.lead_transform(float(lead), state))
        return self._lead_for_player(float(lead), state.current_player)

    def _model_inference(self, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if hasattr(self.model, "inference"):
            return self.model.inference(batch)

        outputs = self.model(batch)
        if not isinstance(outputs, tuple):
            raise RuntimeError("MCTS model must return policy logits and value.")
        if len(outputs) >= 3:
            return outputs[0], outputs[1], outputs[2]
        logits, values = outputs[:2]
        leads = torch.zeros_like(values)
        return logits, values, leads

    def _state_lead(self, state: BarricadeState) -> float:
        red_distance = state.shortest_path_length(Player.RED)
        blue_distance = state.shortest_path_length(Player.BLUE)
        if red_distance is None or blue_distance is None:
            return 0.0
        raw_lead = float(red_distance - blue_distance)
        return self._lead_for_player(raw_lead, state.current_player)

    @staticmethod
    def _lead_for_player(raw_lead: float, player: Player) -> float:
        return float(raw_lead) if player == Player.BLUE else -float(raw_lead)

    def _resolve_device(
        self,
        configured_device: Optional[str | torch.device],
    ) -> torch.device:
        if configured_device is not None:
            return torch.device(configured_device)
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")