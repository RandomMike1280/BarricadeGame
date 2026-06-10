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
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from barricade_env import ACTION_SIZE, BarricadeState, Move, Player, encode_move
from network import encode_state_stack


@dataclass(frozen=True)
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
    history_length: int = 0
    device: Optional[str] = None
    add_root_noise: bool = True
    lead_weight: float = 0.01
    lead_scale: float = 10.0


@dataclass(frozen=True)
class SearchEvaluation:
    value: float
    lead: float


@dataclass
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


@dataclass
class SearchNode:
    state: BarricadeState
    history: Tuple[BarricadeState, ...] = ()
    edges: Dict[int, SearchEdge] = field(default_factory=dict)
    is_expanded: bool = False
    in_flight: int = 0
    value_estimate: Optional[float] = None
    lead_estimate: Optional[float] = None
    root_noise_applied: bool = False

    @property
    def effective_visits(self) -> int:
        return sum(edge.effective_visits for edge in self.edges.values())

    @property
    def real_visits(self) -> int:
        return sum(edge.visits for edge in self.edges.values())


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class MCTSResult:
    action: int
    policy_target: List[float]
    root_value: float
    root_lead: float
    root: SearchNode
    stats: List[ActionStats]
    diagnostics: Dict[str, int] = field(default_factory=dict)


class MCTS:
    def __init__(self, model, config: Optional[MCTSConfig] = None) -> None:
        self.model = model
        self.config = config or MCTSConfig()
        self.device = self._resolve_device(self.config.device)
        self.model.to(self.device)
        self.rng = random.Random()

    def search(
        self,
        state: BarricadeState,
        history: Optional[Sequence[BarricadeState]] = None,
        root: Optional[SearchNode] = None,
    ) -> MCTSResult:
        root = self._prepare_root(state, history, root)

        if self._terminal_evaluation(root) is None and not root.is_expanded:
            self._evaluate_and_expand([root])

        if (
            self.config.add_root_noise
            and root.is_expanded
            and not root.root_noise_applied
        ):
            self._add_root_dirichlet_noise(root)

        completed = 0
        collision_streak = 0
        collisions = 0
        neural_batches = 0
        evaluated_leaves = 0
        while completed < self.config.num_simulations:
            pending: List[Tuple[SearchNode, List[SearchEdge]]] = []
            terminal_paths: List[Tuple[List[SearchEdge], SearchEvaluation]] = []
            target_batch = min(
                max(1, self.config.batch_size),
                self.config.num_simulations - completed,
            )

            while len(pending) + len(terminal_paths) < target_batch:
                selection = self._select_leaf(root)
                kind, leaf, path, evaluation = selection

                if kind == "collision":
                    self._release_virtual_visits(path)
                    collision_streak += 1
                    collisions += 1
                    if collision_streak > max(16, target_batch * 4):
                        break
                    continue

                collision_streak = 0
                if kind == "terminal":
                    terminal_paths.append((path, evaluation))
                elif kind == "leaf":
                    pending.append((leaf, path))
                else:
                    raise RuntimeError(f"Unknown selection kind: {kind}")

                if completed + len(pending) + len(terminal_paths) >= self.config.num_simulations:
                    break

            if not pending and not terminal_paths:
                break

            for path, evaluation in terminal_paths:
                self._backpropagate(path, evaluation)
            completed += len(terminal_paths)

            if pending:
                neural_batches += 1
                evaluated_leaves += len(pending)
                evaluations = self._evaluate_and_expand([leaf for leaf, _ in pending])
                for (leaf, path), evaluation in zip(pending, evaluations):
                    leaf.in_flight = max(0, leaf.in_flight - 1)
                    self._backpropagate(path, evaluation)
                completed += len(pending)

        self._clear_virtual_visits(root)
        policy_target = self._policy_target(root, self.config.action_temperature)
        action = self._sample_policy(policy_target)
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

        policy = self._policy_from_stats(result.stats, temperature)
        return self._sample_policy(policy)

    def advance_root(self, root: SearchNode, action: int) -> Optional[SearchNode]:
        edge = root.edges.get(int(action))
        if edge is None:
            return None
        return edge.child

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
        return SearchNode(state.copy(), trimmed_history)

    def _select_leaf(
        self, root: SearchNode
    ) -> Tuple[str, Optional[SearchNode], List[SearchEdge], Optional[SearchEvaluation]]:
        node = root
        path: List[SearchEdge] = []

        while True:
            terminal_evaluation = self._terminal_evaluation(node)
            if terminal_evaluation is not None:
                return "terminal", node, path, terminal_evaluation

            if not node.is_expanded:
                if node.in_flight > 0:
                    return "collision", node, path, None
                node.in_flight += 1
                return "leaf", node, path, None

            if not node.edges:
                return "terminal", node, path, SearchEvaluation(
                    value=-1.0,
                    lead=self._state_lead(node.state),
                )

            edge = self._select_edge(node)
            edge.virtual_visits += 1
            path.append(edge)
            if edge.child is None:
                edge.child = SearchNode(
                    state=node.state.apply_move(edge.move),
                    history=self._child_history(node),
                )
            node = edge.child

    def _select_edge(self, node: SearchNode) -> SearchEdge:
        parent_effective_visits = max(1, node.effective_visits)
        cpuct = self._cpuct(parent_effective_visits)
        parent_q = self._parent_q(node)
        parent_lead_q = self._parent_lead_q(node)
        explored_prior = sum(
            edge.prior for edge in node.edges.values() if edge.effective_visits > 0
        )
        fpu_q = parent_q - self.config.fpu_reduction * math.sqrt(
            max(0.0, explored_prior)
        )

        best_score = -math.inf
        best_edge = None
        for edge in node.edges.values():
            q_value = edge.q if edge.visits > 0 else fpu_q
            lead_value = edge.lead_q if edge.visits > 0 else parent_lead_q
            u_value = (
                cpuct
                * edge.prior
                * math.sqrt(parent_effective_visits)
                / (1 + edge.effective_visits)
            )
            score = q_value + u_value + self._lead_bonus(lead_value)
            if score > best_score:
                best_score = score
                best_edge = edge

        if best_edge is None:
            raise RuntimeError("Cannot select from a node with no edges.")
        return best_edge

    def _evaluate_and_expand(self, nodes: Sequence[SearchNode]) -> List[SearchEvaluation]:
        if not nodes:
            return []

        batch = torch.stack(
            [
                encode_state_stack(
                    node.state,
                    node.history,
                    history_length=self.config.history_length,
                )
                for node in nodes
            ],
            dim=0,
        ).to(self.device)

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            logits, values, leads = self._model_inference(batch)
        if was_training:
            self.model.train()

        logits = logits.detach().cpu()
        values = values.detach().cpu().view(-1)
        leads = leads.detach().cpu().view(-1)

        evaluations = []
        for node, node_logits, node_value, node_lead in zip(nodes, logits, values, leads):
            value = float(node_value.item())
            lead = self._lead_for_player(float(node_lead.item()), node.state.current_player)
            evaluation = SearchEvaluation(value=value, lead=lead)
            self._expand_node(node, node_logits, evaluation)
            evaluations.append(evaluation)
        return evaluations

    def _expand_node(
        self,
        node: SearchNode,
        logits: Tensor,
        evaluation: SearchEvaluation,
    ) -> None:
        legal_moves = node.state.get_valid_moves()
        legal_actions = [encode_move(move) for move in legal_moves]
        priors = self._masked_priors(logits, legal_actions)
        edges = {}
        for move, action in zip(legal_moves, legal_actions):
            edges[action] = SearchEdge(
                action=action,
                move=move,
                prior=priors.get(action, 0.0),
            )
        node.edges = dict(sorted(edges.items()))
        node.value_estimate = evaluation.value
        node.lead_estimate = evaluation.lead
        node.is_expanded = True

    def _masked_priors(self, logits: Tensor, legal_actions: Sequence[int]) -> Dict[int, float]:
        legal_actions = list(legal_actions)
        if not legal_actions:
            return {}

        legal_logits = logits[legal_actions].float()
        legal_logits = torch.nan_to_num(legal_logits, nan=0.0, posinf=80.0, neginf=-80.0)

        temperature = self.config.policy_temperature
        if temperature <= 0:
            probs = torch.zeros_like(legal_logits)
            probs[int(torch.argmax(legal_logits).item())] = 1.0
        else:
            probs = torch.softmax(legal_logits / max(temperature, 1e-6), dim=0)

        if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0:
            probs = torch.full_like(legal_logits, 1.0 / len(legal_actions))
        else:
            probs = probs / probs.sum()

        return {action: float(prob) for action, prob in zip(legal_actions, probs)}

    def _add_root_dirichlet_noise(self, root: SearchNode) -> None:
        if not root.edges:
            return

        alpha = self.config.root_dirichlet_alpha
        fraction = self.config.root_exploration_fraction
        if alpha <= 0 or fraction <= 0:
            root.root_noise_applied = True
            return

        actions = list(root.edges.keys())
        concentration = torch.full((len(actions),), float(alpha), dtype=torch.float32)
        noise = torch.distributions.Dirichlet(concentration).sample().tolist()
        for action, noise_value in zip(actions, noise):
            edge = root.edges[action]
            edge.prior = (1.0 - fraction) * edge.prior + fraction * float(noise_value)

        total_prior = sum(edge.prior for edge in root.edges.values())
        if total_prior > 0:
            for edge in root.edges.values():
                edge.prior /= total_prior
        root.root_noise_applied = True

    def _backpropagate(
        self,
        path: Sequence[SearchEdge],
        leaf_evaluation: SearchEvaluation,
    ) -> None:
        node_value = float(leaf_evaluation.value)
        node_lead = float(leaf_evaluation.lead)
        for edge in reversed(path):
            edge.virtual_visits = max(0, edge.virtual_visits - 1)
            parent_value = -node_value
            parent_lead = -node_lead
            edge.value_sum += parent_value
            edge.lead_sum += parent_lead
            edge.visits += 1
            node_value = parent_value
            node_lead = parent_lead

    def _release_virtual_visits(self, path: Sequence[SearchEdge]) -> None:
        for edge in path:
            edge.virtual_visits = max(0, edge.virtual_visits - 1)

    def _clear_virtual_visits(self, node: Optional[SearchNode]) -> None:
        if node is None:
            return
        node.in_flight = 0
        for edge in node.edges.values():
            edge.virtual_visits = 0
            self._clear_virtual_visits(edge.child)

    def _terminal_evaluation(self, node: SearchNode) -> Optional[SearchEvaluation]:
        if node.state.winner is not None:
            value = 1.0 if node.state.winner == node.state.current_player else -1.0
            return SearchEvaluation(value=value, lead=self._state_lead(node.state))
        if not node.state.get_valid_moves():
            return SearchEvaluation(value=-1.0, lead=self._state_lead(node.state))
        return None

    def _child_history(self, node: SearchNode) -> Tuple[BarricadeState, ...]:
        if self.config.history_length <= 0:
            return ()
        return self._trim_history((*node.history, node.state))

    def _trim_history(
        self, history: Sequence[BarricadeState]
    ) -> Tuple[BarricadeState, ...]:
        if self.config.history_length <= 0:
            return ()
        return tuple(history[-self.config.history_length :])

    def _cpuct(self, parent_effective_visits: int) -> float:
        base = max(self.config.cpuct_base, 1e-6)
        return self.config.cpuct_init + math.log(
            (parent_effective_visits + base + 1.0) / base
        ) * self.config.cpuct_factor

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
        if self.config.lead_weight <= 0:
            return 0.0
        scale = max(float(self.config.lead_scale), 1e-6)
        return self.config.lead_weight * math.tanh(float(lead) / scale)

    def _policy_target(self, root: SearchNode, temperature: float) -> List[float]:
        stats = [
            (action, edge.visits)
            for action, edge in root.edges.items()
            if edge.visits > 0
        ]
        policy = [0.0] * ACTION_SIZE
        if not stats:
            legal_actions = list(root.edges.keys())
            if not legal_actions:
                return policy
            for action in legal_actions:
                policy[action] = 1.0 / len(legal_actions)
            return policy

        if temperature <= 0:
            best_action, _ = max(
                stats,
                key=lambda item: (
                    item[1],
                    root.edges[item[0]].q,
                    root.edges[item[0]].lead_q,
                ),
            )
            policy[best_action] = 1.0
            return policy

        weights = [
            (action, float(visits) ** (1.0 / max(temperature, 1e-6)))
            for action, visits in stats
        ]
        total = sum(weight for _, weight in weights)
        if total <= 0:
            for action, _ in stats:
                policy[action] = 1.0 / len(stats)
            return policy

        for action, weight in weights:
            policy[action] = weight / total
        return policy

    def _sample_policy(self, policy: Sequence[float]) -> int:
        total = sum(policy)
        if total <= 0:
            return -1
        return self.rng.choices(range(len(policy)), weights=policy, k=1)[0]

    @staticmethod
    def _policy_from_stats(stats: Sequence[ActionStats], temperature: float) -> List[float]:
        policy = [0.0] * ACTION_SIZE
        if not stats:
            return policy

        if temperature <= 0:
            best = max(stats, key=lambda item: (item.visits, item.q, item.lead_q))
            policy[best.action] = 1.0
            return policy

        weights = [
            (stat.action, float(stat.visits) ** (1.0 / max(temperature, 1e-6)))
            for stat in stats
            if stat.visits > 0
        ]
        if not weights:
            for stat in stats:
                policy[stat.action] = 1.0 / len(stats)
            return policy

        total = sum(weight for _, weight in weights)
        for action, weight in weights:
            policy[action] = weight / total
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
        red_distance = state.greedy_path_length(Player.RED)
        blue_distance = state.greedy_path_length(Player.BLUE)
        if red_distance is None or blue_distance is None:
            return 0.0
        raw_lead = float(red_distance - blue_distance)
        return self._lead_for_player(raw_lead, state.current_player)

    @staticmethod
    def _lead_for_player(raw_lead: float, player: Player) -> float:
        return float(raw_lead) if player == Player.BLUE else -float(raw_lead)

    def _resolve_device(self, configured_device: Optional[str]) -> torch.device:
        if configured_device is not None:
            return torch.device(configured_device)
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
