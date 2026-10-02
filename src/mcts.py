"""Bounded deterministic MCTS over production ``ActionCandidate`` boards.

Search owns only ``SearchState`` snapshots and branch-local policy context. It
does not call ``PolicyController.decide``: hypothetical attack boards are
created with the same production build/filter/rank methods, without committing
actions, logging, or changing live controller memory.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field

from .config import DECISION_INTERVAL
from .policy import ActionCandidate, PlayerIntent, PolicyController, PolicyInput
from .search_state import SearchState, advance_search_state, is_within_pitch, transition_candidate
from .state import Ball, GameState, Player, WorldModel
from .teamplan import TeamPlan, TeamPlanManager


@dataclass(frozen=True)
class RootActionStats:
    """Visits and backed-up value for one original production root candidate."""

    candidate: ActionCandidate
    visits: int
    value: float

    @property
    def mean_value(self) -> float:
        return self.value / self.visits if self.visits else 0.0


@dataclass(frozen=True)
class SearchDiagnostics:
    iterations: int = 0
    nodes_created: int = 0
    max_depth: int = 0
    simulated_steps: int = 0
    root_actions: tuple[RootActionStats, ...] = ()


@dataclass
class _Node:
    state: SearchState
    parent: _Node | None
    candidate: ActionCandidate | None
    depth: int
    steps_from_root: int
    team_plan_manager: TeamPlanManager
    candidates: list[ActionCandidate]
    unexpanded: list[int]
    children: dict[int, _Node] = field(default_factory=dict)
    visits: int = 0
    value: float = 0.0


class MCTSPlanner:
    """A small UCT planner whose root and descendant actions are production options.

    ``search`` is the standalone API and returns an original root
    ``ActionCandidate``. ``choose`` retains the old opt-in runtime API by
    replacing only the possessor's intent in a copy of the supplied intent map.
    The runtime remains opt-in through its existing MCTS configuration gate.
    """

    def __init__(
        self,
        genome: dict[str, float] | None = None,
        weights: dict[str, float] | None = None,
        rng: random.Random | None = None,
        iterations: int = 24,
        horizon: float = 1.5,
        cpuct: float = 1.4,
        *,
        max_decision_depth: int = 3,
        max_advance_steps: int = 8,
    ) -> None:
        # Keep the old constructor shape for explicitly opt-in callers. Search
        # is deterministic and does not consume this RNG or use reward weights.
        self.genome = copy.deepcopy(genome or {})
        self.weights = copy.deepcopy(weights or {})
        self.rng = rng or random.Random(0)
        self.iterations = max(0, int(iterations))
        configured_horizon = float(horizon)
        self.horizon = (
            max(0.0, configured_horizon)
            if math.isfinite(configured_horizon)
            else 1.5
        )
        configured_cpuct = float(cpuct)
        self.cpuct = (
            max(0.0, configured_cpuct)
            if math.isfinite(configured_cpuct)
            else 1.4
        )
        self.max_decision_depth = max(1, int(max_decision_depth))
        self.max_advance_steps = max(0, int(max_advance_steps))
        self.max_simulated_steps = max(
            1, int(math.ceil(self.horizon / DECISION_INTERVAL))
        )
        self.stats = SearchDiagnostics()
        self.root_candidates: tuple[ActionCandidate, ...] = ()
        self._root: _Node | None = None

    def search(
        self,
        inp: PolicyInput,
        controller: PolicyController | None = None,
        team_plan: TeamPlan | None = None,
    ) -> ActionCandidate | None:
        """Search from ``inp`` and return one of its production root candidates.

        Candidate generation calls only ``_build_attack_candidates``,
        ``_filter_attack_candidates`` and ``_rank_attack_candidates``. Passing
        the live controller is safe: those methods are read-only; simulated
        team-plan managers and movement-policy controllers are branch-local.
        """
        controller = controller or PolicyController()
        possessor = inp.state.our_possessor()
        if (
            possessor is None
            or not possessor.can_act
            or inp.state.phase != "openPlay"
            or inp.tr <= 0.0
        ):
            self._reset_search(())
            return None

        manager = self._initial_team_plan_manager(inp, controller, team_plan)
        candidates = self._production_candidates(inp, controller, manager.current_plan)
        if not candidates:
            self._reset_search(())
            return None

        candidates = tuple(candidates)
        self._reset_search(candidates)
        root_state = SearchState.from_policy_input(inp)
        root = _Node(
            state=root_state,
            parent=None,
            candidate=None,
            depth=0,
            steps_from_root=0,
            team_plan_manager=manager,
            candidates=list(candidates),
            unexpanded=list(range(len(candidates))),
        )
        self._root = root

        if self.iterations == 0:
            return candidates[0]

        completed_iterations = 0
        nodes_created = 1
        max_depth = 0
        simulated_steps = 0

        for _ in range(self.iterations):
            path = [root]
            node = root

            while (
                node.depth < self.max_decision_depth
                and node.steps_from_root < self.max_simulated_steps
                and node.candidates
            ):
                if node.unexpanded:
                    child, added_steps = self._expand(node, inp, controller)
                    simulated_steps += added_steps
                    if child is not None:
                        node = child
                        path.append(node)
                        nodes_created += 1
                        max_depth = max(max_depth, node.depth)
                    break
                if not node.children:
                    break
                node = self._select(node)
                path.append(node)

            leaf_state, rollout_steps, rollout_depth = self._rollout(
                node, inp, controller
            )
            simulated_steps += rollout_steps
            max_depth = max(max_depth, rollout_depth)
            value = self._evaluate(leaf_state, root_state)
            self._backpropagate(path, value)
            completed_iterations += 1

        root_stats = tuple(
            RootActionStats(
                candidate=candidate,
                visits=(root.children[index].visits if index in root.children else 0),
                value=(root.children[index].value if index in root.children else 0.0),
            )
            for index, candidate in enumerate(candidates)
        )
        self.stats = SearchDiagnostics(
            iterations=completed_iterations,
            nodes_created=nodes_created,
            max_depth=max_depth,
            simulated_steps=simulated_steps,
            root_actions=root_stats,
        )

        # Conventional final rule: most visits, then highest mean value, then
        # original production ranking order. Candidate.reason is never consulted.
        best_index = max(
            range(len(candidates)),
            key=lambda index: (
                root_stats[index].visits,
                root_stats[index].mean_value,
                -index,
            ),
        )
        return candidates[best_index]

    def choose(
        self,
        inp: PolicyInput,
        base_intents: dict[str, PlayerIntent],
        controller: PolicyController | None = None,
        team_plan: TeamPlan | None = None,
    ) -> dict[str, PlayerIntent] | None:
        """Compatibility wrapper for the existing explicitly enabled runtime.

        The standalone planner API is ``search``. This wrapper preserves the
        previous mapping return shape for the opt-in runtime path only.
        """
        selected = self.search(inp, controller=controller, team_plan=team_plan)
        if selected is None:
            return None
        result = dict(base_intents)
        result[selected.intent.pid] = selected.intent
        return result

    def _reset_search(self, candidates: tuple[ActionCandidate, ...]) -> None:
        self.root_candidates = candidates
        self._root = None
        self.stats = SearchDiagnostics()

    @staticmethod
    def _production_candidates(
        inp: PolicyInput,
        controller: PolicyController,
        team_plan: TeamPlan | None,
    ) -> list[ActionCandidate]:
        possessor = inp.state.our_possessor()
        if possessor is None or not possessor.can_act:
            return []
        generated = controller._build_attack_candidates(inp, possessor, team_plan)
        filtered = controller._filter_attack_candidates(possessor, generated)
        ranked = controller._rank_attack_candidates(filtered)
        return [
            candidate
            for candidate in ranked
            if math.isfinite(candidate.value)
        ]

    @staticmethod
    def _initial_team_plan_manager(
        inp: PolicyInput,
        controller: PolicyController,
        team_plan: TeamPlan | None,
    ) -> TeamPlanManager:
        live_manager = getattr(controller, "_team_plan_manager", None)
        manager = copy.deepcopy(live_manager) if live_manager is not None else TeamPlanManager()
        if team_plan is not None:
            manager.current_plan = copy.deepcopy(team_plan)
        return manager

    @staticmethod
    def _policy_input(
        state: SearchState,
        root: PolicyInput,
        steps_from_root: int,
    ) -> PolicyInput:
        """Rebuild only observable production-policy inputs from a search state.

        Cooldowns and other hidden engine timers are whatever LightEngine
        modeled; no observation-only fields are fabricated. Root tactical and
        match context is copied forward because SearchState does not contain it.
        """
        plan = state.plan
        game_state = GameState(
            protocol_version=root.state.protocol_version,
            game_id=root.state.game_id,
            sequence=root.state.sequence + steps_from_root,
            simulation_tick=root.state.simulation_tick + steps_from_root,
            apply_at_tick=root.state.apply_at_tick + steps_from_root,
            time_remaining=state.time_remaining,
            phase=root.state.phase,
            score_us=plan.score_us,
            score_them=plan.score_them,
            ball=Ball(
                x=plan.ball.x,
                y=plan.ball.y,
                vx=plan.ball.vx,
                vy=plan.ball.vy,
                possessing_team=plan.ball.possessing_team,
                possessing_player=plan.ball.possessing_player,
            ),
            us=tuple(
                Player(
                    id=p.pid,
                    team="us",
                    role="goalkeeper" if p.role == "goalkeeper" else "outfield",
                    x=p.x,
                    y=p.y,
                    vx=p.vx,
                    vy=p.vy,
                    facing=p.facing,
                    can_act=p.can_act,
                )
                for p in plan.players
                if p.team == "us"
            ),
            them=tuple(
                Player(
                    id=p.pid,
                    team="them",
                    role="goalkeeper" if p.role == "goalkeeper" else "outfield",
                    x=p.x,
                    y=p.y,
                    vx=p.vx,
                    vy=p.vy,
                    facing=p.facing,
                    can_act=p.can_act,
                )
                for p in plan.players
                if p.team == "them"
            ),
        )
        return PolicyInput(
            state=game_state,
            world=WorldModel.build(game_state),
            config=copy.deepcopy(root.config),
            tactical_state=root.tactical_state,
            press_plan=copy.deepcopy(root.press_plan),
            roles=dict(root.roles),
            their_possession_ticks=root.their_possession_ticks,
            opp=copy.deepcopy(root.opp),
            time_remaining=state.time_remaining,
            score_us=plan.score_us,
            score_them=plan.score_them,
            counter_press_active=root.counter_press_active,
            match_duration=root.match_duration,
        )

    @staticmethod
    def _refresh_team_plan(
        manager: TeamPlanManager,
        inp: PolicyInput,
    ) -> None:
        manager.update(inp, inp.state.our_possessor(), inp.state.simulation_tick)

    @staticmethod
    def _background_intents(
        inp: PolicyInput,
        team_plan: TeamPlan | None,
        candidate: ActionCandidate | None = None,
    ) -> dict[tuple[str, str], PlayerIntent]:
        """Get production off-ball movement using a fresh branch-local controller.

        The opposing side receives no invented policy intent: its players hold
        position while LightEngine still applies automatic control/goalkeeper
        behavior. This deterministic stationary-opponent response is deliberately
        optimistic and is not an adversarial model.
        """
        branch_controller = PolicyController()
        possessor = inp.state.our_possessor()
        if (
            candidate is not None
            and candidate.intent.action_type == "pass"
            and candidate.intent.receiver_id is not None
        ):
            branch_controller._pending_pass_receiver = candidate.intent.receiver_id
            branch_controller._pending_pass_collection = candidate.intent.collection_point

        intents: dict[tuple[str, str], PlayerIntent] = {}
        for player in inp.state.outfield_us():
            if possessor is not None and player.id == possessor.id:
                if not possessor.can_act:
                    intents[("us", player.id)] = branch_controller._decide_possessor(
                        inp, possessor, team_plan
                    )
                continue
            intents[("us", player.id)] = branch_controller._decide_off_ball(
                inp, player, possessor, team_plan
            )

        goalkeeper = inp.state.goalkeeper_us()
        if goalkeeper is not None:
            intents[("us", goalkeeper.id)] = branch_controller._decide_goalkeeper(
                inp, goalkeeper
            )
        return intents

    def _advance_until_decision(
        self,
        state: SearchState,
        manager: TeamPlanManager,
        root_input: PolicyInput,
        controller: PolicyController,
        steps_from_root: int,
    ) -> tuple[SearchState, TeamPlanManager, int, list[ActionCandidate]]:
        """Boundedly advance restarts, loose balls and unavailable decisions."""
        available_advances = min(
            self.max_advance_steps,
            max(0, self.max_simulated_steps - steps_from_root),
        )
        advanced = 0
        for _ in range(available_advances + 1):
            if (
                state.time_remaining <= 0.0
                or not is_within_pitch(state)
            ):
                return state, manager, advanced, []

            policy_input = self._policy_input(state, root_input, steps_from_root + advanced)
            self._refresh_team_plan(manager, policy_input)
            possessor = policy_input.state.our_possessor()
            if possessor is not None and possessor.can_act:
                board = self._production_candidates(
                    policy_input, controller, manager.current_plan
                )
                if board:
                    return state, manager, advanced, board

            if advanced >= available_advances:
                break
            background = self._background_intents(
                policy_input, manager.current_plan
            )
            state = advance_search_state(state, background)
            advanced += 1

        return state, manager, advanced, []

    def _expand(
        self,
        node: _Node,
        root_input: PolicyInput,
        controller: PolicyController,
    ) -> tuple[_Node | None, int]:
        while node.unexpanded:
            candidate_index = node.unexpanded.pop(0)
            candidate = node.candidates[candidate_index]
            policy_input = self._policy_input(
                node.state, root_input, node.steps_from_root
            )
            background = self._background_intents(
                policy_input, node.team_plan_manager.current_plan, candidate
            )
            try:
                child_state = transition_candidate(
                    node.state, candidate, background_intents=background
                )
            except ValueError:
                # The board is production-owned; still fail closed if a future
                # candidate becomes unusable after adaptation to simulated data.
                continue

            child_steps = node.steps_from_root + 1
            child_manager = copy.deepcopy(node.team_plan_manager)
            child_state, child_manager, waited, child_candidates = (
                self._advance_until_decision(
                    child_state,
                    child_manager,
                    root_input,
                    controller,
                    child_steps,
                )
            )
            child_steps += waited
            child = _Node(
                state=child_state,
                parent=node,
                candidate=candidate,
                depth=node.depth + 1,
                steps_from_root=child_steps,
                team_plan_manager=child_manager,
                candidates=child_candidates,
                unexpanded=list(range(len(child_candidates))),
            )
            node.children[candidate_index] = child
            return child, 1 + waited
        return None, 0

    @staticmethod
    def _uct_score(
        parent_visits: int,
        child_visits: int,
        child_value: float,
        exploration_constant: float,
    ) -> float:
        if child_visits <= 0:
            return 0.0
        if not math.isfinite(child_value) or not math.isfinite(exploration_constant):
            return -1e9
        mean_value = child_value / child_visits
        exploration = exploration_constant * math.sqrt(
            math.log(max(1, parent_visits)) / child_visits
        )
        score = mean_value + exploration
        return score if math.isfinite(score) else -1e9

    def _select(self, node: _Node) -> _Node:
        """Select the highest UCT child; sorted keys provide stable tie-breaking."""
        unvisited = [
            node.children[index]
            for index in sorted(node.children)
            if node.children[index].visits == 0
        ]
        if unvisited:
            return unvisited[0]
        best_child: _Node | None = None
        best_score = -math.inf
        for index in sorted(node.children):
            child = node.children[index]
            score = self._uct_score(
                node.visits, child.visits, child.value, self.cpuct
            )
            if score > best_score:
                best_child = child
                best_score = score
        if best_child is None:
            raise ValueError("cannot select from a node without children")
        return best_child

    def _rollout(
        self,
        start: _Node,
        root_input: PolicyInput,
        controller: PolicyController,
    ) -> tuple[SearchState, int, int]:
        """Roll out the highest production-ranked candidate deterministically."""
        state = start.state
        manager = copy.deepcopy(start.team_plan_manager)
        steps_from_root = start.steps_from_root
        added_steps = 0
        depth = start.depth
        while (
            depth < self.max_decision_depth
            and steps_from_root < self.max_simulated_steps
            and state.time_remaining > 0.0
            and is_within_pitch(state)
        ):
            state, manager, waited, board = self._advance_until_decision(
                state,
                manager,
                root_input,
                controller,
                steps_from_root,
            )
            steps_from_root += waited
            added_steps += waited
            if not board or steps_from_root >= self.max_simulated_steps:
                break

            candidate = board[0]  # _production_candidates is EV-ranked.
            policy_input = self._policy_input(state, root_input, steps_from_root)
            background = self._background_intents(
                policy_input, manager.current_plan, candidate
            )
            try:
                state = transition_candidate(
                    state, candidate, background_intents=background
                )
            except ValueError:
                break
            steps_from_root += 1
            added_steps += 1
            depth += 1
            # No random rollout policy: ties and repeated states remain fully
            # reproducible for the same input and planner configuration.

        return state, added_steps, depth

    @staticmethod
    def _evaluate(state: SearchState, root: SearchState) -> float:
        """Small bounded leaf value from our perspective; goals dominate shape."""
        plan = state.plan
        root_plan = root.plan
        goal_difference = (
            (plan.score_us - root_plan.score_us)
            - (plan.score_them - root_plan.score_them)
        )
        value = 2.0 * goal_difference

        progress = (plan.ball.x - root_plan.ball.x) / 60.0
        value += 0.12 * progress
        if plan.ball.possessing_team == "us":
            value += 0.08 + 0.06 * (plan.ball.x / 60.0)
        elif plan.ball.possessing_team == "them":
            value -= 0.08 + 0.06 * (1.0 - plan.ball.x / 60.0)

        if not math.isfinite(value):
            return -2.0
        return max(-2.5, min(2.5, value))

    @staticmethod
    def _backpropagate(path: list[_Node], value: float) -> None:
        if not math.isfinite(value):
            value = -2.0
        for node in path:
            node.visits += 1
            node.value += value
