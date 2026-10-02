"""Isolated search/simulation boundary for production attacking candidates.

The search state contains only the lightweight simulator's physical state and
remaining match time. Policy coordination, logs, and controller memory are not
part of a hypothetical branch.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass

from .geom import PITCH_LENGTH, PITCH_WIDTH
from .light import LIntent, LPlayer, LBall, LightEngine, PlanState
from .policy import ActionCandidate, PlayerIntent, PolicyInput


@dataclass(frozen=True)
class SearchState:
    """An owned snapshot of a mutable ``LightEngine`` state.

    ``PlanState`` stays mutable because that is the established LightEngine
    representation. Every SearchState constructor takes a deep copy, so its
    nested players, ball, events, and restart positions never alias another
    branch or the caller's state.
    """

    plan: PlanState
    time_remaining: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan", copy.deepcopy(self.plan))
        object.__setattr__(self, "time_remaining", max(0.0, float(self.time_remaining)))

    @classmethod
    def from_policy_input(cls, inp: PolicyInput) -> SearchState:
        """Copy the observable physical state into branch-owned LightEngine state.

        Babylon observations expose positions, velocities, facing, possession,
        score, and ``canAct``. They do not expose cooldown/grounded timers,
        possession-protection time, or goalkeeper hold time; those unavailable
        LightEngine fields therefore use their documented initial defaults.
        """
        source = inp.state
        players = [
            LPlayer(
                team=player.team,
                pid=player.id,
                role=player.role,
                x=player.x,
                y=player.y,
                facing=player.facing,
                can_act=player.can_act,
                vx=player.vx,
                vy=player.vy,
            )
            for player in (*source.us, *source.them)
        ]
        ball = LBall(
            x=source.ball.x,
            y=source.ball.y,
            vx=source.ball.vx,
            vy=source.ball.vy,
            possessing_team=source.ball.possessing_team,
            possessing_player=source.ball.possessing_player,
        )
        plan = PlanState(
            score_us=inp.su,
            score_them=inp.st_th,
            ball=ball,
            players=players,
            base_positions=[
                (player.team, player.id, player.role, player.x, player.y)
                for player in (*source.us, *source.them)
            ],
        )
        return cls(plan=plan, time_remaining=inp.tr)

    def clone(self) -> SearchState:
        """Create a fully independent sibling branch."""
        return SearchState(self.plan, self.time_remaining)


def _light_intent(intent: PlayerIntent) -> LIntent:
    return LIntent(
        tx=intent.tx,
        ty=intent.ty,
        speed=intent.speed,
        act=intent.action_type,
        action_target=intent.action_target,
        power=intent.action_power,
        face_target=(intent.face_x, intent.face_y),
    )


def _validate_intent_coordinates(intent: PlayerIntent) -> None:
    values = [intent.tx, intent.ty, intent.speed, intent.face_x, intent.face_y]
    if intent.action_target is not None:
        values.extend(intent.action_target)
    if intent.action_power is not None:
        values.append(intent.action_power)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("intent must contain only finite values")
    if not (0.0 <= intent.tx <= PITCH_LENGTH and 0.0 <= intent.ty <= PITCH_WIDTH):
        raise ValueError("intent movement target must be within the pitch")


def _validate_attacking_candidate(state: PlanState, candidate: ActionCandidate) -> None:
    intent = candidate.intent
    if intent.action_type not in {"none", "pass", "shoot"}:
        raise ValueError(f"unsupported attacking action: {intent.action_type!r}")

    if state.ball.possessing_team != "us" or state.ball.possessing_player != intent.pid:
        raise ValueError("candidate player must possess the ball for the us team")

    player = state.player("us", intent.pid)
    if player is None:
        raise ValueError(f"candidate player {intent.pid!r} is absent from search state")
    if intent.action_type in {"pass", "shoot"}:
        if not player.can_act:
            raise ValueError("a player who cannot act cannot pass or shoot")
        if intent.action_target is None:
            raise ValueError(f"{intent.action_type} candidate requires an action target")

    _validate_intent_coordinates(intent)


def transition_candidate(
    state: SearchState,
    candidate: ActionCandidate,
    *,
    background_intents: Mapping[tuple[str, str], PlayerIntent] | None = None,
) -> SearchState:
    """Apply one production candidate and advance one LightEngine decision.

    The candidate's ``PlayerIntent`` is translated field-for-field to the
    simulator intent; its ``reason`` and EV are metadata and do not affect
    physics. Optional intents let callers retain other players' chosen
    movement/actions without calling or mutating the production controller.
    Unspecified players remain stationary for this one step.
    """
    _validate_attacking_candidate(state.plan, candidate)

    intents = dict(background_intents or {})
    chosen = candidate.intent
    intents[("us", chosen.pid)] = chosen

    return advance_search_state(state, intents)


def _convert_intents(
    intents: Mapping[tuple[str, str], PlayerIntent] | None,
) -> dict[tuple[str, str], LIntent]:
    converted: dict[tuple[str, str], LIntent] = {}
    for key, intent in (intents or {}).items():
        team, pid = key
        if team not in {"us", "them"} or pid != intent.pid:
            raise ValueError("background intent key must match a valid team and PlayerIntent.pid")
        _validate_intent_coordinates(intent)
        converted[key] = _light_intent(intent)
    return converted


def advance_search_state(
    state: SearchState,
    intents: Mapping[tuple[str, str], PlayerIntent] | None = None,
) -> SearchState:
    """Advance a branch one LightEngine decision, including a loose-ball step.

    This is useful after a pass or shot has released the ball: no attacking
    ActionCandidate is required while the branch advances toward its next
    controllable state. Supplied intents are copied and translated without
    mutating the caller's mapping or objects.
    """
    light_intents = _convert_intents(intents)
    # LightEngine is deterministic today; a fresh seeded instance also keeps
    # any future simulator-local mutable state out of sibling branches.
    simulator = LightEngine(seed=0)
    next_plan = simulator.step(state.plan, light_intents)
    return SearchState(
        plan=next_plan,
        time_remaining=max(0.0, state.time_remaining - simulator.dt),
    )


def is_within_pitch(state: SearchState) -> bool:
    """Check finite player/ball coordinates against the simulator pitch bounds."""
    positions = [(state.plan.ball.x, state.plan.ball.y)]
    positions.extend((player.x, player.y) for player in state.plan.players)
    return all(
        math.isfinite(x)
        and math.isfinite(y)
        and 0.0 <= x <= PITCH_LENGTH
        and 0.0 <= y <= PITCH_WIDTH
        for x, y in positions
    )
