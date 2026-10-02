"""Small deterministic defensive responses for isolated MCTS branches.

The response layer reuses the simulator's PressBot for its nearest-player
pressure/tackle primitive. A single non-presser shape target and physics-based
loose-ball pursuit keep the remaining defenders moving without importing the
production attack/defence policy or retaining any controller state.
"""

from __future__ import annotations

import random

from . import geom
from .geom import PITCH_LENGTH
from .physics import pass_collection_point
from .policy import ActionCandidate, PlayerIntent, loose_ball_meeting_point
from .search_state import SearchState

OPPONENT_RESPONSE_MODEL = "deterministic_press_shape_v1"


def _intent_from_simulator(
    pid: str,
    tx: float,
    ty: float,
    speed: float,
    *,
    act: str = "none",
    action_target: tuple[float, float] | None = None,
    power: float | None = None,
    face: tuple[float, float],
) -> PlayerIntent:
    tx, ty = geom.clamp_point(tx, ty, margin=0.5)
    return PlayerIntent(
        pid=pid,
        tx=tx,
        ty=ty,
        speed=speed,
        face_x=face[0],
        face_y=face[1],
        action_type=act,
        action_target=action_target,
        action_power=power,
    )


def _loose_meeting_point(state: SearchState) -> tuple[float, float]:
    ball = state.plan.ball
    x, y, _seconds = loose_ball_meeting_point(ball.x, ball.y, ball.vx, ball.vy)
    return geom.clamp_point(x, y, margin=0.5)


def _shot_lane_point(
    state: SearchState,
    candidate: ActionCandidate,
    defenders,
) -> tuple[float, float, str] | None:
    """Pick the closest outfielder to the shot segment and its lane point."""
    target = candidate.intent.action_target
    if target is None or not defenders:
        return None
    ball = state.plan.ball
    ax, ay = ball.x, ball.y
    bx, by = target
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return None

    ranked: list[tuple[float, str, float, float]] = []
    for player in defenders:
        t = geom.clamp(
            ((player.x - ax) * dx + (player.y - ay) * dy) / length_sq,
            0.0,
            1.0,
        )
        px, py = ax + t * dx, ay + t * dy
        ranked.append((geom.distance_sq(player.x, player.y, px, py), player.pid, px, py))
    distance_sq, pid, px, py = min(ranked, key=lambda row: (row[0], row[1]))
    return px, py, pid


def _response_point(
    state: SearchState,
    candidate: ActionCandidate | None,
    defenders,
    attacking_team: str,
) -> tuple[float, float, str | None, bool]:
    """Return target, presser id, and whether this is a predicted kick response."""
    plan = state.plan
    ball = plan.ball
    attacker = (
        plan.player(attacking_team, ball.possessing_player or "")
        if ball.possessing_team == attacking_team
        else None
    )

    if candidate is not None and candidate.intent.action_type == "pass" and attacker is not None:
        intent = candidate.intent
        target = intent.collection_point
        if target is None and intent.action_target is not None:
            power = 0.5 if intent.action_power is None else intent.action_power
            cx, cy, _power, _speed = pass_collection_point(
                attacker.x,
                attacker.y,
                intent.action_target[0],
                intent.action_target[1],
                power,
            )
            target = (cx, cy)
        if target is not None:
            point = geom.clamp_point(*target, margin=0.5)
            presser = min(
                defenders,
                key=lambda player: (geom.distance_sq(player.x, player.y, *point), player.pid),
            )
            return point[0], point[1], presser.pid, True

    if candidate is not None and candidate.intent.action_type == "shoot":
        lane = _shot_lane_point(state, candidate, defenders)
        if lane is not None:
            return lane[0], lane[1], lane[2], True

    if ball.possessing_team is None:
        point = _loose_meeting_point(state)
        presser = min(
            defenders,
            key=lambda player: (geom.distance_sq(player.x, player.y, *point), player.pid),
        )
        return point[0], point[1], presser.pid, False

    if attacker is not None:
        # The simulated PressBot itself ranks its presser from the ball, whose
        # controlled offset is small; use the carrier position with a stable id
        # tie-break so pressure follows the actual player.
        presser = min(
            defenders,
            key=lambda player: (
                geom.distance_sq(player.x, player.y, attacker.x, attacker.y),
                player.pid,
            ),
        )
        return attacker.x, attacker.y, presser.pid, False

    # The defending side has possession. This tiny model does not invent an
    # attacking policy; retain the simulator PressBot's deterministic movement.
    return ball.x, ball.y, None, False


def _shape_target(
    state: SearchState,
    player,
    anchor: tuple[float, float],
    defending_team: str,
) -> tuple[float, float]:
    """Recover toward a compact goal-side line while retaining the base lane."""
    own_goal_x = 0.0 if defending_team == "us" else PITCH_LENGTH
    anchor_x, anchor_y = anchor
    toward_goal = 1.0 if own_goal_x > anchor_x else -1.0
    goal_distance = abs(own_goal_x - anchor_x)
    depth = min(9.0, goal_distance * 0.35)
    line_x = anchor_x + toward_goal * depth

    base = next(
        (
            (x, y)
            for team, pid, _role, x, y in state.plan.base_positions
            if team == defending_team and pid == player.pid
        ),
        (player.x, player.y),
    )
    # Preserve some formation width/depth, while sliding the block toward the
    # carrier or the predicted second-ball point.
    tx = 0.75 * line_x + 0.25 * base[0]
    ty = 0.65 * base[1] + 0.35 * anchor_y
    return geom.clamp_point(tx, ty, margin=0.5)


def branch_opponent_intents(
    state: SearchState,
    candidate: ActionCandidate | None = None,
    *,
    defending_team: str = "them",
) -> dict[tuple[str, str], PlayerIntent]:
    """Generate branch-local defensive intents; all movement is applied by LightEngine.

    Only outfielders are given intents. LightEngine remains the sole owner of
    goalkeeper handling/catches, and no mutable opponent controller, production
    opponent model, TeamPlan, or live policy state is retained or modified.
    """
    if defending_team not in {"us", "them"}:
        raise ValueError("defending_team must be 'us' or 'them'")
    plan = state.plan
    attacking_team = "them" if defending_team == "us" else "us"
    defenders = sorted(
        (player for player in plan.outfield(defending_team) if player.grounded <= 0.0),
        key=lambda player: player.pid,
    )
    if not defenders:
        return {}

    # PressBot is a deterministic simulator primitive: it has no retained
    # tactical state and does not draw from rng in decide(). A fresh fixed-seed
    # instance makes that dependency branch-local even if it later gains state.
    from .sim import PressBot

    simulated = PressBot(random.Random(0)).decide(plan, defending_team)
    point_x, point_y, presser_id, predicted_kick = _response_point(
        state, candidate, defenders, attacking_team
    )

    # If the response side has possession, this intentionally remains only a
    # generic deterministic movement response; it is not a second attack policy.
    defending_has_ball = plan.ball.possessing_team == defending_team
    intents: dict[tuple[str, str], PlayerIntent] = {}
    for player in defenders:
        face = (plan.ball.x, plan.ball.y)
        if defending_has_ball:
            base = simulated.get((defending_team, player.pid))
            if base is None:
                continue
            intents[(defending_team, player.pid)] = _intent_from_simulator(
                player.pid,
                base.tx,
                base.ty,
                base.speed,
                act=base.act,
                action_target=base.action_target,
                power=base.power,
                face=face,
            )
            continue

        if player.pid == presser_id:
            base = simulated.get((defending_team, player.pid))
            act = "none"
            action_target = None
            power = None
            if not predicted_kick and base is not None:
                # LightEngine decides whether protection, cooldown, distance,
                # and tackle semantics allow this attempt to take effect.
                act = base.act
                action_target = base.action_target
                power = base.power
            intents[(defending_team, player.pid)] = _intent_from_simulator(
                player.pid,
                point_x,
                point_y,
                1.0,
                act=act,
                action_target=action_target,
                power=power,
                face=face,
            )
        else:
            tx, ty = _shape_target(state, player, (point_x, point_y), defending_team)
            intents[(defending_team, player.pid)] = _intent_from_simulator(
                player.pid, tx, ty, 0.7, face=face
            )
    return intents


__all__ = ["OPPONENT_RESPONSE_MODEL", "branch_opponent_intents"]
