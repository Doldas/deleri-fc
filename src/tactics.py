"""Tactics — tactical state machine, roles and coordinated pressing.

Implements the high-level states from AISTRATEGI §20 and the trigger-based,
coordinated pressing rules from §23–25. State detection is deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from itertools import permutations

from . import geom
from .config import BALL_CONTROL_RADIUS, TACKLE_MAX
from .state import GameState, Player, WorldModel

ROLE_DEFENDER = "DEFENDER"
ROLE_WIDE_LEFT = "WIDE_LEFT"
ROLE_WIDE_RIGHT = "WIDE_RIGHT"
ROLE_STRIKER = "STRIKER"
ROLES_TACTICAL = (ROLE_DEFENDER, ROLE_WIDE_LEFT, ROLE_WIDE_RIGHT, ROLE_STRIKER)


class TacticalState(StrEnum):
    KICKOFF = "KICKOFF"
    BUILD_UP = "BUILD_UP"
    PROGRESSION = "PROGRESSION"
    ATTACK = "ATTACK"
    FINAL_ATTACK = "FINAL_ATTACK"
    DEFENSIVE_TRANSITION = "DEFENSIVE_TRANSITION"
    COUNTERPRESS = "COUNTERPRESS"
    MID_BLOCK = "MID_BLOCK"
    HIGH_PRESS = "HIGH_PRESS"
    LOW_BLOCK = "LOW_BLOCK"
    POSSESSION_CONTROL = "POSSESSION_CONTROL"
    GAME_MANAGEMENT = "GAME_MANAGEMENT"


ROLE_BY_SLOT: dict[str, str] = {
    "defender": ROLE_DEFENDER,
    "left": ROLE_WIDE_LEFT,
    "right": ROLE_WIDE_RIGHT,
    "striker": ROLE_STRIKER,
}


def assign_roles(state: GameState, slots: list[dict]) -> dict[str, str]:
    """Stable per-player tactical role assignment based on the builder slots.

    Matches the closest permutation of slots to the current positions; ties are
    broken by canonical ordering so the result is deterministic.
    """
    players = state.outfield_us()
    if not players:
        return {}
    best = None
    best_key = None
    best_sum = math.inf
    for order in permutations(slots):
        total = sum(
            geom.distance(player.x, player.y, slot["position"]["x"], slot["position"]["y"])
            for player, slot in zip(players, order)
        )
        key = tuple(sorted(slot["id"] + "|" + player.id for player, slot in zip(players, order)))
        if total < best_sum - 1e-9 or (abs(total - best_sum) < 1e-9 and (best_key is None or key < best_key)):
            best_sum = total
            best_key = key
            best = tuple((slot["id"], player.id) for player, slot in zip(players, order))
    roles: dict[str, str] = {}
    for slot_id, player_id in best or ():
        roles[player_id] = ROLE_BY_SLOT.get(str(slot_id), ROLE_DEFENDER)
    return roles


def detect(state: GameState, world: WorldModel, config: dict[str, float], prev_had_control: bool) -> TacticalState:
    """High-level tactical state for this tick."""
    phase = state.phase
    if phase != "openPlay" and phase != "goldenGoal":
        return TacticalState.KICKOFF

    time = state.time_remaining
    goal_diff = state.score_us - state.score_them

    if state.has_control():
        zone = world.ball_zone
        if zone == "final":
            base = TacticalState.FINAL_ATTACK
        elif zone == "mid":
            base = TacticalState.PROGRESSION
        else:
            base = TacticalState.BUILD_UP
        # Late-game management when leading.
        if time < 30.0 and goal_diff > 0 and zone in ("own", "mid"):
            return TacticalState.GAME_MANAGEMENT
        # Sustained possession deep in opponent half reads as control.
        if zone == "final" and world.ball_side != "center":
            return TacticalState.POSSESSION_CONTROL
        return base

    if prev_had_control:
        # We just lost the ball: decide counterpress vs recover.
        return TacticalState.COUNTERPRESS if _recovery_likely(state) else TacticalState.DEFENSIVE_TRANSITION

    # They possess.
    zone = world.ball_zone
    pressure = world.pressure_on_ball
    press = config.get("press_intensity", 0.5)
    if world.ball_near_wall and zone == "final" and pressure < 0.3:
        # We can trap them by the wall — high press window.
        return TacticalState.HIGH_PRESS
    if zone == "own":
        return TacticalState.MID_BLOCK if press > 0.3 else TacticalState.LOW_BLOCK
    if zone == "mid":
        if press >= 0.5 * press + 0.25:
            return TacticalState.HIGH_PRESS
        return TacticalState.MID_BLOCK
    # Ball in their final third (deep for them) — defensive block.
    return TacticalState.HIGH_PRESS if press >= 0.5 * press + 0.3 else TacticalState.MID_BLOCK


def _recovery_likely(state: GameState) -> bool:
    """Heuristic: is the ball close enough that a counterpress has a real
    chance to recover it before the opponent can escape?"""
    players = state.outfield_us()
    if not players:
        return False
    nearest = min(players, key=lambda p: geom.distance(p.x, p.y, state.ball.x, state.ball.y))
    return geom.distance(nearest.x, nearest.y, state.ball.x, state.ball.y) <= BALL_CONTROL_RADIUS + TACKLE_MAX + 1.0


@dataclass
class PressPlan:
    pressers: list[str] = field(default_factory=list)
    second_pressers: list[str] = field(default_factory=list)
    cover: list[str] = field(default_factory=list)
    rest_defence: list[str] = field(default_factory=list)
    trigger_note: str = ""

    def role_for(self, player_id: str) -> str:
        if player_id in self.pressers:
            return "PRESS"
        if player_id in self.second_pressers:
            return "PRESS_SUPPORT"
        if player_id in self.cover:
            return "COVER"
        return "REST_DEFENCE"


def _trigger_note(state: GameState, carrier: Player) -> str:
    """Triggers are evaluated on the OPPONENT ball carrier, not our presser."""
    ball = state.ball
    if carrier is None:
        return ""
    note = ""
    if carrier.facing > math.pi / 2.0:
        note += "facing_own_goal "
    if is_trap_near_wall(carrier):
        note += "near_wall "
    if de_isolated(state, carrier):
        note += "isolated "
    if geom.distance(carrier.x, carrier.y, ball.x, ball.y) <= BALL_CONTROL_RADIUS + 1.0:
        note += "ball_close "
    return note.strip()


def is_trap_near_wall(p: Player) -> bool:
    return p.y <= 3.0 or p.y >= 37.0 or p.x <= 3.0 or p.x >= 57.0


def de_isolated(state: GameState, p: Player) -> bool:
    mates = [q for q in state.outfield_them() if q.id != p.id]
    if not mates:
        return True
    nearest = min(geom.distance(p.x, p.y, q.x, q.y) for q in mates)
    return nearest > 6.0


def plan_press(state: GameState, world: WorldModel, config: dict[str, float], roles: dict[str, str]) -> PressPlan:
    """Coordinated press plan. Never lets all outfields chase the ball.

    Up to two pressers are chosen on *triggers about the opponent carrier*; the
    rest cover (goal-side lane protection) or hold rest defence. The player
    closest to the ball still presses once the opponent is inside our half, so
    the defence never stands still.
    """
    ours = state.outfield_us()
    if not ours:
        return PressPlan()
    intensity = config.get("press_intensity", 0.55)
    trigger_thresh = config.get("press_trigger_threshold", 0.4)
    carrier = state.their_possessor()
    ball = state.ball
    trigger_ok_base = _trigger_note(state, carrier) if carrier is not None else ""

    by_dist = sorted(ours, key=lambda p: geom.distance(p.x, p.y, ball.x, ball.y))
    max_pressers = 2 if intensity >= 0.35 else 1

    pressers: list[str] = []
    second: list[str] = []
    for p in by_dist:
        if len(pressers) + len(second) >= max_pressers:
            break
        d = geom.distance(p.x, p.y, ball.x, ball.y)
        # ball_priority: we are physically on the ball; press regardless.
        ball_priority = d <= BALL_CONTROL_RADIUS + 1.5
        # carrier_trigger: opponent is trapped / turned / isolated OR deep in
        # our third and we are close enough to make a race for it worthwhile.
        carrier_trigger = bool(trigger_ok_base) or (ball.x < 24.0 and d <= 7.0)
        if ball_priority or (carrier_trigger and _press_score(p, config) >= trigger_thresh):
            (second if pressers else pressers).append(p.id)
        elif not pressers and d < 9.0 and ball.x < 32.0:
            # Fallback: closest player steps up to give the carrier no free run.
            pressers.append(p.id)

    # Cover: players between ball and our goal, goal-side.
    cover: list[str] = []
    rest: list[str] = []
    for p in ours:
        if p.id in pressers or p.id in second:
            continue
        if p.x < ball.x and not (p.role == "goalkeeper"):
            cover.append(p.id)
        else:
            rest.append(p.id)
    # Never leave the last defender roaming: keep at least one cover.
    if not cover and rest and len(rest) >= 1 and len(pressers) + len(second) < 3:
        move = rest[-1]
        rest.remove(move)
        cover.append(move)
    return PressPlan(pressers, second, cover, rest)


def _press_score(p: Player, config: dict[str, float]) -> float:
    # Balance between distance to ball and role value.
    base = config.get("press_intensity", 0.55)
    if p.role == "goalkeeper":
        return 0.0
    return base