"""Reusable mirror-symmetry helpers for the canonical Babylon frame.

The canonical frame (see ``docs/00-game-engine-rules.md``) is::

    x = 0.0   -> our own goal line
    x = 60.0  -> the opponent goal line (we always attack +x)
    y = 0.0   -> one touchline
    y = 40.0  -> the opposite touchline
    y = 20.0  -> the pitch centre line (and the goal centres, GOAL_CENTER_Y)

A lateral mirror is the reflection ``R`` across the centre line ``y = 20``::

    point p = (x, y)      ->  (x, 40 - y)
    velocity v = (vx, vy) ->  (vx, -vy)
    facing theta           ->  -theta          (mod 2*pi)

Facing deserves a note, because it is the one transform that is easy to get
wrong.  Reflecting a direction across a *horizontal* line negates the y
component: ``(cos t, sin t) -> (cos t, -sin t)``, whose angle is ``-t``.  The
transformation ``theta -> pi - theta`` is the reflection across a *vertical*
line, ``(cos t, sin t) -> (-cos t, sin t)``.  Using it here would swap a
carrier who faces the opponent goal for one who faces their own goal, which
manufactures symmetry failures that do not exist in the code.  So the mirror
facing is ``-theta``.

Any deterministic decision function built only from distances, dot products and
cross products of mirrored entities is automatically mirror-equivariant; the
helpers here exist so the tests can express that invariant once, correctly.
"""

from __future__ import annotations

import dataclasses
import math

from src.geom import (
    GOAL_CENTER_Y,
    PITCH_LENGTH,
    PITCH_WIDTH,
)
from src.state import Ball, GameState, Player

# Mirror mappings on the lateral axis.
MIRROR_ROLES: dict[str, str] = {
    "WIDE_LEFT": "WIDE_RIGHT",
    "WIDE_RIGHT": "WIDE_LEFT",
    "DEFENDER": "DEFENDER",
    "STRIKER": "STRIKER",
}

MIRROR_SIDES: dict[str, str] = {
    "left": "right",
    "right": "left",
    "center": "center",
    # Wall touchlines: `top`/`bottom` are the lateral pair, just as
    # `left`/`right` are the goal-line pair.
    "top": "bottom",
    "bottom": "top",
}

# Mirroring a *labelled* formation reflects the geometry and swaps the lateral
# slot labels, because "left" and "right" name opposite touchlines.
MIRROR_SLOT_IDS: dict[str, str] = {
    "left": "right",
    "right": "left",
    "left_wide": "right_wide",
    "right_wide": "left_wide",
    "WIDE_LEFT": "WIDE_RIGHT",
    "WIDE_RIGHT": "WIDE_LEFT",
}


def mirror_y(y: float) -> float:
    """Reflect a y coordinate across the pitch centre line."""
    return PITCH_WIDTH - y


def mirror_angle(theta: float) -> float:
    """Reflect a heading across the pitch centre line (``theta -> -theta``)."""
    return -theta


def mirror_side(side: str) -> str:
    """Map a lateral label onto its mirrored counterpart."""
    return MIRROR_SIDES.get(side, side)


def mirror_role(role: str) -> str:
    return MIRROR_ROLES.get(role, role)


def mirror_player(p: Player, team: str) -> Player:
    return Player(
        id=p.id,
        team=team,
        role=p.role,
        x=p.x,
        y=mirror_y(p.y),
        vx=p.vx,
        vy=-p.vy,
        facing=mirror_angle(p.facing),
        can_act=p.can_act,
    )


def mirror_ball(b: Ball) -> Ball:
    return Ball(
        x=b.x,
        y=mirror_y(b.y),
        vx=b.vx,
        vy=-b.vy,
        possessing_team=b.possessing_team,
        possessing_player=b.possessing_player,
    )


def mirror_state(s: GameState) -> GameState:
    """Return the lateral mirror of a whole state.

    X, identities, roles, possession and the clock are all preserved: the mirror
    acts only on the lateral axis, which is exactly the axis the pitch is
    symmetric about.
    """
    return GameState(
        protocol_version=s.protocol_version,
        game_id=s.game_id,
        sequence=s.sequence,
        simulation_tick=s.simulation_tick,
        apply_at_tick=s.apply_at_tick,
        time_remaining=s.time_remaining,
        phase=s.phase,
        score_us=s.score_us,
        score_them=s.score_them,
        ball=mirror_ball(s.ball),
        us=tuple(mirror_player(p, "us") for p in s.us),
        them=tuple(mirror_player(p, "them") for p in s.them),
    )


def mirror_slots(slots: list[dict]) -> list[dict]:
    """Mirror a formation slot list: reflect the geometry and swap the
    lateral slot labels, keeping the list order intact.

    ``tactics.json`` ships a laterally symmetric diamond (defender and striker
    on the centre line, the two wingers at y=8 and y=32), so reflecting the
    geometry reproduces the same set of positions with ``left``/``right``
    exchanged.  The list order is preserved so code that indexes slots
    positionally keeps behaving identically.
    """
    out = []
    for slot in slots:
        pos = slot["position"]
        mirrored = dict(slot)
        mirrored["position"] = {"x": float(pos["x"]), "y": mirror_y(float(pos["y"]))}
        mirrored["id"] = MIRROR_SLOT_IDS.get(str(slot["id"]), slot["id"])
        out.append(mirrored)
    return out


# --------------------------------------------------------------------------- #
# comparison helpers
# --------------------------------------------------------------------------- #


def y_mirrored(y: float, tol: float = 1e-6) -> bool:
    return abs(mirror_y(y) - y) <= tol


def assert_y_mirror(
    y: float, y_mirror: float, tol: float = 1e-6, msg: str = ""
) -> None:
    """Assert ``y_mirror == PITCH_WIDTH - y``."""
    if abs(y + y_mirror - PITCH_WIDTH) > tol:
        raise AssertionError(
            f"{msg}y is not laterally mirrored: y={y!r}, mirrored={y_mirror!r} "
            f"(expected {PITCH_WIDTH - y!r})"
        )


def assert_close(a: float, b: float, tol: float = 1e-6, msg: str = "") -> None:
    if not (abs(a - b) <= tol):
        raise AssertionError(f"{msg}{a!r} != {b!r} (tol {tol})")


def intent_is_mirrored(a, b, tol: float = 1e-6) -> bool:
    """True when two PlayerIntents are the same decision, mirrored.

    Every scalar survives the reflection unchanged (``x``, speeds, power) and
    every lateral component satisfies ``y + y_mirror == PITCH_WIDTH``.
    """
    if a is None or b is None:
        return a is None and b is None
    if a.action_type != b.action_type:
        return False
    if abs(a.speed - b.speed) > tol:
        return False
    if (a.action_power is None) != (b.action_power is None):
        return False
    if a.action_power is not None and abs(a.action_power - b.action_power) > tol:
        return False
    if (a.action_target is None) != (b.action_target is None):
        return False
    if a.action_target is not None and b.action_target is not None:
        if abs(a.action_target[0] - b.action_target[0]) > tol:
            return False
        if abs(a.action_target[1] + b.action_target[1] - PITCH_WIDTH) > tol:
            return False
    if abs(a.tx - b.tx) > tol:
        return False
    if abs(a.ty + b.ty - PITCH_WIDTH) > tol:
        return False
    if abs(a.face_x - b.face_x) > tol:
        return False
    if abs(a.face_y + b.face_y - PITCH_WIDTH) > tol:
        return False
    return True


def describe_mismatch(a, b) -> str:
    if a is None or b is None:
        return f"intent presence differs: {a!r} vs {b!r}"
    parts: list[str] = []
    if a.action_type != b.action_type:
        parts.append(f"action {a.action_type!r} vs {b.action_type!r}")
    if abs(a.tx - b.tx) > 1e-6:
        parts.append(f"tx {a.tx:.3f} vs {b.tx:.3f}")
    if abs(a.ty + b.ty - PITCH_WIDTH) > 1e-6:
        parts.append(f"ty {a.ty:.3f} vs {b.ty:.3f} (mirror {PITCH_WIDTH - a.ty:.3f})")
    if abs(a.speed - b.speed) > 1e-6:
        parts.append(f"speed {a.speed:.3f} vs {b.speed:.3f}")
    if a.action_power is not None and b.action_power is not None:
        if abs(a.action_power - b.action_power) > 1e-6:
            parts.append(f"power {a.action_power:.3f} vs {b.action_power:.3f}")
    if abs(a.face_y + b.face_y - PITCH_WIDTH) > 1e-6:
        parts.append(f"face_y {a.face_y:.3f} vs {b.face_y:.3f}")
    return "; ".join(parts) or "intents equal but mirror check failed"


def angle_delta(a: float, b: float) -> float:
    """Smallest absolute difference between two headings, in radians."""
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)

def facing_is_mirrored(a: float, b: float, tol: float = 1e-6) -> bool:
    """True when heading ``b`` is the lateral mirror of heading ``a``."""
    return angle_delta(b, mirror_angle(a)) <= tol


# --------------------------------------------------------------------------- #
# seeded situation builder
# --------------------------------------------------------------------------- #
#
# The mirror invariant is only worth anything if it is checked over a spread of
# pictures rather than one hand-picked fixture, so the tests below sweep seeded
# random situations. Everything is derived from an explicit ``random.Random``, so
# a failure is always reproducible from the seed the test prints.

_SITUATIONS = ("possession", "defending", "loose")


def _mk_player(pid: str, team: str, role: str, x: float, y: float, rng) -> Player:
    return Player(
        id=pid,
        team=team,
        role=role,
        x=x,
        y=y,
        vx=rng.uniform(-4.0, 4.0),
        vy=rng.uniform(-3.0, 3.0),
        facing=rng.uniform(-math.pi, math.pi),
        can_act=True,
    )


def build_random_state(rng, situation: str) -> GameState:
    """Build one deterministic situation of the requested kind.

    ``possession`` puts the ball with one of our outfielders, ``defending`` with
    an opponent, and ``loose`` with nobody, so the sweep exercises the carrier,
    the press plan and the loose-ball meeting point. Positions deliberately
    include the centre line, the goal mouths and both touchlines, because every
    tie-break bug this module guards lived on one of those boundaries.
    """
    ours = [
        _mk_player("us0", "us", "goalkeeper", rng.uniform(1.0, 9.0),
                   GOAL_CENTER_Y + rng.uniform(-3.0, 3.0), rng),
    ]
    for i in range(1, 5):
        ours.append(_mk_player(f"us{i}", "us", "outfield", rng.uniform(6.0, 50.0),
                               rng.uniform(2.0, 38.0), rng))
    theirs = [
        _mk_player("tgk", "them", "goalkeeper", PITCH_LENGTH - rng.uniform(1.0, 9.0),
                   GOAL_CENTER_Y + rng.uniform(-3.0, 3.0), rng),
    ]
    for i in range(1, 5):
        theirs.append(_mk_player(f"t{i}", "them", "outfield", rng.uniform(10.0, 54.0),
                                 rng.uniform(2.0, 38.0), rng))
    # Snap some players onto the mirror's fixed points so exact ties are hit:
    # y = 20 is its own mirror image, so a decision that cannot name one of two
    # equally good flanks has to resolve here rather than quietly pick a side.
    if rng.random() < 0.5:
        ours[2] = dataclasses.replace(ours[2], y=GOAL_CENTER_Y)
    if rng.random() < 0.5:
        theirs[2] = dataclasses.replace(theirs[2], y=GOAL_CENTER_Y)

    if situation == "possession":
        carrier = rng.choice(ours[1:])
        ball = Ball(carrier.x, carrier.y, carrier.vx, carrier.vy, "us", carrier.id)
    elif situation == "defending":
        carrier = rng.choice(theirs[1:])
        ball = Ball(carrier.x, carrier.y, carrier.vx, carrier.vy, "them", carrier.id)
    else:
        ball = Ball(rng.uniform(5.0, 55.0), rng.uniform(2.0, 38.0),
                    rng.uniform(-6.0, 6.0), rng.uniform(-4.0, 4.0), None, None)

    return GameState(
        protocol_version="1",
        game_id="mirror-test",
        sequence=1,
        simulation_tick=0,
        apply_at_tick=0,
        time_remaining=90.0,
        phase="play",
        score_us=0,
        score_them=0,
        ball=ball,
        us=tuple(ours),
        them=tuple(theirs),
    )


def random_state_mismatches(situation: str, count: int, seed: int, probe) -> list[str]:
    """Run ``probe(state, mirrored_state)`` over ``count`` seeded situations.

    Returns a list of human-readable failure descriptions, empty when the probe
    accepted every picture. The seed is part of every message so a failure can
    be replayed on its own.
    """
    import random

    rng = random.Random(seed)
    out: list[str] = []
    for i in range(count):
        st = build_random_state(rng, situation)
        try:
            ok = probe(st, mirror_state(st))
            note = ""
        except Exception as exc:  # noqa: BLE001 - a crash is a failure too
            ok, note = False, f"{type(exc).__name__}: {exc}"
        if not ok:
            out.append(f"{situation}#{i} (seed={seed}) {note}".rstrip())
    return out


__all__ = [
    "GOAL_CENTER_Y",
    "MIRROR_ROLES",
    "MIRROR_SIDES",
    "PITCH_LENGTH",
    "PITCH_WIDTH",
    "angle_delta",
    "assert_close",
    "assert_y_mirror",
    "build_random_state",
    "describe_mismatch",
    "facing_is_mirrored",
    "intent_is_mirrored",
    "mirror_angle",
    "mirror_ball",
    "mirror_player",
    "mirror_role",
    "mirror_side",
    "mirror_slots",
    "mirror_state",
    "mirror_y",
    "random_state_mismatches",
    "y_mirrored",
]