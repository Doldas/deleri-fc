#!/usr/bin/env python3
"""Why do we lose to Vanguard FC (elite)?

`run_authoritative.py` reports the scoreboard; this reads the engine's own
replay and explains it. Every question here is one the scoreboard cannot answer:

  * who is actually taking our shots (role, distance, body side),
  * what power our passes are actually kicked with,
  * where possession actually goes when we lose it,
  * how far forward we get the ball versus how far they get it,
  * which outfielder wins the loose ball, and how often,
  * whether the goalkeeper is on his line or stranded,
  * how our outfield shape compares with theirs, measured not assumed.

Both teams are normalised into our attacking frame: our own goal is x=0 and we
attack towards +x, so `team-b` positions (and the ball when *they* hold it in
their frame) are mirrored with `60 - x`. The ball is a single absolute object,
so it is mirrored by *holder* side, not by a fixed team. Getting that wrong
flips the analysis for every `--sides both` replay.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

PITCH_L = 60.0
PITCH_W = 40.0
GOAL_CY = 20.0
GOAL_HW = 3.0
# The engine assigns literal "team-a"/"team-b" ids by *side*, not by identity:
# in a replay where we are the away side, `teams[0]` is the opponent. Only the
# top-level `teamId` is stable, so our side is resolved per replay by name.
US_NAME_HINT = "Deleri FC"


def _mx(x: float, mirror_us: bool) -> float:
    """Raw replay x -> our attacking frame (our goal 0, we attack +x).

    The replay is stored in the engine's own absolute frame, which puts *our*
    goal at raw x=0 when we are home and at raw x=60 when we are away. So the
    side that needs flipping depends on `fixture.side`, not on team id: mirror
    the opponent at home, mirror ourselves away. Getting this backwards turns
    every distance into nonsense (shots "from 55 m", 8 m-wide shapes).
    """
    return PITCH_L - x if mirror_us else x

SHOT_GOAL_X = 60.0

# A controlled ball sits 0.65 m in front of its player (RULES.md "Movement,
# facing, and dribbling"), and BALL_CONTROL_RADIUS is 0.9 m, so the holder is
# always the closest player within about 1.6 m of the ball.
HOLD_RADIUS = 1.6


def _my(y: float, mirror_us: bool) -> float:
    """Raw replay y -> our attacking frame (our left touchline is y=0)."""
    return PITCH_W - y if mirror_us else y


def _nearest_to_ball(ta, tb, us_id, bx, by):
    """Which team (and player) is holding the ball, by proximity.

    Returns (team_id | None, player_id | None). None team means the ball is
    genuinely loose at this tick.
    """
    best = None
    for team in (ta, tb):
        for q in team.get("players", []):
            pos = q.get("position") or {}
            d = math.hypot(float(pos.get("x", 0.0)) - bx, float(pos.get("y", 0.0)) - by)
            if d <= HOLD_RADIUS and (best is None or d < best[0]):
                best = (d, team.get("id"), q.get("id"))
    if best is None:
        return None, None
    return best[1], best[2]


def _dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def _resolve_sides(teams, replay_team_id: str) -> tuple[str, str]:
    """Return (our team id, their team id) for this replay."""
    for t in teams:
        if replay_team_id in (t.get("id", ""), t.get("name", "")):
            return t["id"], (teams[1] if t is teams[0] else teams[0])["id"]
    # Fall back to the team-name hint; never assume team-a is us.
    for t in teams:
        if US_NAME_HINT.lower() in str(t.get("name", "")).lower():
            return t["id"], (teams[1] if t is teams[0] else teams[0])["id"]
    return teams[0]["id"], teams[1]["id"]


def analyze(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    replay_team_id = d.get("teamId", "")
    side = str((d.get("fixture") or {}).get("side", "home")).lower()
    # When we are the away side the replay's raw frame is flipped for *us*.
    mirror_us = side == "away"
    res = {
        "file": path.name,
        "events": Counter(),
        "our_shots": [], "their_shots": [],
        "our_passes": [], "their_passes": [],
        "our_recoveries": [], "their_recoveries": [],
        "our_zones": Counter(), "their_zones": Counter(),
        "our_ball_x": [], "their_ball_x": [],
        "shape": [], "contest": [], "gk_depth": [],
        "score": None, "n": 0, "phase_n": 0,
        "loose_x": [], "loose_n": 0,
        "recovery_zone": Counter(),
        "side": side,
    }
    seen: set = set()
    by_tick: dict = {}

    for s in d.get("snapshots", []):
        teams = s.get("teams") or []
        us_id, them_id = _resolve_sides(teams, replay_team_id)
        ta = next((t for t in teams if t["id"] == us_id), None)
        tb = next((t for t in teams if t["id"] == them_id), None)
        if ta is None or tb is None:
            continue
        res["n"] += 1
        res["score"] = (ta.get("score"), tb.get("score"))
        tick = s.get("simulationTick")
        # Index players by tick so events (which carry no position) can be
        # resolved to where the actor actually stood.
        by_tick[tick] = {}
        for t in (ta, tb):
            for q in t["players"]:
                by_tick[tick][q["id"]] = q
        if s.get("phase") not in ("openPlay", "kickoff"):
            continue
        res["phase_n"] += 1
        ball = s.get("ball") or {}
        raw_bx = float((ball.get("position") or {}).get("x", 30.0))
        raw_by = float((ball.get("position") or {}).get("y", 20.0))

        # Snapshots carry no `possessedBy` / `possessingTeam` fields -- that
        # data only appears in the event stream. Reading them anyway silently
        # classified every tick as "loose", which overstated the loose-ball
        # share as 57.7% when the real figure is 38.2%. So the holder is
        # derived geometrically: a controlled ball sits BALL_CONTROL_AHEAD
        # (0.65 m) in front of its player, so the holder is simply the closest
        # player within a body-width of the ball.
        holder_team, holder_id = _nearest_to_ball(ta, tb, us_id, raw_bx, raw_by)
        held_by_us = holder_team == us_id
        # Mirror the ball by *who holds it*: if they hold it, their +x is our -x.
        # The ball is one absolute object: flip it by whoever is holding it.
        flip = mirror_us if held_by_us else (not mirror_us)
        bx = _mx(raw_bx, flip)
        by = _my(raw_by, mirror_us if held_by_us else (not mirror_us))

        def _nq(q, mirror):
            pos = q["position"]
            return (q["id"].split(":")[-1],
                    _mx(float(pos["x"]), mirror),
                    _my(float(pos["y"]), mirror),
                    q.get("role", ""))
        ours = {k: _nq(q, mirror_us) for k, q in
                ((x["id"].split(":")[-1], x) for x in ta["players"])}
        theirs = {k: _nq(q, not mirror_us) for k, q in
                  ((x["id"].split(":")[-1], x) for x in tb["players"])}

        def mirror(team_id, x):
            return _mx(x, mirror_us if team_id == us_id else not mirror_us)

        # ---- events, resolved to actor position at their own tick
        for ev in s.get("events") or []:
            eid = ev.get("id")
            if eid is not None:
                if eid in seen:
                    continue
                seen.add(eid)
            et = ev.get("type")
            tid = ev.get("teamId")
            pid = (ev.get("playerId") or "").split(":")[-1]
            res["events"][f"{'us' if tid == us_id else 'them'}:{et}"] += 1
            snap = by_tick.get(ev.get("tick")) or {}
            q = snap.get(f"{tid}:{pid}")
            if q is None:
                continue
            ex = mirror(tid, float(q["position"]["x"]))
            ey = float(q["position"]["y"])
            rec = {
                "actor": pid,
                "role": q.get("role", "?"),
                "x": round(ex, 1), "y": round(ey, 1),
                "dist": round(PITCH_L - ex, 1),
                "power": ev.get("value"),
                "tick": ev.get("tick"),
            }
            if et == "shot":
                (res["our_shots"] if tid == us_id else res["their_shots"]).append(rec)
            elif et == "pass":
                (res["our_passes"] if tid == us_id else res["their_passes"]).append(rec)
            elif et == "recovery":
                (res["our_recoveries"] if tid == us_id else res["their_recoveries"]).append(rec)
                band = "def" if ex < 20 else ("mid" if ex < 40 else "att")
                res["recovery_zone"][f"{'us' if tid == us_id else 'them'}:{band}"] += 1
                if tid == us_id and q.get("role") != "goalkeeper":
                    res["our_ball_x"].append(ex)

        # ---- ball/possession geometry per tick
        if holder_team == us_id:
            res["our_ball_x"].append(bx)
            res["our_zones"]["def" if bx < 20 else ("mid" if bx < 40 else "att")] += 1
        elif holder_team == them_id:
            res["their_ball_x"].append(PITCH_L - bx)
            res["their_zones"]["def" if PITCH_L - bx < 20 else
                              ("mid" if PITCH_L - bx < 40 else "att")] += 1
        else:
            res["loose_x"].append(bx)
            res["loose_n"] += 1

        our_of = [v for v in ours.values() if v[3] != "goalkeeper"]
        th_of = [v for v in theirs.values() if v[3] != "goalkeeper"]
        if len(our_of) == 4 and len(th_of) == 4:
            def spread(p):
                xs = [a[1] for a in p]; ys = [a[2] for a in p]
                return (max(ys) - min(ys), max(xs) - min(xs))
            res["shape"].append((*spread(our_of), *spread(th_of)))
            res["contest"].append((
                min(_dist(a[1], a[2], bx, by) for a in our_of),
                min(_dist(a[1], a[2], bx, by) for a in th_of),
            ))
            if holder_team == them_id and bx < 32.0:
                gk = next((v for v in ours.values() if v[3] == "goalkeeper"), None)
                if gk:
                    res["gk_depth"].append(gk[1])
    return res


def _avg(v):
    return sum(v) / len(v) if v else float("nan")


def report(rows: list[dict]) -> None:
    print("=" * 78)
    print("VANGUARD (elite) DIAGNOSIS  [all numbers in OUR attacking frame]")
    print("=" * 78)
    ev = Counter()
    for r in rows:
        ev.update(r["events"])
    print("\n-- engine events (summed) --")
    for k, v in sorted(ev.items()):
        print(f"  {k:26s} {v:5d}")

    print("\n-- who shoots, and from where --")
    for label, key in (("OURS", "our_shots"), ("THEIRS", "their_shots")):
        s = [r for row in rows for r in row[key]]
        if not s:
            print(f"  {label}: none")
            continue
        d = sorted(x["dist"] for x in s)
        print(f"  {label}: n={len(s):3d} dist mean={_avg(d):5.1f} "
              f"min={d[0]:5.1f} max={d[-1]:5.1f}")
        print(f"       by player: "
              f"{Counter((x['actor'], x['role']) for x in s).most_common()}")
        goals = sum(1 for x in s if x["dist"] > 0 and x["power"] is not None)
        print(f"       power: mean={_avg([x['power'] for x in s if x['power'] is not None]):.2f}")

    print("\n-- pass power actually used --")
    for label, key in (("ours", "our_passes"), ("theirs", "their_passes")):
        p = [r["power"] for row in rows for r in row[key] if r["power"] is not None]
        if not p:
            continue
        weak = sum(1 for v in p if v < 0.15)
        print(f"  {label:7s} n={len(p):3d} mean={_avg(p):.3f} min={min(p):.3f} "
              f"under-0.15={weak:3d} ({100.0*weak/len(p):.0f}%)")

    print("\n-- progression: mean ball x (higher = closer to their goal) --")
    for label, key in (("ours", "our_ball_x"), ("theirs", "their_ball_x")):
        v = [x for row in rows for x in row[key]]
        if not v:
            continue
        v2 = sorted(v)
        print(f"  {label:7s} mean={_avg(v):5.1f} p50={v2[len(v2)//2]:5.1f} "
              f"p90={v2[int(len(v2)*0.9)]:5.1f} max={v2[-1]:5.1f} "
              f"final-third={100.0*sum(1 for x in v if x>40)/len(v):4.1f}%")
    print("  zone occupancy (share of held ticks):")
    for label, key in (("our ball", "our_zones"), ("their ball", "their_zones")):
        agg = Counter()
        for r in rows:
            agg.update(r[key])
        tot = sum(agg.values()) or 1
        print(f"    {label:11s} def {agg['def']/tot:5.1%} mid {agg['mid']/tot:5.1%} "
              f"att {agg['att']/tot:5.1%}")
    loose = sum(r["loose_n"] for r in rows)
    ph = sum(r["phase_n"] for r in rows) or 1
    print(f"    LOOSE (nobody holds it): {100.0*loose/ph:.1f}% of open-play ticks")

    print("\n-- recoveries (who wins the ball back, where) --")
    rz = Counter()
    for r in rows:
        rz.update(r["recovery_zone"])
    for k, v in sorted(rz.items()):
        print(f"  {k:14s} {v:4d}")
    for label, key in (("ours", "our_recoveries"), ("theirs", "their_recoveries")):
        rec = [r for row in rows for r in row[key]]
        if rec:
            print(f"  {label:7s} by player: {Counter(x['actor'] for x in rec).most_common()}")

    print("\n-- outfield shape (width, depth of the 4) --")
    sh = [s for r in rows for s in r["shape"]]
    if sh:
        print(f"  ours   width={_avg([s[0] for s in sh]):5.1f} depth={_avg([s[1] for s in sh]):5.1f}")
        print(f"  theirs width={_avg([s[2] for s in sh]):5.1f} depth={_avg([s[3] for s in sh]):5.1f}")

    print("\n-- nearest outfielder to the ball (contest pressure) --")
    c = [x for r in rows for x in r["contest"]]
    if c:
        print(f"  ours  mean={_avg([x[0] for x in c]):5.2f}m   "
              f"theirs mean={_avg([x[1] for x in c]):5.2f}m")
        print(f"  ours further away on {100.0*sum(1 for x in c if x[0]>x[1])/len(c):.1f}% of ticks")

    print("\n-- our GK depth while they hold the ball in our half --")
    g = [x for r in rows for x in r["gk_depth"]]
    if g:
        gs = sorted(g)
        print(f"  mean x={_avg(g):5.2f}  p50={gs[len(gs)//2]:5.2f}  "
              f"p90={gs[int(len(gs)*0.9)]:5.2f}  max={gs[-1]:5.2f}")


if __name__ == "__main__":
    paths = sorted(Path(p) for p in sys.argv[1:])
    report([analyze(p) for p in paths])
