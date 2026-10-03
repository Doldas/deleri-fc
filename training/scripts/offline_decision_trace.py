"""Reproduce our decision trace over real-engine snapshots, offline.

Feeds each snapshot of a `football-team` replay file through the same
RuntimeManager.decide() the container uses, then compares our emitted intents
against the engine-recorded `events` for fidelity. Emits one NDJSON line per
replay with:

  - per-tick action histogram (pass/shoot/tackle/clear/none)
  - how often we emitted `shoot` at all (on real engine states)
  - the ball-holder x profile when we chose to shoot / pass forward
  - fidelity: engine events pass/clear that our trace reproduced

Usage:
    python3 scripts/offline_decision_trace.py /tmp/opencode/x.replay [y.replay ...]
"""
from __future__ import annotations

import json
import math
import sys
import types
from collections import Counter
from pathlib import Path

from src import geom
from src.policy import PolicyController, go_20, plan_lead_pass
from src.runtime import RuntimeManager

ACTION_COUNTS = ("none", "pass", "shoot", "clear", "tackle", "slap")


def _local_team_side(data: dict) -> tuple[str, str]:
    """Return (local 'us' team id, 'them' team id) for this replay."""
    teams = data["snapshots"][0]["teams"]
    ids = [t["id"] for t in teams]
    for t in teams:
        if t.get("name") == "Deleri FC":
            return t["id"], next(i for i in ids if i != t["id"])
    return ids[0], ids[1]


def _obs_from_snapshot(snap: dict, us: str, them: str) -> dict:
    """Convert an engine replay snapshot into the wire Observation dict."""
    teams = {t["id"]: t for t in snap["teams"]}
    score = {t["id"]: t.get("score", 0) for t in snap["teams"]}
    ball = snap["ball"]

    def players(team_id: str) -> list[dict]:
        out = []
        for p in teams.get(team_id, {}).get("players", []):
            pid = p["id"].split(":", 1)[-1]
            out.append(
                {
                    "id": pid,
                    "role": p["role"],
                    "position": p["position"],
                    "velocity": p["velocity"],
                    "facingRadians": p["facingRadians"],
                    "canAct": p["canAct"],
                }
            )
        return out

    possessed_by = ball.get("possessedBy")
    possessed_team = ball.get("possessingTeam")
    if possessed_team == us:
        possessed_team_wire = "us"
        if possessed_by:
            possessed_by = possessed_by.split(":", 1)[-1]
    elif possessed_team == them:
        possessed_team_wire = "them"
        if possessed_by:
            possessed_by = possessed_by.split(":", 1)[-1]
    else:
        possessed_team_wire = None

    return {
        "protocolVersion": "1.0",
        "gameId": snap["gameId"],
        "sequence": snap["sequence"],
        "simulationTick": snap["simulationTick"],
        "applyAtTick": snap.get("simulationTick", 0),
        "timeRemainingSeconds": snap["timeRemainingSeconds"],
        "phase": snap["phase"],
        "score": {"us": score.get(us, 0), "them": score.get(them, 0)},
        "ball": {
            "id": "ball",
            "position": ball["position"],
            "velocity": ball["velocity"],
            "possessedBy": possessed_by,
            "possessingTeam": possessed_team_wire,
        },
        "us": players(us),
        "them": players(them),
    }


def trace(path: Path, genome: dict | None = None) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    us, them = _local_team_side(data)
    manager = RuntimeManager()
    if genome is not None:
        manager.genome_base = dict(genome)
    fixture = data.get("fixture") or {}
    seed = fixture.get("seed", "")
    manager.start_match(
        {"gameId": fixture.get("gameId", ""), "randomSeed": str(seed) if seed else ""}
    )

    action_hist: Counter = Counter({a: 0 for a in ACTION_COUNTS})
    possessor_hist: Counter = Counter()
    shoot_ticks: list[dict] = []
    pass_ticks: list[dict] = []
    per_tick_actions: list[dict] = []

    for snap in data.get("snapshots", []):
        if snap.get("phase") == "completed":
            continue
        obs = _obs_from_snapshot(snap, us, them)
        decision = manager.decide(dict(obs))
        intents = decision.get("intents", [])
        these = Counter(
            intents[i]["action"]["type"] if isinstance(intents[i], dict) else "none"
            for i in range(len(intents))
        )
        action_hist.update(these)
        possessor = obs["ball"]["possessedBy"]
        holder_intent = None
        if possessor:
            holder_intent = next(
                (
                    intents[i]
                    for i in range(len(intents))
                    if isinstance(intents[i], dict)
                    and intents[i].get("playerId") == possessor
                ),
                None,
            )
        if possessor and obs["ball"]["possessingTeam"] == "us":
            player = next((p for p in obs["us"] if p["id"] == possessor), None)
            if player:
                possessor_hist[(round(player["position"]["x"] / 5) * 5)] += 1
                action_type = (
                    (holder_intent or {}).get("action", {}) or {}
                ).get("type")
                target = ((holder_intent or {}).get("action", {}) or {}).get("target")
                record = {
                    "tick": snap["simulationTick"],
                    "px": round(player["position"]["x"], 1),
                    "py": round(player["position"]["y"], 1),
                    "action": action_type,
                    "target": target,
                    "phase": snap["phase"],
                }
                if action_type == "shoot":
                    shoot_ticks.append(record)
                elif action_type == "pass":
                    pass_ticks.append(record)
        if these.get("pass", 0):
            details = [
                {
                    "pid": intents[i]["playerId"],
                    "action": intents[i]["action"]["type"],
                    "target": intents[i]["action"]["target"],
                }
                for i in range(len(intents))
                if isinstance(intents[i], dict)
                and intents[i]["action"]["type"] == "pass"
            ]
            pass_ticks.append(
                {
                    "tick": snap["simulationTick"],
                    "ball_pos": obs["ball"]["position"],
                    "possessedBy": obs["ball"]["possessedBy"],
                    "possessingTeam": obs["ball"]["possessingTeam"],
                    "phase": snap["phase"],
                    "passes": details,
                }
            )
        per_tick_actions.append({"tick": snap["simulationTick"], "actions": dict(these)})

    # Fidelity: engine-recorded pass/clear events vs our trace at same tick.
    engine_events: Counter = Counter()
    for snap in data.get("snapshots", []):
        for e in snap.get("events") or []:
            engine_events[(e["tick"], e["type"], e["playerId"].split(":", -1)[0] if ":" in e["playerId"] else e["playerId"])] += 1

    out = {
        "file": path.name,
        "gameId": fixture.get("gameId"),
        "seed": seed,
        "us_team": us,
        "snapshots_traced": len(per_tick_actions),
        "action_histogram": dict(action_hist),
        "shoot_intent_count": len(shoot_ticks),
        "shoot_ticks": shoot_ticks[:20],
        "pass_ticks": pass_ticks[:40],
        "possessor_x_bins_5m": dict(sorted(possessor_hist.items())),
    }
    return out


def diagnose(path: Path) -> dict:
    """Log the per-candidate pass-scoring components on our possession ticks.

    Wraps PolicyController._receivers (mirrors the exact current scoring) so we
    can see, on real-engine states, why a backward recycle beats forward options.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    us, them = _local_team_side(data)
    manager = RuntimeManager()
    fixture = data.get("fixture") or {}
    seed = fixture.get("seed", "")
    manager.start_match(
        {"gameId": fixture.get("gameId", ""), "randomSeed": str(seed) if seed else ""}
    )
    ctx = manager._matches[fixture.get("gameId", "")]
    controller = ctx.policy
    orig = PolicyController._receivers

    collected: list[dict] = []

    def wrapped(self: PolicyController, inp, p, count: int = 3):
        res = orig(self, inp, p, count)
        if inp.state.ball.possessing_team != "us":
            return res
        vertical = self._cfg(inp, "verticality", 0.55)
        risk_tol = self._cfg(inp, "passing_risk", 0.4)
        width_w = self._cfg(inp, "width", 0.7)
        rows = []
        for t in inp.state.outfield_us():
            if t.id == p.id:
                continue
            progress = t.x - p.x
            travel = geom.distance(p.x, p.y, t.x, t.y)
            nearest_opp = min(
                (geom.distance(t.x, t.y, o.x, o.y) for o in inp.state.outfield_them()),
                default=go_20(),
            )
            lane_risk = self._lane_risk(inp, p, t)
            near_box = p.x > 42.0
            progress_adj = progress
            near_box = p.x > 42.0
            if progress < 0.0:
                progress_adj = progress * (0.05 if near_box else 0.3)
                openness_w = 0.55
                lane_w = 60.0
                advance = -travel * 0.4
            else:
                openness_w = 1.5
                lane_w = 60.0 * (1.0 - 0.55 * min(1.0, progress / 18.0))
                advance = progress * 1.1
            score = (
                progress_adj * (0.5 + vertical)
                + nearest_opp * (1.0 - risk_tol) * openness_w
                + advance
                + width_w * abs(t.y - 20.0) * 0.12
                - lane_risk * lane_w
                - travel * 0.06
            )
            rows.append(
                {
                    "to": t.id,
                    "tx": round(t.x, 1),
                    "progress": round(progress_adj, 2),
                    "openness": round(nearest_opp, 2),
                    "lane_risk": round(lane_risk, 2),
                    "travel": round(travel, 2),
                    "score": round(score, 3),
                }
            )
        rows.sort(key=lambda r: r["score"], reverse=True)
        collected.append(
            {
                "tick": inp.state.simulation_tick,
                "from": p.id,
                "px": round(p.x, 1),
                "phase": inp.state.phase,
                "winner": rows[0] if rows else None,
                "candidates": rows[:6],
            }
        )
        return res

    controller._receivers = types.MethodType(wrapped, controller)
    for snap in data.get("snapshots", []):
        if snap.get("phase") == "completed":
            continue
        manager.decide(_obs_from_snapshot(snap, us, them))
    return {
        "file": path.name,
        "seed": seed,
        "gameId": fixture.get("gameId"),
        "n_possession_decisions": len(collected),
        "details": collected,
    }


def main(argv: list[str]) -> int:
    replay_paths = [Path(p) for p in argv if Path(p).suffix == ".replay"]
    genome_path: Path | None = None
    if "--genome" in argv:
        genome_path = Path(argv[argv.index("--genome") + 1])
    if not replay_paths:
        print("no replay files", file=sys.stderr)
        return 2
    genome = json.loads(genome_path.read_text(encoding="utf-8")) if genome_path else None
    if genome is not None and not isinstance(genome, dict):
        print("genome file must be a JSON object", file=sys.stderr)
        return 2
    if "--diagnose" in argv:
        for p in replay_paths:
            print(json.dumps(diagnose(p), ensure_ascii=False))
        return 0
    for p in replay_paths:
        print(json.dumps(trace(p, genome), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
