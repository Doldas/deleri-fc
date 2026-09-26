"""Analyze football-team .replay files (ground-truth observability).

Usage:
    python3 scripts/analyze_replay.py /tmp/opencode/a.replay [/tmp/opencode/b.replay ...]

Emits JSON-lines per replay describing the open-play ground truth that the
final simulation summary hides: distinct engine events (pass/shoot/tackle/...),
ball trace (where the ball actually lives), closest approach to goal, and the
resultRreason that decided the match. Deduplicates events by id (the engine
re-broadcasts events across snapshots).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


GOAL_X = 60.0
PITCH_Y = 40.0


def analyze(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    snaps = data.get("snapshots", [])
    events: dict[str, dict] = {}
    zones = Counter()
    poss_team = Counter()
    ball_x = {"min": 99.0, "max": -99.0}
    closest = {"tick": None, "dist": 99.0, "phase": None, "player": None}
    open_play = 0
    for s in snaps:
        phase = s.get("phase")
        ball = s.get("ball") or {}
        if phase == "openPlay" or phase == "kickoff":
            open_play += 1
            x = (ball.get("position") or {}).get("x")
            y = (ball.get("position") or {}).get("y")
            if x is not None:
                ball_x["min"] = min(ball_x["min"], x)
                ball_x["max"] = max(ball_x["max"], x)
                dist = GOAL_X - x
                if y is not None and 12.0 <= y <= 28.0 and dist < closest["dist"]:
                    closest = {"tick": s.get("simulationTick"), "dist": round(dist, 2),
                               "phase": phase, "player": (ball.get("possessedBy") or "none")}
            if x is not None:
                zones["def" if x < 20 else ("att" if x > 40 else "mid")] += 1
            if ball.get("possessingTeam"):
                poss_team[ball["possessingTeam"]] += 1
        for e in s.get("events") or []:
            events[e["id"]] = e
    ex = Counter(e.get("type") for e in events.values())
    completed = next((s for s in snaps if s.get("phase") == "completed"), None)
    stats = (completed or {}).get("statistics") or {"teams": []}
    stat_rows = {t["teamId"]: t for t in stats["teams"]}
    return {
        "file": str(path.name),
        "gameId": data.get("fixture", {}).get("gameId"),
        "seed": data.get("fixture", {}).get("seed"),
        "side": data.get("fixture", {}).get("side"),
        "snapshots": len(snaps),
        "open_play_snaps": open_play,
        "distinct_events": dict(ex),
        "final_score": [stat_rows.get(t, {}).get("score") for t in stat_rows],
        "resultReason": (completed or {}).get("resultReason"),
        "winnerTeamId": (completed or {}).get("winnerTeamId"),
        "statistics": {
            t: {k: stat_rows.get(t, {}).get(k) for k in
                ("possessionPercent", "shots", "passes", "ballRecoveries", "distanceMeters", "missedDecisions")}
            for t in stat_rows
        },
        "ball_x_min": round(ball_x["min"], 2),
        "ball_x_max": round(ball_x["max"], 2),
        "closest_approach_to_attacking_goal": closest,
        "ball_zones_ticks": dict(zones),
        "possession_ticks": dict(poss_team),
    }


def main(argv: list[str]) -> int:
    paths = [Path(p) for p in argv if Path(p).suffix == ".replay"]
    if not paths:
        print("no replay files", file=sys.stderr)
        return 2
    for p in paths:
        print(json.dumps(analyze(p), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))