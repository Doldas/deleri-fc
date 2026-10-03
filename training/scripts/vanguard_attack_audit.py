#!/usr/bin/env python3
"""Trace final-third choices and keeper-displacement opportunities in replays."""
from __future__ import annotations

import json
import sys
import types
from collections import Counter
from pathlib import Path

from src.policy import PolicyController
from src.runtime import RuntimeManager
from training.scripts.offline_decision_trace import _local_team_side, _obs_from_snapshot


def audit(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    us, them = _local_team_side(raw)
    fixture = raw.get("fixture") or {}
    manager = RuntimeManager()
    manager.start_match({"gameId": fixture.get("gameId", ""),
                         "randomSeed": str(fixture.get("seed", ""))})
    ctx = manager._matches[fixture.get("gameId", "")]
    controller = ctx.policy
    counts: Counter = Counter()
    samples: list[dict] = []
    active: dict = {}

    def record(entry: dict) -> None:
        counts[f"selected:{entry.get('reason', 'unknown')}"] += 1
        if len(samples) < 80:
            samples.append({**active, **entry})

    controller.log.record = record
    original_eval = controller._evaluate_attack_actions
    original_cross = controller._cross_choice
    original_cutback = controller._cutback_choice
    original_dribble = controller._dribble_target

    def eval_attack(self, inp, player, team_plan=None):
        keeper = inp.state.goalkeeper_them()
        target = original_dribble(inp, player)
        active.clear()
        active.update({
            "tick": inp.state.simulation_tick,
            "x": round(player.x, 2), "y": round(player.y, 2),
            "role": inp.roles.get(player.id),
            "gkX": round(keeper.x, 2) if keeper else None,
            "gkY": round(keeper.y, 2) if keeper else None,
            "gkCanAct": keeper.can_act if keeper else None,
            "dribbleTarget": [round(target[0], 2), round(target[1], 2)],
        })
        counts["attacking_decisions"] += 1
        if player.x > 40:
            counts["final_third_attacking_decisions"] += 1
        intent = original_eval(inp, player, team_plan)
        if samples and samples[-1].get("tick") == active.get("tick"):
            samples[-1]["actionTarget"] = intent.action_target
            samples[-1]["actionPower"] = intent.action_power
        return intent

    def cross_choice(self, inp, player):
        value = original_cross(inp, player)
        if inp.state.ball.possessing_team == "us":
            counts["cross_checks"] += 1
            counts["cross_available"] += value is not None
        return value

    def cutback_choice(self, inp, player):
        value = original_cutback(inp, player)
        if inp.state.ball.possessing_team == "us":
            counts["cutback_checks"] += 1
            counts["cutback_available"] += value is not None
        return value

    controller._evaluate_attack_actions = types.MethodType(eval_attack, controller)
    controller._cross_choice = types.MethodType(cross_choice, controller)
    controller._cutback_choice = types.MethodType(cutback_choice, controller)

    for snapshot in raw.get("snapshots", []):
        if snapshot.get("phase") == "completed":
            continue
        manager.decide(_obs_from_snapshot(snapshot, us, them))

    seen_events = set()
    for frame in raw.get("snapshots", []):
        for event in frame.get("events") or []:
            event_id = event.get("id")
            if event_id is not None and event_id in seen_events:
                continue
            if event_id is not None:
                seen_events.add(event_id)
            if event.get("type") == "slap" and event.get("teamId") == us:
                counts["our_slap_events"] += 1
                if event.get("outcome") == "knockdown":
                    counts["our_knockdowns"] += 1
    return {
        "file": path.name,
        "seed": fixture.get("seed"),
        "side": (fixture.get("side") or "unknown"),
        "counts": dict(counts),
        "samples": samples,
    }


def main(argv: list[str]) -> int:
    paths = [Path(arg) for arg in argv if arg.endswith(".replay.json")]
    if not paths:
        print("usage: python -m training.scripts.vanguard_attack_audit <replay.json> [...]",
              file=sys.stderr)
        return 2
    for path in paths:
        print(json.dumps(audit(path), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
