"""Measure whether the opponent brain actually contests the ball.

Reports the numbers behind the complaint that the opponents "avoid the ball and
let us take it": how often the brain designates anyone to chase a loose ball,
how close its nearest defender gets to our carrier, and how often it attempts a
tackle or a slap. Run with `--old` to patch the pre-fix behaviour back in and
get a baseline from the same seeds.
"""

import argparse
import random
import statistics
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from engine_probe import build_module, make_controller  # noqa: E402
from src.opponents import engine_brain  # noqa: E402
from src.runtime import RuntimeManager  # noqa: E402
from src.sim import OPPONENTS, play_match  # noqa: E402

PARAMS = {
    "family": "high_press", "press_intensity": 0.95, "press_trigger": 0.55,
    "compactness": 0.85, "tackling": 0.85, "tempo": 1.0, "noise": 0.03,
    "gk_aggression": 0.9, "gk_speed": 1.0, "counterpress": 0.75,
    "directness": 0.5, "press_delay": 0.0, "shoot_range": 22.0,
}

PITCH = 60.0


def patch_old() -> None:
    """Restore the pre-fix press plan: no chaser at all for a loose ball."""
    def old_press_plan(self, bx, by):
        if not self.outfield or not self.they_have:
            self._pressers = set()
            return self._pressers
        if self._soft_press_ttl > 0:
            self._soft_press_ttl -= 1
        trigger_x = float(self.p.get("press_trigger", 0.45)) * PITCH
        dangerous = (self.their_holder is not None
                     and self.their_holder.x < trigger_x) or bx < 18.0
        if not dangerous:
            self._pressers = set()
            return self._pressers
        intensity = min(1.0, max(0.0, float(self.p.get("press_intensity", 0.6))))
        if self._soft_press_ttl > 0:
            intensity = min(intensity, 0.25)
        count = 1 + int(round(intensity * 2.5))
        order = sorted(self.outfield, key=lambda q: (engine_brain._dist(
            q.x, q.y, bx, by), q.pid))
        self._pressers = {q.pid for q in
                          order[: max(1, min(count, len(order)))]}
        return self._pressers

    engine_brain._Brain._press_plan = old_press_plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--decisions", type=int, default=900)
    ap.add_argument("--old", action="store_true",
                    help="use the pre-fix press plan as a baseline")
    args = ap.parse_args()

    if args.old:
        patch_old()

    module = build_module(PARAMS)
    # make_controller returns the controller *class*; the registry instantiates
    # it with the match rng.
    OPPONENTS["probe"] = make_controller(module)

    stats = {
        "loose_ticks": 0, "loose_chased": 0,
        "we_have_ticks": 0, "pressers_when_we_have": 0,
        "nearest_gap": [], "tackle_attempts": 0, "their_carry_ticks": 0,
    }
    rows = []
    real_decide = module.decide
    state = stats

    def decide(obs):
        # Inspect the brain *after* it has updated and planned, which is the
        # only point where `_pressers` reflects this tick's decision.
        decision = real_decide(obs)
        brains = [b for b in module._BRAINS.values()
                  if b.game_id == obs["gameId"]]
        if brains:
            brain = brains[0]
            if brain.loose:
                state["loose_ticks"] += 1
                if brain._pressers:
                    state["loose_chased"] += 1
            elif brain.we_have and brain.their_holder is not None:
                state["we_have_ticks"] += 1
                if brain._pressers:
                    state["pressers_when_we_have"] += 1
                gap = min(engine_brain._dist(q.x, q.y, brain.their_holder.x,
                                             brain.their_holder.y)
                          for q in brain.outfield)
                state["nearest_gap"].append(gap)
            elif brain.they_have:
                state["their_carry_ticks"] += 1
        for intent in decision.get("intents", []):
            if (intent.get("action") or {}).get("type") in ("tackle", "slap"):
                state["tackle_attempts"] += 1
        return decision

    module.decide = decide

    for seed in range(args.games):
        rows.append(play_match(
            RuntimeManager(), "probe", rng=random.Random(1000 + seed),
            decisions=args.decisions).summary())

    gf = sum(r["score_us"] for r in rows)
    ga = sum(r["score_them"] for r in rows)
    poss = statistics.fmean(r["possession%"] for r in rows)
    loose = stats["loose_ticks"] or 1
    we_have = stats["we_have_ticks"] or 1
    gap = stats["nearest_gap"] or [float("nan")]

    label = "OLD (pre-fix)" if args.old else "NEW (fixed)  "
    print(f"\n=== {label} brain vs My Team FC, {args.games} games ===")
    print(f"  score (us-them)        {gf}-{ga}")
    print(f"  possession             {poss:5.1f}%")
    print(f"  shots (for-against)    {sum(r['shots'] for r in rows)}-"
          f"{sum(r['shots_conceded'] for r in rows)}")
    print(f"  loose-ball ticks       {loose}")
    print(f"  .. with a chaser sent  {stats['loose_chased']:6d}  "
          f"({100.0 * stats['loose_chased'] / loose:5.1f}%)   <-- contest rate")
    print(f"  carrier ticks          {we_have}")
    print(f"  .. with a presser      {stats['pressers_when_we_have']:6d}  "
          f"({100.0 * stats['pressers_when_we_have'] / we_have:5.1f}%)")
    print(f"  nearest gap to carrier {statistics.fmean(gap):5.2f} m "
          f"(min {min(gap):.2f})   <-- lower is tighter defending")
    print(f"  their ball-carry ticks {stats['their_carry_ticks']}")
    print(f"  tackle/slap intents    {stats['tackle_attempts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
