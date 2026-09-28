"""P1.1 -- match time context must be relative to the configured duration.

The engine can be configured for 30 s, 60 s or 120 s matches. The historical
implementation was ``late = time_remaining <= 120.0``, which made *every* state of
a 30 s or 60 s match "late" and therefore applied the tied_late context
modifiers (passing_risk 1.05, shooting_threshold 0.95) for the whole game.

These tests pin the duration-relative contract, the exact boundary, and the
requirement that there is a single authoritative definition.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.policy import (
    DEFAULT_MATCH_DURATION,
    LATE_GAME_RATIO,
    PolicyInput,
    is_late,
    match_context,
    remaining_ratio,
)
from src.state import GameState
from src.tactics import TacticalState
from tests.test_policy import make_inp, obs

SUPPORTED_DURATIONS = (30.0, 60.0, 120.0)


def _inp(time_remaining, match_duration, su=0, st=0):
    """A real PolicyInput built through the production fixture helper."""
    observation = obs((35, 20), "us")
    observation["timeRemainingSeconds"] = time_remaining
    observation["score"] = {"us": su, "them": st}
    inp = make_inp(observation, TacticalState.ATTACK)
    inp.match_duration = match_duration
    return inp


class TestRemainingRatio(unittest.TestCase):
    def test_ratio_is_duration_relative(self):
        self.assertAlmostEqual(remaining_ratio(30.0, 60.0), 0.5)
        self.assertAlmostEqual(remaining_ratio(15.0, 30.0), 0.5)
        self.assertAlmostEqual(remaining_ratio(60.0, 120.0), 0.5)

    def test_ratio_clamped_to_unit_interval(self):
        self.assertEqual(remaining_ratio(30.0, 30.0), 1.0)  # kickoff
        self.assertEqual(remaining_ratio(300.0, 60.0), 1.0)  # over-full clock
        self.assertEqual(remaining_ratio(-5.0, 60.0), 0.0)  # negative clock

    def test_invalid_duration_is_defensive(self):
        for bad in (None, 0.0, -30.0):
            # An unknown clock must never masquerade as a late match.
            self.assertEqual(remaining_ratio(5.0, bad), 1.0, bad)
            self.assertFalse(is_late(5.0, bad), bad)

    def test_none_time_remaining_is_defensive(self):
        self.assertEqual(remaining_ratio(None, 60.0), 1.0)
        self.assertFalse(is_late(None, 60.0))


class TestLateThresholdBoundaries(unittest.TestCase):
    """The exact matrix required by P1.1, including one tick either side."""

    # duration -> (kickoff, one tick above threshold, exact threshold)
    CASES = {
        30.0: (30.0, 7.0, 6.0),
        60.0: (60.0, 13.0, 12.0),
        120.0: (120.0, 25.0, 24.0),
    }

    def test_kickoff_is_not_late(self):
        for duration, (kickoff, _, _) in self.CASES.items():
            self.assertEqual(match_context(kickoff, 0, 0, duration), "tied")
            self.assertFalse(is_late(kickoff, duration))

    def test_just_above_threshold_is_not_late(self):
        for duration, (_, above, _) in self.CASES.items():
            self.assertFalse(is_late(above, duration), duration)
            self.assertEqual(match_context(above, 0, 0, duration), "tied")

    def test_exact_threshold_is_late(self):
        for duration, (_, _, boundary) in self.CASES.items():
            self.assertTrue(is_late(boundary, duration), duration)
            self.assertEqual(match_context(boundary, 0, 0, duration), "tied_late")

    def test_boundary_equals_twenty_percent(self):
        for duration, (_, _, boundary) in self.CASES.items():
            self.assertAlmostEqual(boundary / duration, LATE_GAME_RATIO)
            self.assertAlmostEqual(boundary / duration, 0.20)

    def test_full_time_is_late(self):
        for duration in SUPPORTED_DURATIONS:
            self.assertTrue(is_late(0.0, duration))
            self.assertEqual(match_context(0.0, 0, 0, duration), "tied_late")


class TestAllSixScoreStates(unittest.TestCase):
    """leading/leading_late/trailing/trailing_late/tied/tied_late per duration."""

    def test_score_states_at_kickoff(self):
        for duration in SUPPORTED_DURATIONS:
            self.assertEqual(match_context(duration, 1, 0, duration), "leading")
            self.assertEqual(match_context(duration, 0, 1, duration), "trailing")
            self.assertEqual(match_context(duration, 0, 0, duration), "tied")
            self.assertEqual(match_context(duration, 2, 1, duration), "leading")
            self.assertEqual(match_context(duration, 1, 2, duration), "trailing")

    def test_score_states_late(self):
        for duration in SUPPORTED_DURATIONS:
            self.assertEqual(match_context(1.0, 1, 0, duration), "leading_late")
            self.assertEqual(match_context(1.0, 0, 1, duration), "trailing_late")
            self.assertEqual(match_context(1.0, 0, 0, duration), "tied_late")
            self.assertEqual(match_context(1.0, 2, 1, duration), "leading_late")
            self.assertEqual(match_context(1.0, 1, 2, duration), "trailing_late")

    def test_all_six_states_are_reachable(self):
        seen = set()
        for duration in SUPPORTED_DURATIONS:
            for t in (duration, 1.0):
                for su, st in ((1, 0), (0, 1), (0, 0)):
                    seen.add(match_context(t, su, st, duration))
        self.assertEqual(
            seen,
            {
                "leading",
                "trailing",
                "tied",
                "leading_late",
                "trailing_late",
                "tied_late",
            },
        )


class TestSingleAuthoritativeDefinition(unittest.TestCase):
    """PolicyInput.late() and match_context() must never disagree."""

    def test_late_and_context_agree(self):
        for duration in SUPPORTED_DURATIONS:
            for t in (duration, duration * 0.5, duration * 0.21, duration * 0.19, 0.0):
                for su, st in ((0, 0), (1, 0), (0, 1)):
                    inp = _inp(t, duration, su, st)
                    self.assertEqual(
                        inp.late(),
                        inp.context().endswith("_late"),
                        f"dur={duration} t={t} {su}-{st}",
                    )

    def test_late_never_uses_a_fixed_threshold(self):
        """The old 120.0 fallback must not reappear for a short match."""
        for duration in (30.0, 60.0):
            inp = _inp(duration, duration)
            self.assertFalse(inp.late())
            self.assertEqual(inp.context(), "tied")

    def test_unknown_duration_agrees(self):
        inp = _inp(1.0, None)
        self.assertFalse(inp.late())
        self.assertEqual(inp.context(), "tied")


class TestProductionPathPropagation(unittest.TestCase):
    """The duration must survive runtime -> PolicyInput -> context."""

    def test_runtime_threads_duration_into_policy_input(self):
        from src.runtime import MatchContext, RuntimeManager
        from src.config import RuntimeConfig

        for duration in SUPPORTED_DURATIONS:
            manager = RuntimeManager(RuntimeConfig(enable_mcts=False))
            manager.start_match(
                {
                    "gameId": f"g{int(duration)}",
                    "randomSeed": "1",
                    "duration": duration,
                }
            )
            ctx = manager.get_or_create({"gameId": f"g{int(duration)}"})
            self.assertEqual(ctx.match_duration, duration)
            self.assertTrue(ctx.duration_known)
            manager.end_match({"gameId": f"g{int(duration)}"})

    def test_runtime_adopts_duration_from_kickoff_observation(self):
        """No duration in the config: learn it from the first observation."""
        from src.runtime import RuntimeManager
        from src.config import RuntimeConfig

        manager = RuntimeManager(RuntimeConfig(enable_mcts=False))
        manager.start_match({"gameId": "gk", "randomSeed": "1"})
        ctx = manager.get_or_create({"gameId": "gk"})
        self.assertFalse(ctx.duration_known)

        observation = obs((35, 20), "us")
        observation["gameId"] = "gk"
        observation["timeRemainingSeconds"] = 30.0
        manager.decide(observation)

        self.assertTrue(ctx.duration_known)
        self.assertEqual(ctx.match_duration, 30.0)
        # And the policy now sees a correctly proportioned match.
        observation["timeRemainingSeconds"] = 6.0
        decision = manager.decide(observation)
        self.assertTrue(decision["intents"])
        manager.end_match({"gameId": "gk"})

    def test_default_duration_is_only_a_fallback(self):
        self.assertEqual(DEFAULT_MATCH_DURATION, 60.0)


if __name__ == "__main__":
    unittest.main()
