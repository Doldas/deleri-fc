import json
from pathlib import Path
from typing import cast

from custom_strategy import customize
from models import Decision, Observation
from src.runtime import RuntimeManager

TACTICS = json.loads(Path(__file__).with_name("tactics.json").read_text(encoding="utf-8"))

# Shared per-match context registry (keyed by gameId, seeded from randomSeed).
MANAGER = RuntimeManager()

# Kept for the builder contract: reset when a new match starts.
ASSIGNMENTS: dict[str, dict[str, dict]] = {}


def decide(observation: Observation) -> Decision:
    decision: dict = MANAGER.decide(dict(observation))
    return cast(Decision, customize(decision, observation, TACTICS))


def start_match(body: dict) -> None:
    MANAGER.start_match(body)


def _discard_match(game_id: str) -> None:
    ASSIGNMENTS.pop(game_id, None)
    MANAGER.end_match({"gameId": game_id})