"""Runtime — perception → world → tactics → opponent → policy → intents.

Per-match context is keyed by gameId and seeded deterministically
(AISTRATEGI §41–42, §5 of AGENTS.md).
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .config import RuntimeConfig, default_genome, make_genome
from .evaluate import Metrics
from .mcts import MCTSPlanner
from .opponent import OpponentModel
from .policy import (
    DEFAULT_MATCH_DURATION,
    PolicyController,
    PolicyInput,
    is_late,
    match_context,
)
from .state import GameState, WorldModel
from .tactics import PressPlan, TacticalState, assign_roles, detect, plan_press

_TACTICS_JSON = Path(__file__).resolve().parents[1] / "tactics.json"
_EVOLVED_POLICY_JSON = Path(__file__).resolve().parents[1] / "artifacts" / "policies" / "distilled_policy.json"


def load_tactics() -> dict:
    try:
        return json.loads(_TACTICS_JSON.read_text(encoding="utf-8"))
    except OSError:
        return {"formation": {"slots": []}, "style": {}}


def load_evolved_genome() -> dict[str, float] | None:
    try:
        data = json.loads(_EVOLVED_POLICY_JSON.read_text(encoding="utf-8"))
    except OSError:
        return None
    genome = data.get("genome")
    if not isinstance(genome, dict):
        return None
    out = default_genome()
    for k, v in genome.items():
        if k in out:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def seed_int(seed_str: str, game_id: str) -> int:
    """Deterministic seed integer from the match-start random seed + game id.

    Uses hashlib (not builtin `hash`) so it is stable across processes/runs."""
    raw = f"{seed_str}\x00{game_id}".encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:4], "big")


@dataclass
class MatchContext:
    game_id: str
    seed: int
    genome: dict[str, float] = field(default_factory=default_genome)
    slots: list[dict] = field(default_factory=list)
    roles: dict[str, str] = field(default_factory=dict)
    opp: OpponentModel = field(default_factory=OpponentModel)
    metrics: Metrics = field(default_factory=Metrics)
    policy: PolicyController = field(default_factory=PolicyController)
    config: RuntimeConfig = field(default_factory=RuntimeConfig)
    prev_had_control: bool = False
    had_first_state: bool = False
    their_possessor_id: str | None = None
    their_possession_ticks: int = 0
    backward_passes: int = 0
    lost_possession_last_tick: bool = False  # Track if we lost possession last tick
    counter_press_active: bool = False  # Counter-press active flag
    rng: random.Random = field(default_factory=lambda: random.Random())
    explanations: list[dict] = field(default_factory=list)
    mcts_planner: MCTSPlanner | None = None
    match_duration: float = 60.0  # Match duration in seconds (default 60s)
    # True once match_duration came from the match configuration (or the first
    # observation) rather than from the default. Guards the fallback adoption
    # below so a default can never masquerade as a real configuration.
    duration_known: bool = False

    def assign_slots(self, state: GameState) -> None:
        if not self.roles and state.us:
            self.roles = assign_roles(state, self.slots)

    def record_decision(self, action: dict) -> None:
        if len(self.explanations) < 128:
            self.explanations.append(action)

    def summary(self) -> dict:
        return {
            "game_id": self.game_id,
            "seed": self.seed,
            "genome": self.genome,
            "opponent": self.opp.summary(),
            "metrics": self.metrics.summary(),
        }


class RuntimeManager:
    """Thread-safe registry of per-match contexts."""

    def __init__(self, runtime_config: RuntimeConfig | None = None):
        self._lock = threading.Lock()
        self._matches: dict[str, MatchContext] = {}
        self.runtime_config = runtime_config or RuntimeConfig()
        self.configs = load_tactics()
        self.genome_base = default_genome()
        evolved = load_evolved_genome()
        if evolved:
            self.genome_base = evolved
        else:
            self.genome_base = make_genome(self.configs.get("style"))

    def start_match(self, body: dict) -> None:
        game_id = str(body.get("gameId", ""))
        seed_str = str(body.get("randomSeed", ""))
        seed = seed_int(seed_str, game_id) if seed_str else 0
        config = self.runtime_config
        # Capture the authoritative match duration so every tactical time
        # decision is relative to it (30/60/120 s are all valid). We take the
        # value from the match configuration, falling back to the kickoff
        # timeRemainingSeconds observed in decide(); DEFAULT_MATCH_DURATION is
        # the last resort only.
        duration = body.get("duration", body.get("timeRemainingSeconds"))
        duration_known = duration is not None
        match_duration = (
            float(duration) if duration_known else DEFAULT_MATCH_DURATION
        )
        with self._lock:
            ctx = MatchContext(
                game_id=game_id,
                seed=seed,
                genome=dict(self.genome_base),
                slots=list(self.configs.get("formation", {}).get("slots", [])),
                config=config,
                match_duration=match_duration,
                duration_known=duration_known,
            )
            ctx.rng = random.Random(seed ^ int(body.get("seriesId", "")[:8].encode("utf-8").hex() or "0", 16))
            if config.enable_mcts:
                ctx.mcts_planner = MCTSPlanner(config.genome, config.reward_weights, ctx.rng)
            self._matches[game_id] = ctx

    def end_match(self, body: dict) -> dict | None:
        game_id = str(body.get("gameId", ""))
        with self._lock:
            ctx = self._matches.pop(game_id, None)
        return ctx.summary() if ctx else None

    def get_or_create(self, observation: dict) -> MatchContext:
        game_id = str(observation.get("gameId", ""))
        with self._lock:
            ctx = self._matches.get(game_id)
            if ctx is None:
                ctx = MatchContext(
                    game_id=game_id,
                    seed=seed_int("", game_id),
                    genome=dict(self.genome_base),
                    slots=list(self.configs.get("formation", {}).get("slots", [])),
                    config=self.runtime_config,
                )
                ctx.rng = random.Random(ctx.seed % (2**31))
                if self.runtime_config.enable_mcts:
                    ctx.mcts_planner = MCTSPlanner(ctx.genome, self.runtime_config.reward_weights, ctx.rng)
                self._matches[game_id] = ctx
            return ctx

    def decide(self, observation: dict) -> dict:
        ctx = self.get_or_create(observation)
        state = GameState.from_observation(observation)
        ctx.assign_slots(state)
        world = WorldModel.build(state)

        # If the match configuration never stated a duration, adopt it from the
        # kickoff observation: at the start of a match timeRemainingSeconds *is*
        # the duration. Ties the policy to the real 30/60/120 s configuration
        # instead of a default.
        if not ctx.duration_known and state.time_remaining > 0.0:
            ctx.match_duration = state.time_remaining
            ctx.duration_known = True

        # Maintain possession-phase memory.
        ctx.opp.update(state)
        ctx.metrics.observe(state, world)

        tactical_state = self._detect(ctx, state, world)
        genome = ctx.genome

        # ADAPTACT: Apply opponent-specific counter-tactics (AISTRATEGI §38)
        # Blend base genome with archetype counters for a tailored approach.
        counter_tactics = ctx.opp.get_counter_tactics()
        adapted_genome = dict(genome)
        for key, value in counter_tactics.items():
            if key in adapted_genome:
                # Blend 70% base, 30% counter-tactic for smooth adaptation
                adapted_genome[key] = adapted_genome[key] * 0.7 + value * 0.3

# Genome hash logging for debugging
        import hashlib
        def _genome_hash(g):
            raw = ",".join(f"{k}:{g.get(k,0):.6f}" for k in sorted(g.keys()))
            return hashlib.sha1(raw.encode()).hexdigest()[:8]
        
        base_hash = _genome_hash(genome)
        adapted_hash = _genome_hash(adapted_genome)
        # Policy will use adapted_genome; MCTS will be updated below
        mcts_hash = "none"
        
        # §32 context-sensitive press planning: chase harder when trailing late,
        # sit calmer when leading late. Duration must be threaded in, otherwise
        # the context falls back to "not late" and the plan is always calm.
        ctx_str = match_context(
            state.time_remaining,
            state.score_us,
            state.score_them,
            ctx.match_duration,
        )
        press_genome = adapted_genome
        if ctx_str == "trailing_late":
            press_genome = dict(adapted_genome)
            press_genome["press_trigger_threshold"] = adapted_genome.get("press_trigger_threshold", 0.4) * 0.6
            press_genome["press_intensity"] = min(1.0, adapted_genome.get("press_intensity", 0.55) + 0.25)
        elif ctx_str == "leading_late":
            press_genome = dict(adapted_genome)
            press_genome["press_trigger_threshold"] = adapted_genome.get("press_trigger_threshold", 0.4) * 1.4
            press_genome["press_intensity"] = max(0.0, adapted_genome.get("press_intensity", 0.55) - 0.2)

        if state.ball.possessing_team == "them":
            press_plan = plan_press(state, world, press_genome, ctx.roles)
        else:
            press_plan = PressPlan()

        # Counter-press trigger: if we just regained possession, activate counter-press
        has_control_now = state.has_control()
        if not ctx.prev_had_control and has_control_now:
            ctx.counter_press_active = True
        elif ctx.prev_had_control and not has_control_now:
            ctx.lost_possession_last_tick = True
        elif has_control_now:
            ctx.lost_possession_last_tick = False
        ctx.prev_had_control = has_control_now

        inp = PolicyInput(
            state=state,
            world=world,
            # Use adapted genome (with archetype counters + context mods) for full policy
            config=RuntimeConfig(genome=adapted_genome),
            tactical_state=tactical_state,
            press_plan=press_plan,
            roles=ctx.roles,
            their_possession_ticks=ctx.their_possession_ticks,
            opp=ctx.opp,
            time_remaining=state.time_remaining,
            score_us=state.score_us,
            score_them=state.score_them,
            counter_press_active=ctx.counter_press_active,
            match_duration=ctx.match_duration,
        )

        intents = ctx.policy.decide(inp)

        # Reset counter-press after it's been used for one decision cycle
        if ctx.counter_press_active:
            ctx.counter_press_active = False

# Optional limited MCTS override on high-value attacking states only.
        # Use adapted genome for MCTS too.
        mcts_note = None
        if ctx.mcts_planner is not None and state.has_control():
            # Only trigger MCTS in high-value attacking situations:
            # - Final third (ball_zone == "final")
            # - Elite goalkeeper detected
            # - Low block detected
            # - Wall attack opportunity
            # - Late game trailing (desperation)
            # - Counter-press active
            ctx.opp.update(state)  # Refresh opponent model
            # Duration-relative late test, via the single authority in policy.
            # This used to be an inline `ctx.match_duration * 0.2 >= ...`, a
            # third copy of the definition that could drift from the others.
            late_trailing = is_late(
                inp.tr, ctx.match_duration
            ) and state.score_us < state.score_them
            high_value = (
                world.ball_zone == "final"
                or ctx.opp.detect_archetype() == "elite_goalkeeper"
                or ctx.opp.detect_archetype() == "low_block"
                or world.ball_near_wall
                or late_trailing
                or ctx.counter_press_active
            )
            if high_value:
                ctx.mcts_planner.genome = adapted_genome
                best = ctx.mcts_planner.choose(inp, intents)
                if best is not None:
                    intents = best
                    mcts_note = "mcts"

        wire = [player_intent.to_wire() for player_intent in intents.values()]

        # Deterministic ordering by player id keeps the payload stable.
        wire.sort(key=lambda item: item["playerId"])

        decision = {
            "protocolVersion": observation.get("protocolVersion", "1.0"),
            "gameId": observation.get("gameId", ""),
            "sequence": int(observation.get("sequence", 0)),
            "intents": wire,
        }

        # Explainability (development only, §47).
        if ctx.config.log_explanations and ctx.policy.log.possessions_decided:
            ctx.record_decision(
                {
                    "seq": decision["sequence"],
                    "state": str(tactical_state),
                    "decision": ctx.policy.log.possessions_decided[-1],
                    "mcts": mcts_note,
                }
            )
            if len(ctx.explanations) > 64:
                ctx.explanations = ctx.explanations[-64:]

        # Advance per-match memory.
        ctx.prev_had_control = bool(state.has_control())
        if state.ball.possessing_team == "them":
            if state.ball.possessing_player != ctx.their_possessor_id:
                ctx.their_possessor_id = state.ball.possessing_player
                ctx.their_possession_ticks = 1
            else:
                ctx.their_possession_ticks += 1
        else:
            ctx.their_possessor_id = None
            ctx.their_possession_ticks = 0

        return decision

    def _detect(self, ctx: MatchContext, state: GameState, world: WorldModel) -> TacticalState:
        return detect(state, world, ctx.genome, ctx.prev_had_control)