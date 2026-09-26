# AGENTS.md — Football Babylon AI (Deleri FC)

This file pins the operating rules for any agent (human or AI) working on this
team. It derives from `docs/AISTRATEGI.md` and the authoritative engine rules in
`RULES.md` and `docs/00-game-engine-rules.md`. When a rule here conflicts with
anything else, the **engine**, the **protocol schemas** under `protocol/`, and
`docs/00-game-engine-rules.md` win.

## 0. Engine rules reference

**`docs/00-game-engine-rules.md` is the complete developer-facing rules reference
for the Football Babylon five-a-side engine.** Every agent must read and follow
it. It covers: decision contract (10 Hz decisions, 60 Hz simulation, 20 Hz
snapshots), movement/facing/dribbling, loose-ball control, tackling (close vs
slide), goalkeeper handling/diving, slaps, ball actions/physics, kickoffs/goals/phases,
kit resolution, and scope boundaries. If any strategy guide, viewer animation, or
external summary differs from this document, this document describes the intended
engine behavior.

## 1. Scope boundary (hard constraint)

- **Only modify files inside `My Teams/deleri-fc/`.** Never edit files outside
  that folder (engine, worker, tooling, docs, protocol schemas, viewer, other
  teams).
- The tournament artifact is the built image `football-team-deleri-fc:dev`.
- `team.json`, `tactics.json`, `models.py`, `custom_strategy.py` are the builder
  contract. Prefer editing `src/`, `strategy.py`, `server.py`, `tests/`.

## 2. Authority ordering (from AISTRATEGI §58)

1. Hard engine rules (**`docs/00-game-engine-rules.md`**, `RULES.md`) **always** win over our internal physics model.
2. Measured Babylon simulation wins over real-world football intuition.
3. Hard engine rules win over MCTS.

When in doubt, re-read `docs/00-game-engine-rules.md`, `RULES.md`, and `protocol/*.schema.json`.

## 3. Action contract

- Only requests `none | pass | shoot | clear | tackle | slap`.
- Only the possessor may `pass | shoot | clear`; those need a target (+ power 0..1).
- Outfield players may `tackle | slap`; goalkeepers never `tackle`.
- Echo `protocolVersion`, `gameId`, `sequence` exactly.
- One intent per player id; all numbers finite; move speed / power in 0..1.
- Never touch authoritative state (score, possession, positions, cooldowns,
  animation, time).

## 4. Determinism & runtime rules (AISTRATEGI §41–42)

- The production bot must be deterministic for a fixed seed.
- Seedable randomness only: every module accepts an explicit `rng`.
- No network calls, no external APIs, no LLM calls, no clock-dependent
  randomness, no `time.time()` in decision paths. `time.monotonic()` only for
  optional dev profiling / timeout guards.
- Target <50 ms per decision; hard deadline is 100 ms.

## 5. Per-match memory

- Key all retained state by `gameId`. Initialize at `/v1/matches/start`
  (with the supplied `randomSeed`). Discard at `/v1/matches/end`.
- Never mutate one global match variable across games.

## 6. Strategy principles (non-negotiable)

- Formation is a starting structure, not the objective. Maintain width, depth,
  support, ball security and goal threat.
- Pressing must be trigger-based and coordinated, never
  `if opponent_has_ball: everyone_press()`.
- After losing the ball: counterpress only when recovery probability is high,
  otherwise recover shape. Never send all four outfields chasing the ball.
- While attacking, always keep rest defence so a transition cannot beat us 1v0.
- Always model walls (`WallModel`): tangential speed retained, 75% perpendicular
  restitution. Wall plays are hypotheses to test, never guaranteed good.
- The engine's goalkeeper distribution after 1.25 s must be avoided by passing
  early.

## 7. Training / simulation rules

- Prefer the internal deterministic simulator (`src/sim.py`) for self-play and
  evolution. Validate its numbers against `RULES.md`.
- If the internal model and the engine disagree, **fix the model**.
- Every experiment records: seed, strategy hash, config hash, engine version,
  code version, results, metrics.

## 8. Testing gates (AISTRATEGI §51–53)

- Every change must keep `python -m unittest discover -s tests -v` green.
- Cover at minimum: geometry, pass/shot/wall physics, tackle/slap thresholds,
  space/pressure, role assignment, state transitions, reward, MCTS, policy.
- A change is not done until:
  - `football-team build` succeeds,
  - `football-team validate` reports PASS,
  - unit tests pass,
  - and it was practiced against at least the `reference` opponent.