# Football Babylon AI
## Implementation & Requirements Specification

**Version:** 0.1  
**Target:** Football Babylon Team Developer Kit v1.1.0  
**Platform:** Linux / AMD64  
**Primary implementation language:** Python initially  
**Optimization language:** Rust or Go only if profiling proves necessary  
**Architecture:** Hierarchical tactical AI + MCTS + self-play + evolutionary optimization + distilled runtime policy

---

# 1. Project Vision

Build an autonomous Football Babylon team that combines:

1. Principles extracted from elite real-world football.
2. Tactical concepts from other sports where useful, especially hockey.
3. Monte Carlo Tree Search for tactical planning.
4. Self-play and large-scale deterministic simulation.
5. Evolutionary optimization of tactical parameters.
6. Opponent modelling.
7. A final lightweight runtime policy distilled from expensive offline search.

The objective is **not** to reproduce one real-world football team.

The objective is:

> Use elite football as a prior, use Football Babylon physics as the environment, and let large-scale simulation discover the strongest strategy for the Babylon rules.

The final tournament bot must be fast, deterministic, robust and compliant with the Football Babylon strategy contract.

---

# 2. Core Design Principle

The project must distinguish between:

## 2.1 Real-world football knowledge

Used as:

- tactical priors
- initial policy biases
- feature definitions
- reward design
- role definitions
- pressing concepts
- possession concepts
- transition concepts
- spatial concepts

Sources should include, where licensing permits:

- UEFA Champions League
- FIFA World Cup
- UEFA European Championship
- major international tournaments
- top domestic leagues
- elite clubs
- publicly available event/tracking datasets
- reputable tactical analysis

The system must not blindly copy formations.

It should extract principles such as:

- spacing
- width
- depth
- overloads
- pressing triggers
- counterpressing
- rest defence
- progression
- third-man concepts
- defensive compactness
- transition speed
- risk management

---

# 3. Babylon-Specific Tactical Knowledge

Football Babylon must be treated as its own sport.

The following properties must be first-class tactical concepts:

- walls
- ball rebounds
- no throw-ins
- no corners
- no goal kicks
- no offside
- deterministic physics
- five-player teams
- 10 Hz decisions
- 60 Hz simulation
- fixed pitch
- fixed movement physics

The system must therefore discover tactics that would not exist in ordinary football.

Examples:

- wall passes
- wall-assisted progression
- wall-assisted shots
- rebound traps
- wall-based pressure escapes
- wall-based defensive containment
- deliberate possession against the wall
- using the wall as an additional "virtual player"

These are hypotheses to test, not hard-coded assumptions.

---

# 4. Authoritative Football Babylon Constraints

The implementation MUST respect the Football Babylon engine.

## 4.1 Match

Each team has:

- 1 goalkeeper
- 4 outfield players

Pitch:

- 60 × 40 metres
- 6 metre goal width

The team makes decisions at:

- 10 Hz

The engine simulates at:

- 60 Hz

Coordinates are normalized so our team always attacks toward positive X.

The engine resolves authoritative state.

The bot must not directly modify:

- score
- possession
- player position
- ball position
- velocity
- cooldowns
- animation
- authoritative game state

The bot requests intents only.

---

# 5. Actions

The only valid actions are:

```text
none
pass
shoot
clear
tackle
slap
```

Only the player in possession may:

```text
pass
shoot
clear
```

Those actions require a target.

Outfield players may request:

```text
tackle
slap
```

Goalkeeper behaviour is largely engine-controlled.

The AI must never invent unsupported actions.

---

# 6. Important Engine Physics

The AI must explicitly model the following.

## Movement

Maximum running speed:

```text
8 m/s
```

Maximum dribbling speed:

```text
6.4 m/s
```

Movement turns immediately toward the latest absolute target.

There is no acceleration model.

Controlled ball position:

```text
0.65 m ahead of player
```

---

## Possession

New possession is protected from outfield tackles for:

```text
0.25 s
```

Loose-ball recovery occurs within:

```text
0.9 m
```

when the ball speed is no more than:

```text
5 m/s
```

Closest eligible player wins a loose ball.

Equal distances resolve by player ID.

The AI must always read:

```text
ball.possessingTeam
ball.possessedBy
```

together.

---

# 7. Tackling Model

A tackle can succeed within:

```text
1.15 m
```

of an opponent-controlled ball after protection expires.

Close tackle:

```text
distance < 0.8 m
```

transfers possession directly.

Slide tackle:

```text
0.8 m <= distance <= 1.15 m
```

knocks the ball loose at:

```text
10 m/s
```

Slide tackler becomes unable to move/act for:

```text
1.05 s
```

Both players receive:

```text
0.3 s
```

action cooldown.

There are:

- no random tackle failures
- no fouls

Therefore tackle decisions should be deterministic and geometry-driven.

---

# 8. Slap Model

Slap within:

```text
1.15 m
```

causes:

```text
0.3 s stagger
0.25 m knockback
```

Within:

```text
0.75 m
```

and facing within:

```text
60 degrees
```

produces:

```text
1.2 s knockdown
```

and spills a carried ball.

Recovered players receive:

```text
1.25 s knockdown immunity
```

Slapper cooldown:

```text
1.5 s
```

Same-tick slaps resolve simultaneously.

All outcomes are deterministic.

The AI should model slap opportunities separately from tackles.

---

# 9. Goalkeeper Model

Inside the defensive fifth:

A goalkeeper can catch a loose ball when its swept path comes within:

```text
1.65 m
```

Shot save reach:

```text
< 0.8 m
```

keeps goalkeeper upright.

From:

```text
0.8–1.65 m
```

the goalkeeper dives and remains grounded:

```text
0.9 s
```

Inside the defensive fifth, the goalkeeper automatically takes an opponent-controlled ball within:

```text
1.65 m
```

The goalkeeper has no explicit save/handle action.

When in possession, goalkeeper may:

```text
pass
shoot
clear
```

After holding for:

```text
1.25 s
```

the engine automatically distributes the ball at:

```text
20 m/s
```

toward a wide teammate.

The AI must model this forced distribution.

---

# 10. Ball Physics

Power:

```text
0.0–1.0
```

maps to initial ball speed:

```text
12–26 m/s
```

Ball velocity is multiplied by:

```text
0.992
```

each engine tick.

---

# 11. Wall Physics

Walls:

- retain tangential speed
- return 75% of perpendicular speed

This is a critical feature.

The AI must have a `WallModel` capable of predicting:

```text
ball → wall
wall → rebound
rebound position
rebound velocity
interception opportunities
```

The wall must be represented as a tactical resource.

---

# 12. Tactical Architecture

Use a hierarchical architecture.

```text
Game State
    |
    v
Perception
    |
    v
World Model
    |
    +----------------+
    |                |
    v                v
Tactical Brain   Opponent Model
    |
    v
MCTS Planner
    |
    v
Team Intent
    |
    v
Role Assignment
    |
    v
Player Actions
    |
    v
Babylon Engine
```

---

# 13. Game State Representation

Create a compact immutable representation:

```python
GameState
```

containing at minimum:

```text
time
score
our possession
opponent possession

ball:
    x
    y
    vx
    vy
    possessing_team
    possessing_player

players:
    id
    team
    role
    x
    y
    target
    facing
    can_act
    cooldown
    grounded
    staggered
    knockdown_state

derived:
    pressure
    space
    passing_lanes
    wall_options
    defensive_shape
    attacking_shape
    transition_state
```

The state representation must be compact enough for large-scale simulation.

---

# 14. World Model

Create a derived tactical representation.

The World Model should calculate:

```text
ball zone
pressure zones
free space
danger zones
passing lanes
support lanes
progression lanes
shooting lanes
defensive cover
wall rebound lanes
interception points
```

The World Model must be deterministic.

---

# 15. Space Model

Implement a spatial control model.

For each player calculate:

```text
distance_to_ball
distance_to_goal
distance_to_teammates
distance_to_opponents
nearest_opponent
nearest_teammate
available_space
pressure
support_value
progression_value
defensive_value
```

Create a continuous approximation of:

```text
our_control(x, y)
opponent_control(x, y)
```

This should be used by:

- movement
- passing
- pressing
- MCTS
- evaluation

---

# 16. Passing Model

For each potential pass calculate:

```text
pass_success_probability
progression
receiver_space
receiver_pressure
turnover_risk
goal_threat
support_after_pass
defensive_security_after_pass
```

The AI should prefer passes that improve the team state, not merely successful passes.

---

# 17. Wall Passing

Implement:

```text
WallPassCandidate
```

with:

```text
origin
wall_contact
rebound
receiver
opponent_interception_probability
progression
risk
```

A wall pass should compete directly with ordinary passes.

Example:

```text
direct_pass_value
wall_pass_value
dribble_value
```

MCTS decides which is best.

---

# 18. Wall Shooting

Implement prediction for:

```text
direct_shot
wall_shot
```

The system should evaluate whether a wall rebound can produce a better shot angle than a direct shot.

Do not assume wall shots are good.

Let simulation determine their value.

---

# 19. Hockey-Inspired Concepts

Only transferable concepts should be considered.

Potential concepts:

### Wall cycling

Use wall + player movement to retain possession.

### Pressure escape

Use the wall as an additional escape route.

### Rebound anticipation

Position a second player where a rebound is likely to arrive.

### Defensive wall trap

Force the opponent toward a wall and reduce available exits.

### Support triangle

Maintain multiple recovery options around the ball.

### Zone occupation

Control important areas rather than following players everywhere.

All concepts must be experimentally validated.

---

# 20. Tactical State Machine

The team must operate in high-level tactical states:

```text
KICKOFF
BUILD_UP
PROGRESSION
ATTACK
FINAL_ATTACK
DEFENSIVE_TRANSITION
COUNTERPRESS
MID_BLOCK
HIGH_PRESS
LOW_BLOCK
POSSESSION_CONTROL
GAME_MANAGEMENT
```

State transitions must depend on:

```text
possession
ball location
pressure
score
time
opponent shape
player availability
space
transition opportunity
```

---

# 21. Offensive Principles

The initial strategy should prioritize:

1. Width
2. Depth
3. Support
4. Progression
5. Numerical/positional overload
6. Passing triangles
7. Ball security
8. Counterpress readiness
9. Goal threat
10. Wall exploitation

Avoid static formations.

Formation is a starting structure, not the tactical objective.

---

# 22. Defensive Principles

Prioritize:

1. Compactness
2. Central protection
3. Pressing triggers
4. Cover
5. Pressure + support
6. Recovery after failed press
7. Transition protection
8. Wall containment
9. Shot denial
10. Goalkeeper protection

---

# 23. Pressing Triggers

Implement trigger-based pressing.

Potential triggers:

```text
opponent receives facing own goal
opponent receives near wall
opponent takes poor position
opponent loses support
opponent becomes isolated
opponent enters predictable passing lane
loose ball becomes recoverable
```

Do not use:

```python
if opponent_has_ball:
    everyone_press()
```

Pressing must be coordinated.

---

# 24. Counterpress

Immediately after losing possession:

Calculate:

```text
can_recover?
nearest_player
second_pressing_player
cover_player
rest_defence_player
opponent_escape_routes
```

If recovery probability is high:

```text
COUNTERPRESS
```

Otherwise:

```text
RECOVER_SHAPE
```

Never allow all four outfield players to blindly chase the ball.

---

# 25. Rest Defence

While attacking, always maintain defensive security.

At least one or more players should be positioned to:

```text
cover central counter
protect goalkeeper
intercept direct transition
recover loose ball
```

Exact number should be discovered through simulation.

---

# 26. MCTS

Implement Monte Carlo Tree Search as an offline/high-compute tactical planner.

Basic algorithm:

```text
selection
expansion
simulation
backpropagation
```

Use:

```text
UCT / PUCT
```

as appropriate.

Prefer PUCT if a tactical prior policy is available.

---

# 27. MCTS Search Horizon

Do not search only one 100 ms tick.

Initial target:

```text
2–5 seconds
```

with adaptive horizon.

Use shorter searches in:

```text
high-frequency defensive situations
```

and longer searches in:

```text
possession/build-up situations
```

The horizon must be configurable.

---

# 28. MCTS Action Abstraction

Do not expose every possible coordinate as a separate tree action.

Use tactical actions:

```text
PASS_SUPPORT
PASS_FORWARD
PASS_WIDE
PASS_SPACE
DRIBBLE_FORWARD
DRIBBLE_SPACE
MOVE_SUPPORT
MOVE_WIDTH
MOVE_DEPTH
PRESS
CONTAIN
RECOVER
COUNTERPRESS
WALL_PASS
WALL_CONTROL
SHOOT
CLEAR
TACKLE
SLAP
```

The low-level controller converts tactical actions to engine-compatible targets.

---

# 29. Joint Action Problem

Because four outfield players act as a team, avoid independently optimizing four unrelated actions.

Use:

```text
TeamAction
```

first.

Example:

```text
TeamAction:
    ball_player = PASS_FORWARD
    player_2 = SUPPORT
    player_3 = WIDTH
    player_4 = COVER
```

Then optionally refine individual actions.

This dramatically reduces the MCTS branching factor.

---

# 30. MCTS Prior

MCTS should combine:

```text
elite_football_prior
+
learned_policy_prior
+
Babylon_specific_prior
+
opponent_model
```

Conceptually:

```text
P(action | state)
=
football_prior
+
babylon_prior
+
learned_prior
+
opponent_adaptation
```

The priors must guide search but never completely prevent exploration.

---

# 31. Reward Function

Do NOT use only:

```text
win = +1
loss = -1
```

Use shaped tactical reward.

Initial components:

```text
goal
goal_conceded

territorial_progression
possession_quality
space_created
chance_creation
shot_quality
successful_press
counterpress_recovery
defensive_compactness
passing_quality
wall_progression

dangerous_turnover
broken_shape
failed_press
bad_shot
unnecessary_risk
```

Reward weights must be configurable.

---

# 32. Score and Time Awareness

The reward model must depend on match state.

When leading late:

```text
risk ↓
possession security ↑
counterattack value ↑
```

When losing late:

```text
risk ↑
verticality ↑
shot generation ↑
press intensity ↑
```

When tied:

Use balanced risk.

---

# 33. Self-Play

Create an automated tournament environment.

Every strategy must play against:

```text
reference opponent
possession bot
pressing bot
direct bot
defensive bot
random baseline
previous champion
current champion
```

Do not optimize against a single opponent.

---

# 34. Evolution

Maintain a population of strategies.

Each individual contains parameters such as:

```yaml
press_intensity
press_trigger_threshold
width
depth
verticality
risk
counterpress_intensity
compactness
passing_risk
shooting_threshold
wall_usage
wall_pass_threshold
wall_shot_threshold
transition_speed
support_distance
defensive_line
```

Evolution operations:

```text
mutation
crossover
selection
elitism
```

Use tournament evaluation.

---

# 35. Fitness Function

Fitness must not be simple win percentage.

Example:

```text
fitness =
    match_results
    + goal_difference
    + chance_quality
    + defensive_quality
    + tactical_consistency
    + opponent_robustness
```

The exact weights must be configurable and tested.

A strategy that dominates one opponent but fails against the rest must not become champion.

---

# 36. Hall of Fame

Maintain historical champions:

```text
champion_001
champion_002
champion_003
...
```

Every new strategy must periodically be tested against the Hall of Fame.

This prevents evolutionary regression.

---

# 37. Diversity Preservation

The evolutionary population should retain different tactical styles.

Do not allow the entire population to converge immediately.

Preserve examples of:

```text
possession
pressing
direct
defensive
wall-heavy
balanced
```

This improves robustness.

---

# 38. Opponent Model

Maintain a lightweight opponent model.

Track:

```text
preferred progression side
press intensity
average defensive shape
passing risk
wall usage
transition behaviour
shooting behaviour
```

Use this to modify tactical priorities during a match.

Do not require expensive online learning.

---

# 39. Offline Training Pipeline

Required pipeline:

```text
generate strategies
        ↓
simulate matches
        ↓
collect trajectories
        ↓
calculate tactical metrics
        ↓
MCTS search
        ↓
store improved decisions
        ↓
train policy
        ↓
evaluate
        ↓
evolve
        ↓
repeat
```

---

# 40. Policy Distillation

The final tournament bot should not depend on expensive MCTS for every decision.

Use MCTS offline to generate high-quality examples:

```text
state
→ MCTS best action
```

Train a compact policy:

```text
state → action
```

The final runtime becomes:

```text
observe
→ encode
→ policy
→ action
```

MCTS may remain available as a fallback for selected critical states if performance permits.

---

# 41. Runtime Architecture

Target:

```text
10 decisions/sec
```

Per decision:

```text
observe
↓
update world model
↓
detect tactical state
↓
predict opponent
↓
policy
↓
optional limited MCTS
↓
role assignment
↓
validate action
↓
submit intents
```

No blocking network calls.

No external API calls during matches.

No LLM calls during matches.

No nondeterministic external dependencies.

---

# 42. Determinism

The production bot must be deterministic.

All randomness must use an explicitly controlled seed.

MCTS randomness:

```text
seeded
```

Evolution randomness:

```text
seeded
```

Simulation:

```text
reproducible
```

Every experiment must record:

```text
seed
strategy hash
configuration hash
engine version
code version
```

---

# 43. Simulation Infrastructure

Build a high-throughput simulation runner.

Example:

```bash
./simulate \
    --strategy strategy_a \
    --opponent strategy_b \
    --games 10000 \
    --seed 42
```

Parallelize matches across CPU cores.

Example:

```text
CPU 0 → games 1-1000
CPU 1 → games 1001-2000
CPU 2 → games 2001-3000
...
```

---

# 44. Experiment Tracking

Every experiment must produce:

```text
experiment_id
timestamp
git_commit
engine_version
strategy_config
random_seed
opponent_pool
games
wins
draws
losses
goals_for
goals_against
tactical_metrics
```

Store machine-readable JSON.

Prefer SQLite or Parquet for larger datasets.

---

# 45. Tactical Metrics

At minimum calculate:

```text
possession %
territorial control
shots
shots conceded
shot quality
goals
goals conceded

successful passes
failed passes
turnovers

press attempts
successful presses
press failures

counterpress attempts
counterpress recoveries

average team width
average team depth

defensive compactness
support distance

wall passes
successful wall passes
wall possessions
wall progression
wall shots

dangerous turnovers
```

---

# 46. Replay Analyzer

Every match should optionally produce a replay dataset.

Example:

```json
{
  "tick": 1234,
  "time": 20.5,
  "ball": {},
  "players": [],
  "tactical_state": "COUNTERPRESS",
  "action": {},
  "evaluation": {}
}
```

The analyzer should allow us to answer:

```text
Why did we concede?
Why did we lose possession?
Why did MCTS select this action?
Why did a wall pass work?
Why did pressing fail?
```

---

# 47. Explainability

The AI should log a compact tactical explanation:

```text
state:
    opponent_high_press

decision:
    WALL_PASS

reason:
    central lane blocked
    wall lane open
    receiver has space
    recovery probability 0.81
```

This is for development only.

Do not make runtime decisions dependent on natural-language reasoning.

---

# 48. Real Football Data Pipeline

Create a separate data ingestion layer.

```text
raw football data
        ↓
normalization
        ↓
events
        ↓
spatial features
        ↓
tactical patterns
        ↓
priors
```

Do not embed source-specific assumptions into the Babylon engine.

---

# 49. Elite Football Feature Extraction

Look for patterns such as:

```text
pressing triggers
counterpress behaviour
build-up structures
width
support angles
third-man movement
overloads
switches
progression
defensive rest shape
transition behaviour
```

The purpose is to generate tactical priors, not copy formations.

---

# 50. Data Licensing

Every external dataset must have:

```text
source
license
allowed use
download date
version
```

The project must never assume that publicly viewable data is automatically legal to redistribute.

Keep raw datasets separate from the code repository where necessary.

---

# 51. Testing

Create unit tests for:

```text
distance calculations
passing geometry
wall rebound prediction
shot prediction
tackle geometry
slap geometry
space calculation
pressure calculation
role assignment
state transitions
reward calculation
MCTS
policy
```

---

# 52. Physics Validation

Before tactical AI is trusted, validate our internal models against the authoritative Babylon engine.

Example:

```text
prediction:
wall rebound → (x,y,vx,vy)

engine:
actual rebound → (x,y,vx,vy)
```

The model must remain within configurable tolerance.

If the model disagrees with the engine, the engine wins.

---

# 53. Regression Tests

Maintain fixed scenarios:

```text
wall pass
wall shot
counterpress
1v1 tackle
slide tackle
slap
loose ball
goalkeeper save
goalkeeper possession
kickoff
goal restart
late-game defence
late-game attack
```

Every code change runs these scenarios.

---

# 54. Performance Requirements

Development target:

```text
10,000+ simulated matches/hour
```

on a modern multi-core development machine, subject to engine overhead.

MCTS should be benchmarked independently.

Target metrics:

```text
states/sec
simulations/sec
decisions/sec
CPU usage
memory
```

Do not optimize prematurely.

Profile first.

---

# 55. Development Phases

## Phase 1 — Engine Integration

Deliver:

```text
GameState
Action API
player control
basic strategy
logging
```

No ML.

---

## Phase 2 — Tactical Baseline

Deliver:

```text
roles
formation
space
passing
pressing
transitions
counterpress
```

---

## Phase 3 — Wall Intelligence

Deliver:

```text
WallModel
rebound prediction
wall pass
wall shot
wall pressure
```

---

## Phase 4 — MCTS

Deliver:

```text
TeamAction
search tree
rollout
reward
opponent model
```

---

## Phase 5 — Self Play

Deliver:

```text
simulation runner
tournament runner
metrics
experiment tracking
```

---

## Phase 6 — Evolution

Deliver:

```text
population
mutation
crossover
selection
Hall of Fame
diversity preservation
```

---

## Phase 7 — Elite Football Priors

Deliver:

```text
football data pipeline
tactical feature extraction
prior policy
```

---

## Phase 8 — Policy Distillation

Deliver:

```text
MCTS dataset
policy model
runtime inference
```

---

## Phase 9 — Tournament Hardening

Deliver:

```text
official validator
practice arena
regression suite
performance benchmark
submission ZIP
```

---

# 56. Definition of Done

The project is considered ready when:

- It passes the official validator.
- It obeys the authoritative action contract.
- It never modifies authoritative game state.
- It operates within 10 Hz decision constraints.
- It is deterministic under a fixed seed.
- It survives malformed/unexpected states.
- It has no external runtime dependencies.
- It has automated regression tests.
- It has automated simulation tests.
- It has been evaluated against multiple tactical opponents.
- It has been evaluated against Hall of Fame strategies.
- Wall tactics have been experimentally evaluated.
- The final policy performs at least as well as the expensive development strategy on held-out simulations.

---

# 57. Non-Goals

Do NOT initially build:

- a giant end-to-end neural network
- an LLM-driven player controller
- real-time internet data during matches
- a perfect FIFA simulator
- a copied Manchester City/Liverpool/Barcelona/etc. formation
- individual-player RL from scratch
- full physics reproduction outside the Babylon engine

The project should remain:

```text
small
fast
measurable
reproducible
experiment-driven
```

---

# 58. Key Engineering Principle

When there is a conflict between:

```text
real-world football intuition
```

and:

```text
measured Babylon simulation
```

the simulation wins.

When there is a conflict between:

```text
our internal physics model
```

and:

```text
authoritative Babylon engine
```

the engine wins.

When there is a conflict between:

```text
MCTS
```

and:

```text
hard engine rules
```

the engine rules win.

---

# 59. Recommended Repository

```text
football-babylon-ai/

README.md
SPEC.md

src/
  engine/
  perception/
  world_model/
  tactics/
  roles/
  wall/
  mcts/
  evolution/
  opponent/
  policy/
  runtime/

data/
  priors/
  experiments/

sim/
  runners/
  tournaments/
  scenarios/

tests/
  unit/
  physics/
  tactical/
  regression/

tools/
  analyze/
  replay/
  benchmark/
  tournament/

configs/
  baseline.yaml
  mcts.yaml
  evolution.yaml
  reward.yaml

artifacts/
  champions/
  policies/
  reports/
```

---

# 60. Initial Baseline Strategy

Before MCTS exists, implement a competent baseline:

```text
formation:
    1 goalkeeper
    2 defenders
    1 midfielder
    1 attacker

principles:
    maintain width
    maintain depth
    support ball
    protect centre
    press selectively
    counterpress when recoverable
    otherwise recover shape
    avoid unnecessary turnovers
    use walls when advantageous
```

This baseline becomes:

```text
BASELINE_001
```

and is the first opponent in the evolutionary environment.

---

# 61. First MCTS Experiment

The first experiment should be intentionally small.

State:

```text
ball
5 players
opponents
score
time
tactical state
```

Action abstraction:

```text
pass
move
shoot
press
recover
wall_pass
```

Horizon:

```text
2 seconds
```

Run:

```text
1000 matches
```

Compare:

```text
baseline
vs
baseline + MCTS
```

Measure:

```text
win rate
goal difference
turnovers
shots
dangerous attacks
counterpress recoveries
wall usage
```

Do not add neural networks until this experiment works.

---

# 62. Second MCTS Experiment

Introduce:

```text
football tactical priors
```

Compare:

```text
MCTS
vs
MCTS + elite football prior
```

Then introduce:

```text
Babylon wall prior
```

Compare:

```text
MCTS + football
vs
MCTS + football + Babylon
```

This lets us determine whether each component actually contributes.

---

# 63. Evolution Experiment

Create:

```text
population = 32
```

Parameters:

```text
press
width
depth
risk
verticality
counterpress
wall_usage
shoot_threshold
pass_risk
```

Run:

```text
20–50 generations
```

Each generation:

```text
evaluate
select
mutate
crossover
preserve elite
```

Maintain Hall of Fame.

---

# 64. Final Target Architecture

```text
                  REAL FOOTBALL
                       |
                       v
              ELITE TACTICAL PRIORS
                       |
                       |
                       v
              +------------------+
              |  TACTICAL MODEL  |
              +--------+---------+
                       |
                       v
                 BABYLON STATE
                       |
          +------------+-------------+
          |                          |
          v                          v
   OPPONENT MODEL              WALL MODEL
          |                          |
          +------------+-------------+
                       |
                       v
                 MCTS PLANNER
                       |
                       v
                 SELF PLAY
                       |
                       v
                 EVOLUTION
                       |
                       v
                  HALL OF FAME
                       |
                       v
               TRAINED POLICY
                       |
                       v
                DISTILLED BOT
                       |
                       v
             FOOTBALL BABYLON
```

---

# 65. Ultimate Objective

The final system should behave as follows:

```text
Observe the game.

Understand the tactical situation.

Understand space.

Understand the opponent.

Understand the wall.

Use elite-football knowledge as a prior.

Search possible tactical futures.

Choose a coordinated team action.

Execute it deterministically.

Observe the result.

Adapt.

Learn offline from millions of simulations.

Distill the discovered strategy into a fast runtime policy.
```

The goal is not to make the AI "play like a human".

The goal is:

> **Build a system that understands modern football, understands the unique physics of Football Babylon, and is capable of discovering strategies that humans did not explicitly program.**