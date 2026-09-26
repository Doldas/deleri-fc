# Deleri FC — Goal-First Adaptive AI Implementation Plan

## Mission

Build Deleri FC into a highly adaptive, goal-focused Football Babylon team.

The objective is not simply to maximize possession, pass completion, or internal fitness. The objective is to create a team that:

1. creates more genuine scoring opportunities,
2. converts those opportunities into goals,
3. adapts its attacking behaviour to the opponent,
4. remains defensively resilient against dangerous opponents,
5. understands Football Babylon's unique physics, especially walls and high-speed loose balls,
6. uses expensive search/training offline and a fast deterministic policy at match time.

Do not discard the existing architecture. Extend it systematically.

---

## 0. Non-negotiable engineering rules

### 0.1 Never optimize blind

For every meaningful change:

```text
HYPOTHESIS
    ↓
ONE DISCRIMINATING TEST
    ↓
RUN
    ↓
MEASURE
    ↓
KEEP / REJECT
    ↓
REGRESSION TEST
```

Do not make a large collection of tactical changes and then guess which one helped.

### 0.2 Instrument before speculating

When two policies have the same real-engine result, do not conclude that the policy is ineffective.

Trace:

```text
L1 genome/config differs
L2 emitted intent differs
L3 player trajectory differs
L4 ball trajectory differs
L5 possession differs
L6 shot event differs
L7 goal differs
L8 match result differs
```

Find the first level where the signal disappears.

### 0.3 Internal simulation != authoritative engine

Always label results as one of:

- internal simulator
- scenario lab
- real Football Babylon engine

Never use internal fitness as proof of real-engine performance.

### 0.4 Preserve determinism

Every experiment must record:

- random seed
- strategy/genome hash
- artifact hash
- container/image version
- engine version
- opponent
- side/kickoff side
- scenario
- command
- result

### 0.5 No LLM/network dependency during tournament matches

All expensive reasoning, MCTS, evolution and training happen offline.

Runtime must be deterministic, fast and self-contained.

---

# 1. Current architecture: preserve and strengthen

The repository already contains valuable components:

```text
GameState
WorldModel
OpponentModel
TacticalState
PolicyController
MCTS
Evolution
Scenario Lab
Wall model
Shot physics
Goalkeeper model
Real-engine campaign tooling
```

Do not rewrite these wholesale.

Connect them into one coherent attacking loop:

```text
OBSERVATION
    ↓
WORLD MODEL
    ↓
OPPONENT BELIEF ──────→ GOALKEEPER MODEL
    ↓                         ↓
    └──────────→ THREAT MAP ←┘
                    ↓
              ATTACKING STATE
                    ↓
             ACTION CANDIDATES
                    ↓
          SHORT-HORIZON MCTS
                    ↓
             TEAM COORDINATION
                    ↓
                 EXECUTE
                    ↓
              OBSERVE RESULT
                    ↓
                 ADAPT
```

---

# 2. Phase 1 — Attack observability

Before changing tactical behaviour, add a comprehensive attacking telemetry layer.

Create a development-only event recorder.

At minimum record:

```text
tick
decision_time
game_time
score
ball position
ball velocity
ball possession
player positions
player roles
opponent positions
opponent estimated roles
tactical state
opponent profile
goalkeeper position
goalkeeper estimated reach
emitted action
target
power
pass target
shot target
wall contact
rebound prediction
```

Counters:

```text
final_third_entries
high_value_entries
shots
shots_on_target
blocked_shots
goalkeeper_saves
goals
keeper_displacements
keeper_shift_events
wall_attacks
wall_rebounds
rebound_recoveries
second_shots
dangerous_turnovers
successful_progressions
1v1_attacks
1v1_wins
cutbacks
third_man_actions
```

Also log the propagation chain:

```text
genome
→ policy
→ intent
→ player movement
→ ball movement
→ possession
→ shot
→ goal
```

Make the dev-only event log persistent or bind-mounted so Docker cannot silently discard it.

---

# 3. Phase 2 — Build the Threat Map

The team currently reasons heavily about space.

Extend this into a goal-threat model.

Create something equivalent to:

```python
ThreatMap
```

For relevant candidate positions/actions estimate:

```text
shot_probability
goal_probability
pass_probability
rebound_probability
keeper_reach
defender_reach
turnover_probability
future_shot_probability
```

The purpose is to answer:

> Where can we move the ball or a player so that the probability of scoring increases?

Not merely:

> Where is there free space?

Start with:

```text
ThreatValue =
    w_goal * P(goal)
  + w_rebound * P(rebound)
  + w_second_shot * P(second_shot)
  + w_future * P(future_high_quality_shot)
  - w_turnover * P(dangerous_turnover)
```

Expose the weights for later evolution.

---

# 4. Phase 3 — Goalkeeper model

Create a first-class:

```text
GoalkeeperModel
```

Track:

```text
position
lateral position
distance from goal
distance from defensive fifth
predicted movement
predicted lateral reach
predicted arrival time
vulnerable goal regions
current keeper bias
```

Distinguish:

```text
keeper can comfortably save
keeper can dive
keeper cannot reach
keeper can be displaced
keeper is outside useful handling area
```

Do not reduce the goalkeeper to a single "good/bad" variable.

---

# 5. Phase 4 — Replace "can shoot?" with "what creates the most goal threat?"

Do not make shooting a simple binary gate.

Generate candidate actions:

```text
DIRECT_SHOT
WALL_SHOT
DRIBBLE
PASS
CUTBACK
CROSS
SWITCH
ONE_TWO
THIRD_MAN_RUN
WALL_PASS
KEEPER_DISPLACEMENT
REBOUND_ATTACK
```

Evaluate each action using predicted outcome:

```text
ActionValue =
    expected_goal_value
  + rebound_value
  + next_shot_value
  + territorial_value
  - turnover_risk
  - counterattack_risk
```

A shot is good when its expected scoring value is better than the best realistic alternative.

---

# 6. Phase 5 — Goalkeeper manipulation

Do not respond to a strong goalkeeper by simply shooting less.

Teach the attack to manipulate the goalkeeper.

Examples:

```text
keeper central
→ attack left
→ keeper shifts left
→ attack right / far-side shot
```

or:

```text
carrier threatens near side
→ keeper commits
→ second attacker attacks far side
```

Create:

```text
KEEPER_DISPLACEMENT
```

Measure:

```text
keeper displacement
time required
resulting shot quality
resulting goal probability
```

A strong goalkeeper should cause Deleri to create better shooting geometry, not simply surrender.

---

# 7. Phase 6 — Rebounds and second balls

Football Babylon has unusual physics:

- high-speed kicks
- walls
- deterministic ball decay
- loose-ball recovery rules
- no conventional corners/throw-ins

Create:

```text
ReboundModel
```

Predict:

```text
shot/save
→ rebound location
→ first reachable teammate
→ opponent arrival
→ second-shot probability
```

Track:

```text
rebound_recoveries
second_shots
second_shot_goals
```

A low-probability direct shot can still be valuable if it creates a high-quality second-ball situation.

---

# 8. Phase 7 — Wall attack as a real attacking weapon

Treat walls as a tactical resource, not merely emergency passing surfaces.

Build explicit tactics:

```text
WALL_PASS
WALL_ESCAPE
WALL_SHOT
WALL_REBOUND
WALL_SWITCH
WALL_TRAP
```

For every wall action calculate:

```text
contact point
post-bounce trajectory
receiver/rebound position
opponent arrival
goalkeeper position
next action
```

Validate all wall predictions against the authoritative engine.

If the analytical model disagrees with the engine, the engine wins.

---

# 9. Phase 8 — Future receiver / pass-and-move model

Babylon's high-speed ball physics make ordinary short-passing assumptions dangerous.

Extend the existing collection-point logic into:

```text
FutureReceiverModel
```

Use:

```text
pass to predicted future collection point
```

instead of always passing to the teammate's current position.

Train coordinated patterns:

```text
pass
→ receiver moves into collection point
→ passer moves immediately
→ third-man option opens
```

Implement:

```text
THIRD_MAN_RUN
PASS_AND_MOVE
RECEIVER_LEAD
```

where physics permits.

---

# 10. Phase 9 — Striker movement should optimize future shot quality

Do not make the striker simply move toward goal.

Evaluate candidate striker positions using:

```text
future_shot_angle
goalkeeper_reach
defender_interference
pass_collection_probability
rebound_position
distance to next action
```

The striker should sometimes move laterally or away from goal if that creates a better future shot.

Examples:

```text
move sideways
→ defender follows
→ lane opens
```

```text
move away
→ defender follows
→ teammate attacks gap
```

```text
hold far-side position
→ keeper shifts near side
→ receive / finish
```

---

# 11. Phase 10 — Real opponent adaptation

The OpponentModel must become a real attacking input.

Do not blindly mutate the entire genome at runtime.

Create:

```text
OpponentBelief
```

with probabilities such as:

```text
high_press
low_block
counter
direct
possession
wall_specialist
chaos
elite_goalkeeper
unknown
```

Update the belief online.

Then derive:

```text
AttackPlan
DefencePlan
```

from that belief.

---

# 12. Opponent-specific attacking behaviour

## HIGH_PRESS

```text
increase width
use wall escapes
use future receiver positions
reduce dangerous central carries
attack space behind press
```

## LOW_BLOCK

```text
increase width
move defenders laterally
create 1v1s
use cutbacks
use keeper displacement
use wall attacks
avoid pointless central shots
```

## COUNTER SPECIALIST

```text
retain rest defence
attack with controlled numbers
avoid losing ball centrally
attack weak-side space
```

## POSSESSION OPPONENT

```text
press passing lanes
counter quickly after recovery
attack before their shape resets
```

## DIRECT OPPONENT

```text
protect central transition
keep defensive depth
attack second balls
```

## WALL SPECIALIST

```text
deny wall lanes
anticipate rebound
force play away from preferred wall
counter wall usage
```

## ELITE GOALKEEPER

```text
manipulate keeper
create wider shooting angles
attack rebounds
use second attackers
avoid low-value central shots
```

---

# 13. Phase 11 — Move MCTS toward scoring decisions

Keep the existing MCTS.

Move its valuable search budget toward:

```text
final third
1v1
low block
keeper displacement
wall attack
rebound
high-value shot
late-game attack
```

Use less search on routine build-up.

Trigger tactical MCTS when, for example:

```text
ball.x > FINAL_THIRD_X
OR keeper_displacement_opportunity
OR 1v1
OR low_block
OR wall_attack
OR trailing_late
```

Search abstract team actions:

```text
shoot
pass
carry
wall pass
wall shot
switch
cutback
keeper manipulation
rebound attack
```

Do not search every microscopic movement.

---

# 14. Phase 12 — Team-level coordination

Every attacking action must account for teammate availability.

Before shooting/passing, estimate:

```text
supporting_players
rebound attackers
rest defence
counterpress coverage
next action
```

Maintain:

```text
attackers
support
rest defence
```

Do not send all four outfield players toward the ball.

---

# 15. Phase 13 — Tactical score/time modes

Create explicit modes.

## WINNING_LATE

```text
risk ↓
possession security ↑
counterattack readiness ↑
unnecessary shooting ↓
defensive transition protection ↑
```

## DRAWING_LATE

```text
final-third occupation ↑
shot threshold ↓
rebound hunting ↑
attacking numbers ↑ moderately
```

## LOSING_LATE

```text
risk ↑↑
width ↑
verticality ↑
shot threshold ↓
counterpress ↑↑
attacking numbers ↑
```

These should be explicit tactical modes, not only genome multipliers.

---

# 16. Phase 14 — Scenario corpus

Expand the scenario laboratory.

## Attacking

```text
final_third_possession
central_2v2
wide_2v1
wide_1v1
cutback
cross
keeper_shift_left
keeper_shift_right
keeper_out_of_position
remaining_1v1_gk
elite_goalkeeper
rebound_after_save
blocked_shot_rebound
wall_rebound_finish
wall_escape
third_man_run
high_press_escape
low_block_central
low_block_wide
late_equalizer
late_winner
```

## Defensive

```text
kickoff_defend
opponent_1v1
opponent_counter
opponent_wall_attack
opponent_high_press
opponent_parkbus
opponent_possession
opponent_direct
```

Every scenario needs:

```text
deterministic seed
initial state
expected measurable objectives
repeat count
fitness output
```

---

# 17. Phase 15 — Train against an opponent population

Do not evolve against one opponent.

Population:

```text
aggressive/high_press
early_scorer
deep_block
parkbus
counter
possession
direct
wall_specialist
chaotic
striker_heavy
elite_goalkeeper
adaptive
Hall_of_Fame bots
```

Use both:

```text
scenario fitness
+
full-match fitness
```

Do not allow a policy to dominate solely by exploiting one deterministic fixture.

---

# 18. Phase 16 — Multi-objective fitness

Track:

```text
goals
goal_difference
shots
shots_on_target
shot_quality
final_third_entries
high_value_entries
keeper_displacements
wall_attacks
wall_rebounds
rebound_recoveries
second_shots
dangerous_turnovers
goals_conceded
```

Conceptual priorities:

```text
goals / goal difference       very high
high-quality chances          high
shot quality                  high
dangerous final-third entries medium-high
keeper displacement           medium
rebound creation              medium
wall attack quality           medium
defensive survival            high
possession                    low/secondary
```

Do not let possession dominate the objective.

---

# 19. Phase 17 — Evolution of attacking intelligence

Expose additional genes gradually:

```text
shot_value
keeper_displacement_value
rebound_value
wall_attack_value
final_third_entry_value
one_v_one_value
cutback_value
third_man_value
turnover_risk
transition_risk
pass_progression_value
future_receiver_value
```

Introduce them in groups, not all at once.

Recommended order:

```text
1. shot_value
2. keeper_displacement_value
3. rebound_value
4. wall_attack_value
5. future_receiver_value
6. third_man_value
7. transition_risk
```

Maintain Hall of Fame and population diversity.

---

# 20. Phase 18 — Elite-opponent stress testing

For every candidate:

```text
vs normal
vs high press
vs low block
vs counter
vs wall specialist
vs elite GK
vs adaptive
```

Record full metrics.

Do not promote candidates solely for beating weak opponents.

---

# 21. Phase 19 — Real-engine validation

Every important candidate must be tested in the authoritative Football Babylon engine.

Minimum battery:

```text
20+ matches vs reference
20+ vs reference-strikers
20+ vs slapstick
20+ vs strongest available opponent
```

Test both kickoff sides.

Use deterministic recorded seeds.

Compare:

```text
goals
goals conceded
shots
shots on target
possession
dangerous turnovers
final-third entries
wall events
rebound events
```

---

# 22. Phase 20 — Diagnose propagation before further evolution

If two candidates produce identical real-engine outcomes:

**STOP EVOLVING FOR THAT TEST.**

Run propagation instrumentation.

Determine:

```text
Did intents differ?
Did movement differ?
Did ball trajectories differ?
Did possession differ?
Did shots differ?
Did goals differ?
```

Only continue evolution after the first blocked link is understood.

---

# 23. Phase 21 — Policy distillation

Only after the tactical system demonstrates real-engine improvement:

```text
offline MCTS
+
scenario training
+
self-play
+
evolution
+
opponent population
        ↓
high-quality trajectories
        ↓
distilled policy
        ↓
fast deterministic runtime bot
```

The tournament bot should not require expensive search unless runtime performance has been proven safe.

---

# 24. Phase 22 — Regression protection

Every discovered improvement becomes a regression test.

Examples:

```text
elite goalkeeper scenario
→ must create a viable attacking alternative

low block
→ must preserve width

high press
→ must not collapse team shape

wall attack
→ predicted rebound agrees with engine

shot target
→ remains inside goal opening

keeper displacement
→ measured movement matches prediction

future receiver
→ receiver reaches collection region

extreme genome
→ intent propagation remains observable
```

Run the complete test suite after every major tactical change.

---

# 25. Development order — follow this order

## Stage A — Observe

1. Add attack event logger.
2. Add propagation chain.
3. Add attacking metrics.
4. Verify real-engine telemetry.

## Stage B — Understand

5. Implement ThreatMap.
6. Implement GoalkeeperModel.
7. Implement ReboundModel.
8. Validate each against the authoritative engine.

## Stage C — Attack

9. Replace binary shot logic with action-value evaluation.
10. Add keeper displacement.
11. Add rebound/second-ball attacks.
12. Improve future-receiver passing.
13. Improve striker future-shot positioning.
14. Strengthen wall attack.

## Stage D — Adapt

15. Connect OpponentModel to AttackPlan.
16. Add opponent belief probabilities.
17. Add explicit opponent-specific attacking modes.
18. Add score/time tactical modes.

## Stage E — Search

19. Move MCTS budget toward high-value attacking states.
20. Search abstract attacking actions.
21. Validate MCTS decisions in the real engine.

## Stage F — Learn

22. Expand scenarios.
23. Build opponent population.
24. Add multi-objective fitness.
25. Evolve new attacking genes.
26. Maintain Hall of Fame and diversity.

## Stage G — Harden

27. Real-engine stress suite.
28. Regression tests.
29. Deterministic replay.
30. Distill winning policy.
31. Re-test final artifact.
32. Build final tournament image.

---

# 26. Required OpenCode reporting format

At the end of every iteration:

```text
HYPOTHESIS:
...

TEST:
...

EXPECTED:
...

OBSERVED:
...

REAL ENGINE OR INTERNAL:
...

METRICS:
...

FIRST BROKEN PROPAGATION LEVEL:
...

CHANGE:
...

REGRESSION:
...

NEXT ACTION:
...
```

Do not end with:

```text
"What would you like me to do next?"
```

when there is an obvious next engineering step.

Continue autonomously.

Only ask the user when a genuinely external decision is required.

---

# 27. Definition of success

Success is not:

```text
tests pass
```

or:

```text
internal fitness improved
```

Success means measurable real-engine improvement:

```text
more goals scored
better goal difference
more high-quality chances
more successful final-third attacks
better performance against elite goalkeepers
better adaptation to different opponents
fewer dangerous turnovers
strong defensive transition
effective wall usage
effective rebound attacks
```

The final Deleri FC policy should be:

```text
GOAL-FIRST
ADAPTIVE
PHYSICS-AWARE
OPPONENT-AWARE
WALL-AWARE
REBOUND-AWARE
SEARCH-ASSISTED
DEFENSIVELY RESILIENT
DETERMINISTIC
FAST
```

---

# 28. Final principle

Do not try to make Deleri FC merely "play beautiful football."

Do not maximize possession for its own sake.

Do not maximize pass count.

Do not blindly maximize aggression.

Make the team maximize **scoring opportunity quality**, while understanding when the risk of losing the ball is unacceptable.

The attacking brain should continuously ask:

> What action most increases our probability of scoring next?

When the answer is not shooting:

> What action creates the best future shot?

Against an elite goalkeeper:

> How do we change the geometry so that the goalkeeper is no longer in the optimal position?

Against a strong opponent:

> How much attacking risk can we take without exposing a dangerous transition?

Against a radically different opponent:

> What evidence have we observed, and what tactical behaviour should change because of it?

Build it incrementally, measure every claim, validate against the authoritative engine, and keep the best discovered policies in the Hall of Fame.
