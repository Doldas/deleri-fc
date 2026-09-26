# Task: Implement a controlled Elite Goalkeeper wiring test

We suspect that the goalkeeper model / shot-value logic may not actually be wired correctly into Deleri FC's attacking decision pipeline.

Do **not** tune the strategy to make this test pass.

First create a deterministic controlled test that proves whether changing only the opponent goalkeeper position materially changes Deleri FC's attacking evaluation and/or decision.

## Objective

Given an otherwise identical attacking game state, change **only** the opponent goalkeeper position.

The goalkeeper position must propagate through:

```text
GameState
→ WorldModel
→ GoalkeeperModel / goalkeeper evaluation
→ Threat / shot evaluation
→ candidate action values
→ selected attacking action
```

We need evidence of where this signal propagates or disappears.

---

## Coordinate assumptions

Deleri FC's normalized coordinate system attacks toward `+X`.

Therefore:

```text
own goal      ≈ x=0
opponent goal ≈ x=60
pitch center  ≈ y=20
```

Verify these assumptions against the authoritative repository implementation before writing the assertions.

If the repository contradicts these coordinates, STOP and document the discrepancy rather than silently adapting the test.

---

# Controlled attacking state

Construct one deterministic attacking state.

Keep ALL of the following identical between test cases:

```text
ball position
ball velocity
ball possessor
Deleri player positions
Deleri player velocities
opponent outfield positions
score
time remaining
game phase
genome
opponent model
RNG seed
tactical state
```

The ONLY variable allowed to change is:

```text
opponent goalkeeper position
```

Choose an attacking position where:

- Deleri has possession;
- the ball carrier is in a realistic dangerous attacking position;
- shooting is a legitimate candidate;
- passing/dribbling may also be legitimate;
- no unrelated rule makes the decision trivially predetermined.

Document the chosen state.

---

# Test A — Central goalkeeper

Set opponent goalkeeper to:

```text
x = 60
y = 20
```

Record:

```text
keeper classification/state
keeper distance from goal
keeper lateral displacement
predicted keeper reach
available goal region
shot target
shot probability / shot value
keeper displacement value
rebound value if available
all generated attacking candidates
value/score of every candidate
final selected action
```

Store this as `result_A`.

---

# Test B — Laterally displaced goalkeeper

Run the EXACT same state.

Change only:

```text
x = 60
y = 24
```

Record the same information as Test A.

Store as `result_B`.

## Required expectation

The attacking evaluation must detect that the goalkeeper geometry changed.

At minimum, one or more goalkeeper-sensitive values must materially differ:

```text
predicted keeper reach
vulnerable side
available goal region
preferred shot target
shot value
keeper-displacement value
candidate ranking
selected action
```

Do **not** require the final selected action to differ if the same action remains objectively best.

The important requirement is that the internal attacking evaluation changes for the correct geometric reason.

For example, this is valid:

```text
A:
shoot far-left
value = 0.41

B:
shoot far-left
value = 0.68
```

The action stayed `SHOOT`, but the goalkeeper position clearly propagated into the evaluation.

Also valid:

```text
A:
DRIBBLE / KEEPER_DISPLACEMENT

B:
SHOOT
```

Not valid:

```text
A and B:
identical goalkeeper evaluation
identical candidate values
identical shot target
identical reasoning
```

That would strongly indicate that goalkeeper position is not properly influencing the attack.

---

# Test C — Goalkeeper advanced from goal

Run the EXACT same state again.

Change only:

```text
x = 52
y = 20
```

Store as `result_C`.

The policy must recognize that this goalkeeper geometry is substantially different from:

```text
x = 60
y = 20
```

Verify the actual Football Babylon definition of the goalkeeper's defensive fifth / handling region before asserting its classification.

Do not invent the boundary.

## Required expectation

Test C should change relevant evaluation such as:

```text
keeper defensive-region status
keeper reach geometry
available shooting geometry
shot target
shot value
keeper-displacement value
candidate ranking
selected action
```

Again, the final action does not necessarily need to change if the same action remains optimal.

But the underlying values must demonstrate that the goalkeeper's new position was actually observed and evaluated.

---

# Add a fourth sanity test

Add:

## Test D — No goalkeeper / impossible keeper observation

Construct the appropriate valid state for the existing engine/model where the goalkeeper is unavailable, absent, or cannot influence the shot.

Do not fabricate an invalid GameState merely to satisfy this test.

Verify that the attacking evaluation responds sensibly and does not accidentally use stale goalkeeper information from the previous test.

This is specifically intended to catch cached/stale GoalkeeperModel state.

---

# Critical anti-cheating requirement

DO NOT implement logic such as:

```python
if keeper.y == 24:
    increase_shot_value()
```

DO NOT special-case the test coordinates.

DO NOT modify tactical constants merely to make the test produce different actions.

The test must exercise the **real production attacking pipeline**.

Mocks/stubs may be used only for constructing deterministic engine observations, not for replacing the actual:

```text
GoalkeeperModel
ThreatMap
shot evaluation
action-value evaluation
PolicyController
```

where those production components exist.

---

# Propagation assertions

Instrument the test so we can identify exactly where differences appear.

Compare:

```text
L1 raw goalkeeper position
L2 normalized goalkeeper position
L3 GoalkeeperModel state
L4 shot/threat geometry
L5 candidate action values
L6 selected action
```

For A vs B and A vs C print/assert whether each level differs.

Example diagnostic:

```text
A vs B

L1 raw GK position:       DIFFERENT
L2 normalized position:   DIFFERENT
L3 goalkeeper model:      DIFFERENT
L4 threat evaluation:     DIFFERENT
L5 action values:         DIFFERENT
L6 selected action:       SAME

RESULT: PASS
The goalkeeper signal reaches attacking action evaluation.
```

Failure example:

```text
A vs B

L1 raw GK position:       DIFFERENT
L2 normalized position:   DIFFERENT
L3 goalkeeper model:      SAME
L4 threat evaluation:     SAME
L5 action values:         SAME
L6 selected action:       SAME

RESULT: FAIL

FIRST BROKEN PROPAGATION LEVEL:
L3 GoalkeeperModel
```

This diagnostic is more important than merely returning PASS/FAIL.

---

# Specifically investigate the suspected coordinate bug

Search the repository for every goalkeeper-related coordinate comparison.

In particular investigate logic equivalent to:

```python
gkx > 2.0
gkx < 10.0
```

when evaluating the **opponent goalkeeper**.

Given normalized attacking coordinates where:

```text
own goal ≈ x=0
opponent goal ≈ x=60
```

determine whether any goalkeeper logic has accidentally been written using the own-goal coordinate system.

Search for all related concepts:

```text
keeper
goalkeeper
gkx
gky
defensive fifth
goal distance
shot_beats_keeper
keeper displacement
keeper reach
```

Do not fix unrelated occurrences until you understand whether they are wrong.

Report every suspicious coordinate comparison with:

```text
file
function
current condition
expected coordinate semantics
whether it is actually a bug
```

---

# If the test exposes a bug

Do not immediately perform a broad refactor.

First report:

```text
ROOT CAUSE:
...

FIRST BROKEN PROPAGATION LEVEL:
...

FILES INVOLVED:
...

WHY A/B/C WERE PREVIOUSLY EQUIVALENT:
...
```

Then implement the **smallest correct production fix**.

After the fix:

1. rerun A/B/C/D;
2. rerun all existing goalkeeper tests;
3. rerun shot-selection tests;
4. rerun policy tests;
5. rerun the complete test suite.

Do not weaken existing assertions to make the new implementation pass.

---

# Required final report

Return:

```text
ELITE GK WIRING TEST
====================

TEST A
GK: (60,20)
Goalkeeper state:
Shot/threat value:
Candidate actions:
Selected action:

TEST B
GK: (60,24)
Goalkeeper state:
Shot/threat value:
Candidate actions:
Selected action:

TEST C
GK: (52,20)
Goalkeeper state:
Shot/threat value:
Candidate actions:
Selected action:

TEST D
GK: <state>
Goalkeeper state:
Shot/threat value:
Candidate actions:
Selected action:

A vs B propagation:
L1:
L2:
L3:
L4:
L5:
L6:

A vs C propagation:
L1:
L2:
L3:
L4:
L5:
L6:

FIRST BROKEN LEVEL:
<none or level>

COORDINATE BUGS FOUND:
...

FIXES APPLIED:
...

REGRESSION TESTS:
...

FULL TEST SUITE:
...

CONCLUSION:
Is goalkeeper position demonstrably wired into Deleri FC's real attacking decision pipeline?
YES / NO

EVIDENCE:
...
```

Do not declare the feature complete merely because the unit test passes.

The goal is to prove that goalkeeper position travels through the **actual production attacking pipeline** and materially affects the evaluation used to choose actions.