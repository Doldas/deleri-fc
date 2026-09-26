# A/B testing against Vanguard FC (elite)

The CLI has native A/B support, so nothing here reimplements it:

```sh
football-team build --path DIR --tag ab-<name>          # distinct tag per variant
football-team simulate --image ab-a --compare-image ab-b \
    --opponent-path opponents/counter-elite \
    --games 30 --seed-prefix NAME --sides both --skip-build
```

`simulate` also takes `--require-win-rate`, `--min-average-goal-difference`,
`--max-missed-decisions` and the `*-drop` gates, so acceptance can be enforced
rather than eyeballed. Variants are compared as git refs checked out into
`git worktree`s, so neither the working tree nor the `:dev` tag is disturbed.

## The engine is not deterministic, and that changes everything

Re-running the *same* command with the *same* seed prefix does not reproduce:

| run | variant | W/D/L | goals | missed decisions |
|-----|---------|-------|-------|------------------|
| 1 | main    | 12/0/48 | 12-60 | 11 |
| 2 | main    | 12/0/48 | 12-60 | 11 |
| 1 | pre-fix | 14/0/46 | 14-64 | 15 |
| 2 | pre-fix | 12/0/48 | 12-66 | 15 |

`main` happened to repeat exactly; the pre-fix policy did not. It is not
parallelism: with `--jobs 1` and only 20 matches, two runs still differed
(shots 2.80 vs 2.85, missed decisions 0 vs 1).

Outcomes are therefore a function of wall-clock timing as well as the seed, most
likely a decision deadline in the engine: under load it drops decisions, and a
dropped decision is a dropped match. The diagnostic to watch is
`Missed decisions`, and it rises with total load (0 at 20 matches, 11-15 at 120).

Consequences for how results are read:

* A win-rate gap of a few percent is noise. 12/60 versus 14/60 is not a result.
* Goal difference is steadier than win rate, and missed decisions is the best
  single health signal, because it measures a real defect rather than luck.
* `simulate`'s PASS line is only meaningful when gate flags are actually passed.
  Run without them it passes trivially, and it did exactly that here.

## Protocol

* One seed prefix per experiment, recorded in the results table below, so a
  result can be re-examined rather than silently regenerated.
* At least 30 games (60 matches) per variant; treat anything smaller as a smoke
  test. The n=8 and n=12 numbers quoted in earlier commit messages are below the
  noise floor and should not be cited as evidence.
* Low `--jobs` to reduce decision drops, and report missed decisions next to any
  win-rate claim.
* Confirm the two images really differ before trusting a comparison:
  `docker image inspect --format '{{.Id}}' ab-a ab-b`.

## Results

Vanguard FC (elite) = `opponents/counter-elite`, 60 s, both sides, seed prefix
`abtest-v1`, `--jobs 8`, 60 matches per variant.

| variant | W/D/L | win rate | goals | avg diff | shots | missed |
|---------|-------|----------|-------|----------|-------|--------|
| main (3 fixes) | 12/0/48 | 20.0% | 12-60 | -0.800 | 3.15 | 11 |
| pre-fix baseline | 14/0/46 | 23.3% | 14-64 | -0.833 | 4.13 | 15 |
| pre-fix, rerun | 12/0/48 | 20.0% | 12-66 | -0.900 | - | 15 |

**The three fixes did not improve the win rate.** The pre-fix baseline scored
14 wins in one run and 12 in another, so the -3.3% "regression" attributed to
the fixes is inside the baseline's own run-to-run spread. Goal difference is a
wash (-0.800 vs -0.833/-0.900).

What the fixes do buy is measurable and worth keeping:

* Shots fell 4.13 -> 3.15, i.e. the 26-of-35 centre-back efforts from range that
  could never beat a set keeper are gone.
* Missed decisions fell 15 -> 11 under identical load, so the policy answers
  within the engine's deadline more reliably.

Those are correctness and robustness wins, not scoring wins. The open problem
is unchanged: we hold the ball about 26 m from goal, rarely reach the 7 m where
a shot is winnable at all, and score nothing in open play.

## Goalkeeper rework

The keeper was rebuilt against the strongest thing in the generated set:
`opponents/counter-elite`'s `_gk_plan`, which gets the geometry right where
ours did not. Ours clamped himself into the goal mouth (y 17.6..22.4, a 4.8 m
band in a 6 m goal) and sat at x=2.0-2.5 for the whole match. The opponent
stands on the ball-to-centre bisector at 0.62 and scales his depth with the
ball, clamped inside the defensive fifth.

Against Vanguard, 60 matches per variant:

| seed | variant | W/D/L | win rate | goals | conceded/match | clean sheets |
|------|---------|-------|----------|-------|----------------|--------------|
| gk-v1 | new keeper | 18/0/42 | 30.0% | 18-47 | 0.78 | 30.0% |
| gk-v1 | old keeper | 9/0/51 | 15.0% | 9-71 | 1.18 | 15.0% |
| gk-v2 | new keeper | 24/0/36 | 40.0% | 24-39 | 0.65 | 40.0% |
| gk-v2 | old keeper | 17/0/43 | 28.3% | 17-63 | 1.05 | 28.3% |
| gk-v2 | positioning only | 22/0/38 | 36.7% | 22-42 | 0.70 | 36.7% |

Replicated across two independent seed sets: +15.0 and +11.7 points of win
rate, goal difference +0.55 and +0.517. Pooled 42/120 (35.0%) against 26/120
(21.7%), and 86 goals conceded against 134, a 36% reduction. That is well
outside the run-to-run spread, so unlike the earlier three fixes this one is
real.

### The part that nearly shipped by accident

The same commit also pointed the keeper's distribution at the ball's
collection point, on the argument that a pass aimed at a teammate's feet
sails 15.6 m over their head. It passed all 194 unit tests, built, validated,
and looked like a further +3.4 points against Vanguard. Against the reference
side it cut attacking output from 7.0 to 2.5 goals per match:

| vs reference, 40 matches | W/D/L | goals |
|--------------------------|-------|-------|
| keeper + distribution fix | 40/0/0 | 101-0 |
| keeper positioning only | 40/0/0 | 277-0 |
| old keeper | 40/0/0 | 280-0 |

A keeper pass is a long ball by nature and a forward can run onto one. Solving
for the collection point lands the ball ~15.6 m beyond the receiver, usually in
the opponent's half where we have nobody, and nothing orders a teammate to go
and collect it. The distribution change was reverted; positioning was kept.

No unit test could have caught this. The keeper's chosen target is
byte-identical between the two versions across every single-tick scenario in
`tests/test_policy.py`; the difference only accumulates over a whole match.
`scripts/practice_gate.sh` exists for that reason and now runs both opponents
on every change, because the two have opposite incentives: reference punishes
losing the ball and rewards building attacks, Vanguard pressures us and
attacks our goal. A change that improves one and wrecks the other is not an
improvement.

### Calibration

The gate's thresholds are set from measurement, not guesswork. An initial run
at 40 matches flagged 1.60 conceded per match against a 1.05 ceiling; at 80
matches the same build measured 0.74, and 7.00 goals per match against
reference, which reproduces the 280-0 figure exactly. So a marginal FAIL means
re-run with a different `SEED`, and the default is 20 games per opponent for
speed with `GAMES=40` before believing any marginal result.

