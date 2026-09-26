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
