# Opponent pool: authoritative measurements

All numbers from the real Babylon engine via `football-team simulate`, not from
`LightEngine`. Our side is `my-team-fc` throughout; goals are written
`us-them`.

## The pool is by far the hardest opposition in the kit

Goals conceded per game by My Team FC, 12-24 games per opponent:

| opponent | record | we score | we concede | possession |
| --- | --- | --- | --- | --- |
| `reference` | 12-0-0 | 7.00 | 0.00 | 87.4% |
| `reference-strikers` | 12-0-0 | 6.00 | 0.00 | 51.6% |
| `slapstick-united` | 12-0-0 | 3.00 | 0.00 | 88.1% |
| generated `*-rookie` | 24-0-0 | 1.92-2.88 | 0.00 | 81-96% |
| generated `*-solid/pro/elite` | 14-0-10 | 0.58 | **0.42** | 63-80% |

The two rates must not be confused. A *low* conceded rate is ambiguous: it
means either "this side is a fortress" or "this side never gets near our goal".
The bundled teams concede nothing simply because we steamroll them and they
rarely test us. Among generated teams the rate does separate the rookies (0.00)
from the solid/pro/elite band (0.42), which is the only real separation the
current pool produces.

Every bundled and rookie opponent is beaten without conceding. The 40
solid/pro/elite variants are the only opposition that scores at all, and they
do so at roughly one goal every two and a half matches. Measured against our own
team, the generated pool is a different order of difficulty from anything the
kit ships with.

## The difficulty ladder is not calibrated

This is the main open finding, and it is a real limitation rather than a
measurement artefact.

Against our team, **all 40 solid/pro/elite opponents are indistinguishable**:

```
direct-elite       14/0/10  14-10  poss 75.4%  shots 5.58
high_press-elite   14/0/10  14-10  poss 63.3%  shots 5.88
low_block-elite    14/0/10  14-10  poss 79.8%  shots 4.58
gk_hell-elite      14/0/10  14-10  poss 76.0%  shots 5.33
wall-elite         14/0/10  14-10  poss 73.2%  shots 4.92
```

Same 14 seeds won in all five cases. Possession and shot volume differ by 16
points and 1.3 shots respectively, so the opponents genuinely play differently,
but none of that difference reaches the scoreline.

### Why short runs cannot rank anything

Against a solid side, **every 60 s match is either 1-0 or 0-1** — across 24
games there was not one draw, one 2-1, or one goalless:

```
14  1-0 win
10  0-1 loss
```

Tripling the duration to 180 s tripled shots (6.3 -> 18.2 per run) but left
goals at exactly 3-3 with identical per-seed outcomes. The engine is fully
deterministic per seed, and the single goal is decided early; after it, play
does not change the result.

Consequence: **win rate and goal difference are useless for ranking these
opponents.** They measure which of the deterministic seeds produced the one
goal. A 2- or 6-game sweep of several opponents will report identical numbers
for opponents of genuinely different strength, which is exactly what the first
two sweeps of this pool did.

`scripts/run_authoritative.py` now:

* defaults to `--games 24` rather than 2;
* ranks by **goals conceded per game**, which still separates a generated
  rookie (0.00) from a solid/pro/elite side (~0.42), and records the scoring
  rate separately so the two are not conflated;
* records the per-game scoreline histogram, so saturation is visible instead of
  being hidden inside an aggregate;
* prints a `WARNING` when every opponent concedes an identical rate, because a
  flat report must never be read as a ranking.

### Where the extra tuning has to go

`BAND_TWEAKS` already spreads the bands widely on paper (noise 0.22 -> 0.03,
press 0.50 -> 0.95, shoot_range 12 -> 22 m), and the offline arena separates
them clearly. None of it reaches the real-engine scoreline. The bands differ in
*quality*; the engine rewards only *conversion and defensive solidity*.

To make solid/pro/elite genuinely harder than each other the tuning has to move
the two quantities that actually decide matches here:

1. **Shot generation and quality.** Roughly 5 shots and 1 goal per match. More
   shots from closer range beat more pressing.
2. **Goals conceded.** Defend the goalmouth, force play wide, and protect the
   six-yard area rather than optimising compactness and tempo, which currently
   buy possession and nothing else.

Until that recalibration is done against the real engine, the family and band
names describe *style* accurately and *strength* only at the rookie/non-rookie
boundary. Treat the labels as stylistic, not as a difficulty ranking.

## Bugs this pass found

* **The six `learned-*` teams could not start.** `build_opponent_teams.py`
  inlined the parameter block with `json.dumps`, so `"learns": true` landed in
  Python source as `true` and raised `NameError` on the first line. The teams
  never bound port 8080 and the engine reported "did not become healthy within
  15 seconds" for every seed and both sides. The file was byte-stable, so
  `--check` happily blessed it. Fixed by emitting Python literals with
  `pprint.pformat`; `--check` now also compiles every generated module, and
  `scripts/_import_probe.py` imports and exercises all 51 team directories.
* **`LightEngine.can_act` latched false forever after a knockdown** — grounded
  decayed to zero but only a goal restart cleared the flag, so a slapped player
  ran but could never pass or shoot again. Fixed in `src/light.py`.
* **`plan_to_obs` reported `canAct: true` unconditionally**, hiding the 0.3 s
  kick cooldown from every policy and producing silently dropped actions.
* **`shoot_range` was clamped into 0..1** by the bias path, silently turning a
  22 m shooting range into 1 m.
* **The pass search could play the ball into its own sixth**, retreating to the
  corner when the carrier was deep.
* **`OPPONENTS` was annotated `dict[str, type[OpponentController]]`** although
  `play_match` only calls the value and the arena and engine probe both register
  duck-typed factories. Those registrations were type errors that worked fine at
  runtime; it is now a structural `OpponentLike` protocol.
* **`engine_probe` used `hash()` for module names.** String hashing is salted
  per process, so names changed between runs; it now uses a stable blake2b
  digest.

## Reproducing

```sh
# fast, offline, LightEngine -- relative difficulty only
python scripts/opponent_arena.py --games 3 --decisions 1200

# authoritative, real engine
python scripts/run_authoritative.py --only high_press-elite direct-elite --games 24
python scripts/run_authoritative.py --games 24 --json artifacts/opponents/authoritative.json

# bundled opponents
football-team simulate --path . --opponent reference --games 12 --duration 60
```

A full 50-team sweep at 24 games is roughly 25 minutes of Docker; `--only` a
handful of teams for a few minutes.
