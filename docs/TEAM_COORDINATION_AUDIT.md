# TEAM COORDINATION AUDIT

Deleri FC — investigation only. No production behaviour was changed and no
`TeamPlan` mechanism was introduced. Every number below is measured; every code
claim is anchored to a file and line in the current `main` tree.

## Evidence base

| Source | What it is | Coverage |
| --- | --- | --- |
| Authoritative engine replays | `tools/team-cli/linux-x64/football-team simulate`, real engine, 60 s matches, both sides, 6 opponents (`reference-strikers`, `counter-elite`, `gk_hell-pro`, `high_press-pro`, `learned-methodic`, `slapstick-united`) | 36 matches, 45,027 snapshots, 0 replay errors |
| Offline policy replay | Deleri FC's own `RuntimeManager`/`PolicyController` re-run over every 20 Hz engine snapshot | 9,927 our-possession, 8,309 their-possession snapshots |
| Mirror probes | 400 randomised states per state class per subsystem, `S` vs `S` mirrored about the halfway line | 8 subsystems |
| Forced-action probes | Candidate board and off-ball response re-computed with the carrier's action held fixed | 200 real possession states |

The engine replay is authoritative. `src/sim.py` (LightEngine) was used only for
early structural probing and is labelled as such; no conclusion below rests on it.

Scripts: `/tmp/opencode/audit/exp{8,10,11,12,13}*.py` (read-only w.r.t. `src/`).

---

## 1. Verdict

**The team's weakness is team coordination, and it is almost entirely an
attacking-side problem.**

The decision layer is fast and honest. When a chance appears, it is taken on the
same tick. The failure is that the team almost never creates the chance, because
it never commits players forward with the ball: **87.0% of our possession
snapshots have zero teammates in the final third, and 99.3% have zero teammates
in the box.** Only 2.1% of the ticks where our carrier is inside shooting range
contain a shot that is actually worth taking.

The defensive side is comparatively sound: press count is 1 or 2, never 4, and
cover/rest-defence is preserved.

---

## 2. Attacking coordination (primary finding)

### 2.1 Measured shape, real engine

| Metric | Value |
| --- | --- |
| Our-possession snapshots | 9,927 |
| Off-ball teammates at `x >= 40` (final third) | **5.7%** of 30,991 teammate-ticks |
| Snapshots with **0** teammates in the final third | **87.0%** |
| Snapshots with **0** teammates in the box | **99.3%** |
| Off-ball teammate `x`, median | 22.0 m |
| Carrier `x`, median | 32.1 m |
| Off-ball `x` spread (max − min), median | 9.9 m |

Per role (formation slots: defender 14/20, left 28/8, right 28/32, striker 43/20):

| Role | n | median x | p90 x | % x ≥ 40 |
| --- | --- | --- | --- | --- |
| DEFENDER | 9,425 | 19.0 | 26.1 | 0.1% |
| WIDE_LEFT | 8,064 | 18.6 | 36.3 | 5.1% |
| WIDE_RIGHT | 6,415 | 27.4 | 38.9 | 6.9% |
| STRIKER | 7,087 | 30.1 | 40.9 | 12.6% |

Even the designated striker sits at a median `x` of 30.1 against a nominal slot
of 43.

### 2.2 Why — the anchors are functions of the ball, not of a team decision

Every off-ball attacking target is computed from the ball's *current* `x`:

- `src/policy.py:3161` — winger, ball in midfield/our half:
  `tx = clamp(ball.x + 5.0 * width, 15.0, 40.0)`. **Capped at 40.** A winger
  physically cannot hold a position beyond `x = 40` unless the ball is already
  in the final third.
- `src/policy.py:3153` — the winger's forward target
  (`tx = clamp(ball.x + 12.0 * width, 45.0, 52.0)`) is gated behind
  `ball.x >= 35`. That gate fires in only **41.2%** of our possession snapshots.
- `src/policy.py:3046-3048` — striker against a low block drops into pockets:
  `tx = clamp(ball.x + 8.0, 30.0, 40.0)`.
- `src/policy.py:3175-3182` — defender as rest defence:
  `tx = clamp(ball.x * 0.4 + 6.0, 8.0, 22.0)`, i.e. structurally behind the ball.

This is a deadlock, and it is self-reinforcing:

> The carrier has no forward passing option because nobody is ahead of him.
> Nobody moves ahead because moving ahead is gated on the carrier already being
> advanced. The measured result is a four-player shape compressed into a ~10 m
> band with a median `x` of 22, ten metres behind the ball.

### 2.3 There is no team layer to override this

`PolicyController.decide()` (`src/policy.py:1393-1424`) evaluates the carrier
first, then each off-ball player **independently**:

```
1408  if possessor is not None:  intents[possessor.id] = self._decide_possessor(...)
1412  for p in ours:  ...  self._decide_off_ball(inp, p, possessor)
1418  intents[gk.id] = self._decide_goalkeeper(...)
```

The only coupling from the carrier's decision to the rest of the team is two
instance fields, set at `src/policy.py:2171-2172`:

```
1373  _pending_pass_receiver: str | None = None
1374  _pending_pass_collection: tuple[float, float] | None = None
```

Measured over 200 real possession states with the carrier's action forced:

| Comparison | Off-ball player-states that changed target |
| --- | --- |
| carry → pass | 126 / 378 (exactly one player: the intended receiver, moving to its collection point) |
| carry → shoot | **0 / 147** |

So the team has a **one-bit** channel from the carrier to the team: *who is the
receiver*. There is no mechanism by which a shot, a switch, a wall play, or a
carry can cause a teammate to occupy a second-ball, far-post, or overload
position.

Two consequences, both structural:

- **A shot publishes nothing.** `_pending_pass_receiver` is assigned only under
  `if best_intent.action_type == "pass"`. On the tick after a shot the ball is
  uncontrolled, all outfielders route to `_off_ball_recover`, and the response is
  a loose-ball chase — never a rebound or far-post occupation.
- **The loose-ball receiver branch is dead code.** `_pending_pass_receiver` is
  set and consumed inside a *single* `decide()` call. `_off_ball_attack` runs
  first and clears it at `src/policy.py:3024-3025`; the duplicate branch in
  `_off_ball_recover` at `src/policy.py:3566-3571` is unreachable in practice.

### 2.4 The candidate board cannot see any of this

The carrier's choice comes from a flat, current-state board
(`_evaluate_attack_actions`, `src/policy.py:1851-2182`), all values from
`ExpectedValueCalculator` (goal-equivalent, one scale). Verified against
production logging: **160/160** top values matched.

Across 160 real traced states:

| | |
| --- | --- |
| Board size | 1 candidate in 20 states, 2 in 140 |
| At least one pass candidate | 100 / 160 |
| Shoot candidate | 40 / 160 |
| Winner | carry 100, shoot 40, switch 20 |

No candidate encodes what the teammates will do next, so the board cannot
prefer an action that *sets the team up*. It is a correct myopic optimiser over
a badly conditioned team state.

### 2.5 Duplicate off-ball targets

| Context | Snapshots with ≥2 off-fielders sharing one move target |
| --- | --- |
| We have the ball | 33.1% |
| They have the ball | 86.7% |

Cause (defensive side): `_cover_point(inp, p)` at `src/policy.py:3537-3550`
takes `p` and **never reads it**. Every cover player is sent to one identical
ball→our-goal interpolation point. Direct calls with four outfielders returned a
single distinct point in 4/4 players, and in 3,663/3,663 checked loose-ball
states there were exactly two distinct outfield targets.

The loose-ball path *does* contain a working coordination mechanism —
`loose_ball_meeting_point` plus ETA ranking at `src/policy.py:3552+` selects one
primary meeting-point runner (81% of decisions) and a second (19%). The problem
is the non-running players, who are all collapsed onto the same cover point.

---

## 3. Time to danger

Real-engine funnel over 36 matches:

| Stage | Count | Share |
| --- | --- | --- |
| Our-possession snapshots | 9,927 | — |
| Possessions lasting ≥ 3 snapshots | 328 | — |
| Reached the final third (ball `x ≥ 40`) | 145 | 44.2% of possessions |
| Reached the box (ball `x ≥ 50.5`) | 23 | 7.0% |
| A positive-EV shot became available | 29 | 8.8% |
| A shot was taken | 26 | 7.9% |
| Chance opened, possession ended with no shot | 3 | 10.3% of chances |

Per-tick:

| | |
| --- | --- |
| Ticks with carrier inside shooting range (≤ 30 m) | 5,391 |
| Of those, a positive-EV shot was available | **115 (2.1%)** |
| Ticks where a shot was taken | 74 |
| Shot taken on the same tick the chance appeared | 74 / 115 (64.3%) |
| Reaction latency, snapshots between chance and shot | median 0, p90 1, max 67 |
| Possessions shot at (near) their best available chance | 17 / 26 |

**Reading.** Reaction latency is already at the floor — one 20 Hz snapshot is
50 ms and the median is 0. There is no value in speeding up the decision. The
entire deficit is upstream: 91.2% of possessions never see a positive-EV shot at
all, and that is a *shape* outcome, not a decision outcome. The best shot EV seen
per possession had a median of +7.0 and a p90 of +65.4, so when the shape does
open a chance it is usually a good one.

---

## 4. Symmetry

Mirror probe: for each state `S`, build `S'` by reflecting the pitch about the
halfway line (`y → 40 − y`, negate `vy`, `facing → π − facing`) and require every
intent to satisfy `x(S) == x(S')` and `y(S) + y(S') == 40`. 400 randomised
states per state class.

| Subsystem | Result |
| --- | --- |
| `tactics.assign_roles` (us / them / loose) | symmetric |
| `tactics.detect` (us / them / loose) | symmetric |
| `loose_ball_meeting_point` | symmetric |
| `policy._off_ball_attack` | **breaks 393/400** |
| `policy._off_ball_defend` | **breaks 400/400** |
| `policy._off_ball_recover` | symmetric |
| `policy._decide_goalkeeper` (all three) | symmetric |
| `policy.decide` — loose ball | symmetric |
| `policy.decide` — us in possession | **breaks 304/400** |
| `policy.decide` — them in possession | **breaks 400/400** |
| `tactics.plan_press` | **breaks 47/400** |

`ExpectedValueCalculator` is symmetric (values swap exactly under mirroring;
`OpponentModel.prefer_side()` correctly changes sign, so the learned opponent
bias is not the problem).

### Attribution — earliest break first

**1. `tactics._trigger_note` — `src/tactics.py:158`, `carrier.facing > math.pi / 2`. Breaks 400/400.**
Under a `y`-mirror the facing angle maps `θ → π − θ`, so `θ > π/2` becomes
`θ < π/2`. Example: `S` returns `'facing_own_goal'`, the mirrored state returns
`''`. This is the first asymmetry in the pipeline: it runs before any policy code
and it decides who presses.

**2. `tactics.plan_press` — breaks 47/400.** Direct consequence of (1): the press
plan is mis-assigned, e.g. `p3` COVER ↔ PRESS_SUPPORT, `p2` REST_DEFENCE ↔ PRESS.

**3. `policy._pick_striker_channel` — `src/policy.py:2995-3005`. Breaks 369/400.**
The two candidate bands are scanned in the fixed order `[(10.0, 8.0, 12.0),
(30.0, 28.0, 32.0)]` with a strict `<` comparison, so a tie in opponent count
always resolves to the **left** band. Example: S and mirrored both return `10.0`
where mirroring requires `10.0`/`30.0`.

**4. `policy._off_ball_defend` — `src/policy.py:3236-3353`. Breaks 400/400.**
Two independent causes:
- Rest-defence anchor, `src/policy.py:3340-3341`: `elif state.goalkeeper_us() is
  not None and ball.y < 20.0:` → `anchor_y = 24.0 if role == ROLE_WIDE_RIGHT else
  16.0`. A **one-sided** condition: the rest-defence block is split 24/16 only
  when the ball is on the left half, and sits at a flat 20.0 when it is on the
  right. Example: `p2`/`p3` `ty` 20.0 vs 24.0/16.0.
- `_block_wall_lanes` / `_intercept_wall_pass`, `src/policy.py:3355-3404` and
  `3452-3523`: these test the opponent carrier's proximity to a **touchline**
  (`y < 8` / `y > 32`) and then set `wall_x = 0.0` / `PITCH_LENGTH` and derive
  `tx` from it. The touchline side is conflated with the **goal-line** `x`, so
  mirroring `y` flips the chosen `x` between ~4 (our goal) and ~56 (their goal).
  `_intercept_wall_pass` compounds this with a fixed `for wall_side in ("left",
  "right")` loop that returns on the first match. Isolated probe on
  touchline-heavy states: breaks 372/400, e.g. `tx` 4.0 vs 56.0.

**5. `policy._off_ball_attack` — breaks 393/400.** Entirely downstream of (3).

### Not broken

`assign_roles`, `detect`, `plan_press`'s distance ordering, `loose_ball_meeting_point`,
`_off_ball_recover`, `_decide_goalkeeper`, and the whole loose-ball path of
`decide()` are all mirror-equivariant. The asymmetry is confined to the
press-trigger facing test, the striker channel tie-break, the rest-defence anchor,
and the two wall-lane helpers.

---

## 5. Answering the original question

**Is the weakness team coordination?** Yes, and specifically this: the team has
**no mechanism for players other than the ball carrier to influence each other.**

Three layers, all measured:

1. **No team plan above the players.** `decide()` is a possessor call followed by
   N independent per-player calls. There is no object that represents "we are
   building a 3-player attack" or "the shot is on, get rebounds".
2. **A one-bit channel from carrier to team.** One pass receiver id. A shot
   conveys nothing, and the one place that could have made it persist across
   ticks (`_off_ball_recover:3566`) is unreachable.
3. **Shape anchors that cannot lead the ball.** Winger targets cap at `x = 40`
   unless the ball is already past `x = 35`; rest defence is pinned to
   `ball.x * 0.4 + 6`. The result is measured directly: 87.0% of possession
   snapshots with nobody in the final third, 99.3% with nobody in the box, and
   only 2.1% of in-range ticks with a shot worth taking.

The decision layer is not the problem. Median reaction latency from "a positive
shot exists" to "we shoot" is 0 snapshots, and 64.3% of positive-EV chances are
taken on the very tick they appear. Speeding up or re-weighting the candidate
board cannot fix a team that is not in the positions that create chances.

**Secondary, real but smaller:**

- Defensive cover collapses to one shared point (86.7% of their-possession
  snapshots have duplicate off-ball targets) because `_cover_point` ignores its
  own `p` argument.
- Four genuine left/right asymmetries, two of which (`_block_wall_lanes`,
  `_intercept_wall_pass`) are axis confusions — a touchline test driving a
  goal-line coordinate. These make the away side measurably different from the
  home side and are cheap to fix.

---

## 6. What exists today and must be preserved

| Mechanism | Location | Status |
| --- | --- | --- |
| Single-scale EV action selection | `src/policy.py:1851-2182`, `ExpectedValueCalculator` | sound; 160/160 verified |
| Pass receiver + collection-point movement | `src/policy.py:1373-1374, 2171-2172, 3018-3025` | works, but single-tick only |
| Loose-ball meeting point + ETA ranking | `src/policy.py:3552+` | works (81% / 19% primary / secondary) |
| Trigger-based coordinated press plan | `tactics.plan_press` | works (never 4 pressers); facing test is asymmetric |
| Rest defence | `src/policy.py:3175-3182` | works; anchor is one-sided |
| Per-match opponent model, keyed by `gameId` | `src/opponent.py` | symmetric under mirroring |

## 7. Baseline

`python -m unittest discover -s tests` → 282 tests, 1 failure, 3 skipped.
The single failure is pre-existing and unrelated:
`test_opponents.GeneratedTeamTests.test_generated_teams_are_up_to_date`.

## 8. Not done in this audit

No production code was modified. No `TeamPlan` was introduced. No tuning
constants were changed. The next step is a design decision, not an edit: the
evidence above points at a single team-level intent that sits above
`decide()` — something that can commit 2–3 players into the final third on its
own initiative rather than as a function of `ball.x`, and that can publish a
shot as a team event so the off-ball players occupy rebound and far-post
positions. That is a design for a follow-up, not a change made here.
