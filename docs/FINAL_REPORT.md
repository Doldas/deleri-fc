# Round 4 — Why the team never went for goal

## Symptom

The team looked busy, held the ball, and never scored. Recorded matches showed
blue (us) circulating in our own half while the opponent simply walked the ball
into our net. Simulations agreed: **10-0-10** against `reference` with **0.5
shots per match** and 2 goals scored in 20 games.

## Root cause — an engine physics law the policy had been fighting

`RULES.md` states two facts that together make passing nearly impossible:

1. *"An eligible outfield player controls a free ball when the ball is within
   0.9 m and travelling at no more than 5 m/s."*
2. `kick_speed(power) = KICK_MIN_SPEED + (KICK_MAX_SPEED - KICK_MIN_SPEED) * power`
   with `KICK_MIN_SPEED = 12.0` and `KICK_MAX_SPEED = 26.0`.

So **every pass leaves the ball travelling at 12–26 m/s**, while a free ball is
only collectable below **5 m/s**. Players run at 8 m/s. The ball is therefore
faster than any player and cannot be touched until it decays below the control
limit, which takes:

| pass speed | metres rolled before it can be collected |
| --- | --- |
| 12 m/s (minimum power) | 15.6 m |
| 16 m/s | 24.0 m |
| 20 m/s | 32.3 m |
| 26 m/s (maximum power) | 44.8 m |

A minimum-power pass is a 15.6 m ball that nobody can catch.

### Measured proof from the replay (`resultReason: normalTime`)

| metric | before |
| --- | --- |
| snapshots with the ball **loose** (unowned) | 487 / 601 = **81.0 %** |
| ball speed while loose | mean **10.5 m/s**, max 23.7 m/s |
| mean distance from our nearest player to the loose ball | **3.26 m** (control radius is 0.9 m) |
| samples with a player within 0.9 m **and** ball ≤ 5 m/s | **0 of 487** |

Not one pass in the whole match could legally be collected. The old policy made
this worse by aiming at coordinates where no teammate existed:

* `route_one_chip` passed to a hardcoded `tx = 52.0` regardless of where the
  striker actually was;
* kickoff passed to a fixed `(35, 28)` while the winger started *behind* the
  passer at `(23, 25)`;
* `_dribble_target` aimed at "8 m behind their deepest defender", i.e. up to
  25 m away — and `move.target` is an absolute destination, so the carrier
  simply parked in a corner while the defence walked into it.

## The fix

Carrying the ball is the only movement that reliably keeps it: a dribbling
player holds the ball 0.65 m in front and moves at 6.4 m/s.

1. **`ball_travel_before_control()` / `loose_ball_meeting_point()`** (new, in
   `src/policy.py`) compute, from the ball's own velocity and the 0.992/tick
   decay, where and when a loose ball will first be collectable, clamped to
   what a runner can actually reach.
2. **Dribbling is now the default with possession.** A pass is only taken when
   `_collectable_pass()` finds a receiver already within
   `MAX_COLLECTABLE_PASS` (16.6 m), unmarked (> 5 m), with a clear lane, and
   actually moving us forward. Otherwise we carry.
3. **`_dribble_target()`** is now a short, re-aimable 7–9 m step toward goal
   that steers around the nearest marker and slides wide when someone blocks
   the lane, instead of a 25 m dash to the goal line.
4. **`_off_ball_recover()`** sends the first runner to the *meeting point*
   rather than the ball's current position, lets the second join only if it is
   reachable in time, and keeps everyone else goal-side so winning the ball
   does not expose us to a counter.
5. **Kickoff** starts by carrying the ball, with the striker and a winger
   pushed ahead of the ball as outlets and the centre-backs held for rest
   defence.

Shooting was already the highest-value action (`goal = +1.0`,
`goal_conceded = -1.0` in `REWARD_DEFAULTS`) and the shooting logic is
untouched — the team simply never *reached* shooting range before.

## Results

Image `sha256:808b59e7869561eb69126f95e8e23db1d4e3c44205e1e197cb8c7cbc47033c1f`,
team `My Team FC - R4` / `MTF-R4`.

| opponent | result | goals | possession | shots/game | clean sheets |
| --- | --- | --- | --- | --- | --- |
| `reference` | **10-0-0** | 35-0 | 88.2 % | 4.0 | 100 % |
| `reference-strikers` | **10-0-0** | 25-0 | 53.8 % | 2.5 | 100 % |
| `slapstick-united` | **10-0-0** | 20-0 | 87.8 % | 3.0 | 100 % |

Before → after vs `reference`: **10-0-10 → 20-0-0**, goals 10-10 → **70-0**,
shots 0.5 → **4.0**, loose ball 81 % → **52 %** (20 games, fresh seeds).

All three opponents at 100 % shows this is a physics fix, not tuning against
one opponent.

### Replay verification

`resultReason: normalTime` (not `deterministicPenalties`), and all three goals
attributed to our own players:

```
GOAL tick  775  teamId=team-a  playerId=team-a:d2
GOAL tick 1321  teamId=team-a  playerId=team-a:f1
GOAL tick 1867  teamId=team-a  playerId=team-a:f1
ball_x_max 59.98   closest approach to attacking goal: 0.02 m
```

No own goals.

## Gates

- `python -m unittest discover -s tests` — **98 tests OK** (11 new tests pin
  the loose-ball physics, the collectable-pass filter, the reachable dribble
  step, the meeting-point chase and pitch bounds).
- `football-team build` — OK
- `football-team validate` — **PASS**, 25 seeded fixtures replayed identically,
  p50 0.8 ms / p95 1.3 ms / max 17.7 ms (limit 100 ms), no remediation.

## Note on the recordings

The CV tooling cannot be trusted on these broadcast captures: `recording.webm`
yields unstable pitch bounds (it switches camera/zoom mid-file) and reports 22
"players" for a five-a-side team, so it is picking up crowd and HUD pixels. The
diagnosis above therefore comes from engine replays, which `AGENTS.md` §2 ranks
above video and intuition anyway. The recordings are still useful as a
qualitative symptom report, and `recording.webm` does show 0-0 with blue pinned
in its own half, consistent with the measured 81 % loose ball.
