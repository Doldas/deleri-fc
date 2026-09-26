# Game engine rules and team-control boundaries

This is the complete developer-facing rules reference for the current Football Babylon five-a-side engine. The match engine is authoritative: a team returns requested intents, and the engine decides what is valid and resolves movement, possession, tackles, saves, goals, timing, and results.

If a strategy guide, starter-team comment, viewer animation, or website summary differs from this document, this document describes the intended engine behavior.

## What your team controls

At each observation, your team may return no more than one intent for each supplied player ID. An intent can contain:

- `move`: an absolute target in metres and a normalized speed from `0` to `1`;
- `face`: an absolute point for an outfield player to face;
- `action`: `none`, `pass`, `shoot`, `clear`, `tackle`, or `slap`.

Your team controls the goalkeeper as well as the four outfield players:

- You may set the goalkeeper's movement target and speed.
- The engine keeps the goalkeeper facing the field, so a goalkeeper `face` request does not change its facing direction.
- Saves, dives, loose-ball handling, and close challenges are automatic. There is no separate goalkeeper action.
- A goalkeeper cannot use `tackle`.
- When the goalkeeper has possession and `canAct` is true, it may request `pass`, `shoot`, or `clear` like another possessor.

An accepted intent is still only a request. A requested action may do nothing when its role, possession, range, target, phase, or cooldown requirements are not satisfied. Teams cannot directly set player or ball positions, velocities, possession, scores, cooldowns, grounded state, match phase, events, or animations.

## Decision contract

- The simulation runs at 60 ticks per second.
- Teams receive observations and make decisions at 10 Hz. Accepted decisions for both teams are applied at the same future tick.
- The viewer receives authoritative snapshots at 20 Hz.
- Copy `protocolVersion`, `gameId`, and `sequence` from the current observation into the decision.
- Return at most one intent for each player in `us`; unknown and duplicate player IDs are invalid.
- All coordinates, speeds, and power values must be finite. `NaN` and infinity are invalid JSON values.
- Movement speed and kick power must be in the inclusive range `0..1`.
- `pass`, `shoot`, and `clear` require a finite target. Power defaults to `0.5` when omitted.
- `canAct` is false while a player is in an action cooldown, staggered, grounded, or knocked down. Movement and action resolution remain authoritative even if an intent was submitted earlier.
- The full wire contract is defined by `protocol/openapi.yaml` and the JSON Schemas under `protocol/`.

## Match, teams, and field

- Each team has exactly five players: one goalkeeper and four outfield players.
- The standard pitch is 60 m long and 40 m wide, with a 6 m wide goal centered on each goal line.
- In team observations, `(0, 0)` is the corner beside your own goal line and your team always attacks toward positive X. Away-team coordinates are already normalized; do not mirror them yourself.
- Match duration is configurable.
- If enabled, a golden-goal period follows a tie. If it remains tied, the engine runs a staged best-of-five shootout: an opening announcement is followed by a camera/setup hold, strike, and clearly held goal-or-save result for every kick. Each kick independently chooses left, center, or right for both the shooter and goalkeeper. Matching choices produce a save; every other combination is a goal. The choices look random but are seeded so the match remains exactly reproducible. If the scores are still level after five kicks each, the existing seeded winner fallback decides the match.
- Offside, fouls, cards, stamina, substitutions, corners, throw-ins, and goal kicks are not part of this ruleset.
- Engine outcomes are deterministic for the supplied inputs and seed.

## Movement, facing, and dribbling

- An outfield player or goalkeeper without the ball can run at up to 8 m/s.
- A player in possession dribbles at 80% of maximum speed: up to 6.4 m/s.
- `move.target` is an absolute destination, not a direction. The latest valid movement target and speed remain active until replaced or the engine resets them.
- Movement changes immediately toward the target. There is no acceleration model.
- A staggered, grounded, or knocked-down player has zero velocity and does not advance toward its stored movement target. It resumes from its actual position after recovery; it does not catch up to a position it would otherwise have reached.
- An outfield player's valid `face` point changes its facing direction. A staggered, grounded, or knocked-down player's facing does not change.
- The goalkeeper always faces toward the opponent's goal.
- A controlled ball is held 0.65 m in front of its player in the player's facing direction and moves with that player.
- New possession is protected from outfield tackles for 0.25 seconds. Goalkeeper handling is not blocked by this protection.

## Loose-ball control

- An eligible outfield player controls a free ball when the ball is within 0.9 m and travelling at no more than 5 m/s.
- A goalkeeper can control a free ball of any speed when the ball's swept path for the current tick passes within 1.65 m and the goalkeeper is inside its own defensive fifth.
- A player in cooldown, staggered, grounded, or knocked down cannot recover a loose ball.
- If several players qualify, the closest eligible player wins. Equal distances are resolved deterministically by player ID.
- In a normalized observation, `ball.possessingTeam` is `us`, `them`, or `null`. Always check it together with `ball.possessedBy`, because both teams use the same local IDs.

## Tackling

- Only an outfield player with `canAct: true` can request `tackle`.
- A tackle succeeds when an opponent controls the ball, the 0.25-second possession protection has expired, and the tackler reaches within 1.15 m of the controlled ball during resolution.
- A close tackle from less than 0.8 m transfers possession directly to the tackler.
- A reaching tackle from 0.8 m through 1.15 m is a slide tackle. It knocks the ball loose at 10 m/s instead of giving the tackler possession.
- The slide travels toward the controlled ball, stops at contact distance, and leaves the tackler grounded for 1.05 seconds.
- The grounded tackler cannot move, turn, act, or recover the ball. It resumes from the slide's actual endpoint when recovery finishes.
- Both involved players receive a 0.3-second action cooldown after either kind of successful tackle.
- Tackle events contain the distance from the tackler to the controlled ball at contact so the viewer can distinguish close and slide tackles.
- Tackles are deterministic. There are no random tackle failures, fouls, cards, or injuries in this ruleset.

## Goalkeeper handling and diving

- Automatic goalkeeper handling only applies while the goalkeeper is inside its own defensive fifth: the first 20% of the pitch measured from its own goal line.
- In that area, the goalkeeper automatically takes an opponent-controlled ball that comes within 1.65 m.
- A goalkeeper can catch a loose ball of any speed when its swept path passes within 1.65 m.
- When the loose ball came from an opponent's shot, a save needing less than 0.8 m of lateral reach is a standing catch.
- A shot save needing 0.8 m through 1.65 m of lateral reach is a dive. The goalkeeper moves laterally toward the ball and remains grounded for 0.9 seconds while holding it.
- A grounded goalkeeper cannot move or act, but retains possession during the dive recovery.
- The strategy chooses positioning and distribution; it does not request the catch or dive animation.
- The goalkeeper may request `pass`, `shoot`, or `clear` while it possesses the ball and can act.
- If the goalkeeper still possesses the ball after 1.25 seconds, the engine automatically distributes it at 20 m/s toward a wide outfield teammate and applies a 0.5-second cooldown.

## Slaps

- Any player with `canAct: true` can request `slap`; it needs no target or power.
- The slap hits the nearest opponent within 1.15 m who is not grounded or already knocked down. It does nothing when nobody is in range.
- A glancing slap knocks the opponent back 0.25 m and staggers them for 0.3 seconds. They cannot move, turn, act, or recover the ball during the stagger.
- A slap is a clean hit when the opponent is within 0.75 m, the slapper is facing within 60 degrees of them, and the opponent has no knockdown immunity.
- A clean hit knocks the opponent back 0.7 m and knocks them down for 1.2 seconds: 0.2 seconds falling, 0.6 seconds down, and 0.4 seconds getting up. They cannot move, turn, act, or recover the ball during the full knockdown.
- A knocked-down ball carrier drops the ball, which is knocked loose at 8 m/s. A glancing slap does not force possession loose.
- After getting up, the player has 1.25 seconds of knockdown immunity. A slap during immunity can still stagger them.
- The slapper receives a 1.5-second cooldown, making timing more useful than repeated requests.
- Slaps submitted for the same tick resolve simultaneously, so two opponents can slap each other.
- Grounded or already knocked-down players cannot be slapped again. Knockback positions are clamped to the pitch boundary.
- Slaps are deterministic: range, facing, cooldown, and immunity completely determine the result.
- Successful slap events identify both players and report a `stagger` or `knockdown` outcome.
- `slaps` counts every successful contact; `knockdowns` counts only clean hits that put an opponent down.

## Ball actions and physics

- Valid actions are `none`, `pass`, `shoot`, `clear`, `tackle`, and `slap`.
- `none`: performs no immediate ball action. Any requested movement or facing is still applied.
- `pass`: kicks the ball toward the supplied target and records a pass. Use it to send the ball to a teammate or into space.
- `shoot`: kicks the ball toward the supplied target and records a shot. Use it for an attempt on goal.
- `clear`: kicks the ball toward the supplied target without recording a pass or shot. Use it to move the ball away from danger.
- `tackle`: asks an outfield player to challenge the opponent possessing the ball. A valid close tackle wins possession; a valid reaching tackle becomes a slide and knocks the ball loose.
- `slap`: tries to briefly knock down the nearest opponent in arm's reach, whether or not either player has the ball.
- Only the current possessor can perform `pass`, `shoot`, or `clear`.
- A valid kick releases the ball toward its target and applies a 0.3-second cooldown to the kicker.
- Kick power `0..1` maps linearly to an initial speed from 12 to 26 m/s.
- A free ball's velocity is multiplied by `0.992` on every 60 Hz engine tick.
- On the touchlines, the wall retains tangential velocity and returns 75% of the perpendicular velocity.
- Outside the goal opening, goal-line walls use the same 75% perpendicular rebound.
- A goal is scored when the ball crosses a goal line inside the 6 m goal opening.

## Kickoffs, goals, and match phases

- The designated kickoff team's `f1` receives possession at the center spot.
- After a goal, the conceding team receives the next kickoff.
- In the event host and broadcast, a normal goal enters a four-second celebration followed by a five-second replay.
- The Team Practice Lab omits celebration and replay presentation and proceeds directly to the restart formation.
- When play will continue, players are reset to their formation for a two-second restart hold.
- Celebration, replay, and restart presentation time does not consume active match time. Omitting celebration and replay in practice therefore does not change the amount of active play.
- A golden goal completes the match after its presentation sequence rather than restarting play.
- During presentation and completed phases, the engine controls the state and normal strategy requests may not take effect.

## Kits and viewer presentation

- Team branding colors are never rewritten by match setup.
- Before kickoff, the engine resolves a separate visible kit color for each team.
- It uses both primary kits when they are perceptually distinct. On a clash it tries supplied alternate colors and selects the most distinguishable pairing.
- If supplied colors still cannot produce a readable matchup, the engine uses a guaranteed light/dark fallback pairing.
- Resolved kit colors are part of the authoritative match snapshot so every viewer shows the same matchup.
- When either team first falls six goals behind, the engine deterministically selects one seeded broadcast surprise and records it in every later match snapshot. Without an explanatory overlay, the viewer either sends almost all losing supporters to the exits, stages a six-supporter invasion that knocks down and kicks a losing player, or has the bench-seated Oddball jump in anger, slap every losing-side player down, and return to his seat. This selection happens at most once per game and never changes gameplay state or statistics.
- Slide tackles, slap staggers and knockdowns, grounded recovery, goalkeeper dives, celebrations, and replays are visualizations of engine state. Viewer animation does not alter gameplay.

## Scope summary

The strategy controls requested movement, outfield facing, and eligible ball actions. The engine controls whether those requests succeed and owns every physical, timing, scoring, restart, kit-resolution, and result rule described above.