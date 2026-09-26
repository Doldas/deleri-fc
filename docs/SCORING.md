# Why we score nothing in open play, and what would actually fix it

120 matches against Vanguard FC (elite) produced zero open-play goals. The
cause is geometric, not a tuning problem, and it is worth writing down
because three separate changes have now failed to move it.

## A centred keeper cannot be beaten from anywhere realistic

From `RULES.md`:

* :50 a goalkeeper without the ball runs at up to 8 m/s, the same as an
  outfield player;
* :63 he controls a free ball of **any** speed whose swept path passes within
  1.65 m while he is inside his own defensive fifth;
* :86 a save needing 0.8 m to 1.65 m of lateral reach is a dive.

So the keeper tracks the ball continuously at 8 m/s for the whole flight, and
the test is not the raw angle but the angle left over once he has moved. With
a shot at 25.3 m/s (power 0.95) from the centre line at the far post, the
target sits 2.4 m from a keeper on the centre line:

| range | flight | keeper shift | reach left | outcome |
|-------|--------|---------------|------------|---------|
| 6 m  | 0.24 s | 1.90 m | 0.50 m | standing catch |
| 10 m | 0.40 s | 3.16 m | -0.76 m | standing catch |
| 18 m | 0.71 s | 5.69 m | -3.29 m | standing catch |
| 26 m | 1.03 s | 8.22 m | -5.82 m | standing catch |

He only has to cover 2.4 - 1.65 = 0.75 m, and at 6 m he has 0.24 s to do it,
which needs 3.2 m/s against the 8 m/s he has. Even point blank he makes it.
The shot only beats him from roughly 2 m, or once he is no longer central.

`physics.shot_beats_keeper` encodes exactly this and is behaving correctly.
Verified: given a keeper on the centre line it returns False at 6, 10, 14, 18,
22 and 26 m, and returns True once he is dragged out of his defensive fifth.

## Consequence: the keeper's y is the entire game

The goal is 6 m wide and he re-centres at 8 m/s. `pick_shot_target` already
picks whichever of the near post, far post and far-from-shooter corner is
furthest from his actual position, so the aiming logic is right. What is
missing is any mechanism to move him off the centre line first, because
without that the best available shot is a standing catch at every range.

That is the whole scoring problem, and it is why these all failed to help:

* **Suppressing hopeless shots.** Cut shots 4.13 -> 3.15 per match and
  correctly removed the 26-of-35 centre-back efforts from range. Win rate
  unchanged, because the shots it removed could not have scored anyway.
* **Pass collection geometry.** Cut loose ball 47.1% -> 38.1% and raised
  final-third possession 24.2% -> 38.4%. Genuine progress, but possessing the
  ball 26 m out does not produce a shot that beats a set keeper.
* **Wider shape.** Width 11.05 m -> 18.9 m. Angle helps only against a keeper
  who is already displaced; from the centre line the far post is 2.4 m away
  whichever side we are on.

## What would actually work

Ranked by how directly each attacks the cause:

1. **Drag the keeper, then shoot.** Attack the goal he is covering, sell the
   far post, and take the near one. A cross or a square pass in the box
   forces him to commit, and a keeper who has moved 2 m concedes a 4.4 m
   gap that is already beyond his 1.65 m reach. This is the only route to a
   goal that the geometry actually permits.
2. **Get the ball inside ~7 m.** We hold the ball about 26 m out and are in
   the final third only 38% of the time, so we rarely even reach the range
   where the decision is close. Short passing is impossible by construction
   (every pass rolls MIN_PASS_TRAVEL ~15.6 m), so progress has to come from
   dribbling at 6.4 m/s and from long balls into space.
3. **Attack the far post from a wide angle with the keeper pulled.** Only
   worth doing once (1) is in place.

Note the tension in (2): the 15.6 m minimum pass travel means a short pass in
the box does not exist, so the cross in (1) has to be a real cross, not a
give-and-go. That has not been tried yet and is the most promising untested
idea.
