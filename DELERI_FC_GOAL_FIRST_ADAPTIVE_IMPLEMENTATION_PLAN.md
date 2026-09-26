# Deleri FC — Goal-First Adaptive Implementation Plan

## Vision
Deleri FC is an intelligent, adaptive team that scores **first and often**. Every decision flows from a single question: *how do we create a high-probability shot in the next 3–5 seconds?* The team adapts in real time to opponent structure, exploiting space behind high lines, stretching low blocks wide, punishing passive defending with wall passes, and winning transition moments through coordinated counter-pressing.

---

## 1. Core Tactical Identity

| Principle | Implementation |
|-----------|----------------|
| **Goal-first decision tree** | Shot → Cross/Cutback → Through ball → Wall pass → Switch → Carry → Safe pass |
| **Adaptive shape** | Detect opponent archetype (high press / low block / counter / possession / wall / physical) and auto-select counter-tactics |
| **Transition dominance** | Immediate counter-press on possession loss; instant vertical outlet on regain |
| **Wall intelligence** | Offensive wall passes in final third; defensive wall-lane blocking |
| **Set-piece alternatives** | No corners/throw-ins — use quick restarts to catch defence unorganised |

---

## 2. Opponent Archetype Detection & Counter-Tactics

The `OpponentModel` already tracks: `press_intensity`, `defensive_shape_height`, `wall_usage`, `side_bias`, `transition_speed`, `shot_distance`. Extend `detect_archetype()` and `get_counter_tactics()`:

```python
ARCHETYPES = {
    "high_press": {
        "triggers": ["press_intensity > 0.7", "shape_height < 25"],
        "counters": {
            "bypass_to_striker": True,           # long ball over press
            "one_twos": True,                    # quick combinations
            "third_man_runs": True,              # player between lines
            "gk_direct": True,                   # GK throws to wingers
            "defensive_line": 28,                # deeper rest defence
            "width": 1.0,                        # full width
        }
    },
    "low_block": {
        "triggers": ["shape_height > 40", "compact_width < 22", "deep_players >= 3"],
        "counters": {
            "wing_crosses": True,                # pin fullbacks, cross to box
            "cutbacks": True,                    # byline pull-backs
            "switch_play": True,                 # rapid side-to-side
            "pull_backs": True,                  # striker to edge of box
            "wall_passes": True,                 # wall combinations
            "defensive_line": 18,                # high line to stretch
            "width": 1.0,
        }
    },
    "counter_attack": {
        "triggers": ["transition_speed > 0.6", "direct_play > 0.5"],
        "counters": {
            "rest_defence_depth": 22,            # two deepest cover
            "counterpress_intensity": 0.4,       # don't overcommit
            "intercept_through_balls": True,     # read and cut
            "possession_priority": 0.7,          # control tempo
        }
    },
    "possession": {
        "triggers": ["possession_ticks > 0.6", "press_intensity < 0.4"],
        "counters": {
            "disrupt_rhythm": True,              # targeted press triggers
            "force_errors": True,                # press when facing own goal
            "counter_on_turnover": True,         # fast vertical
        }
    },
    "wall_play": {
        "triggers": ["wall_usage > 0.3"],
        "counters": {
            "block_wall_lanes": True,            # proactive positioning
            "force_central": True,               # deny wide wall access
            "intercept_rebounds": True,          # read bounce trajectory
        }
    },
    "physical": {
        "triggers": ["slap_usage > 0.2", "tackle_aggression > 0.7"],
        "counters": {
            "quick_release": True,               # one-touch before contact
            "body_positioning": True,            # shield ball, draw fouls
            "avoid_50_50": True,                 # don't fight fair duels
        }
    },
}
```

---

## 3. Possessor Decision Tree (Priority Order)

**In `_decide_possessor()` — replace current order with:**

1. **SHOOT** — if `shot_beats_keeper()` and lane clear (or striker in box with any angle)
2. **CROSS** — winger at byline (x > 48), striker in box → driven cross to penalty spot
3. **CUTBACK** — winger at byline, no striker → pull back to edge of box (x 32–40)
4. **THROUGH BALL** — striker/winger making run behind high line → weighted pass to space
5. **WALL PASS** — near touchline (margin 6m), wall lane clear → bounce to advancing runner
6. **SWITCH PLAY** — ball one flank, opposite winger free (x > 25) → long diagonal
6. **PULL BACK** — striker at edge of box (38–46), winger arriving → lay off for shot
7. **ONE-TWO** — under high press, teammate close (< 8m) and ahead → quick layoff
8. **THIRD MAN** — teammate with space ahead (no defender within 6m) → pass and run
9. **GK BYPASS** — GK has ball, press high → direct throw to wingers/striker
10. **CARRY** — dribble toward goal via `_dribble_target()` (probes lanes, avoids markers)
11. **SAFE PASS** — only if collectable (receiver within 15m of landing spot) and forward

**Key constraints:**
- Every pass target uses `pass_collection_point()` so ball is actually collectable
- Backward passes vetoed unless `target_x > p.x - 2.0` (small combinations only)
- Shots only when `beats_keeper == True` OR (striker in box AND any angle)
- GK never shoots/clears to own goal (veto via `_veto_own_goal`)

---

## 4. Off-Ball Movement (Role-Specific)

### Striker
| Situation | Target |
|-----------|--------|
| Ball in our half | x=45–54, channel between CB/FB (y 10–12 or 28–32) |
| Ball midfield, high line | Run in behind: x = defensive_line + 3–5 |
| Ball midfield, low block | Drop to pocket: x = ball.x + 8 (30–40) |
| Ball final third, low block | Attack near/far post if winger wide; else drop deep to create overload |
| Ball final third, high line | Threaten in-behind run; hold until passer ready |

### Wingers (Left/Right)
| Situation | Target |
|-----------|--------|
| Ball attacking third, low block | Drive to byline (x 45–52), stay wide (y 1–6 / 34–39) for cross |
| Byline, striker in box | Cross to penalty spot |
| Byline, no striker | Cut inside (y ±10) for shot/combo |
| Ball attacking third, high line | Run in behind fullback: x = their_highest + 5 |
| Ball midfield | Provide width, support build-up (x = ball.x + 5) |
| Ball opposite flank | Tuck in for overload (y ±5 toward centre) |

### Defender (Rest Defence)
| Situation | Target |
|-----------|--------|
| Standard | x = ball.x * 0.4 + 6 (8–22), split y = 15 / 25 |
| Counter-press active | x = ball.x - 5 (15–40), y = ball.y, speed 1.0 |
| Opponent high line | x = min(base + 3, 30) |
| Opponent presses high | x = base - 2 |
| Man-to-man assignment | Track assigned attacker goal-side, 4m gap |

---

## 5. Defensive Systems

### Press Triggers (in `plan_press()`)
- **PRESS**: Carrier facing own goal OR isolated (> 6m to nearest teammate) OR near wall OR ball in our defensive third (x < 24) and we're close (< 7m)
- **Max 2 pressers** (intensity ≥ 0.35) or 1 (lower intensity)
- **COVER**: Goal-side of ball, between ball and our goal
- **REST_DEFENCE**: Anchor at `defensive_line` (adaptive), never both CBs on centre line

### Counter-Press
- Trigger: we just won possession (`!prev_had_control && has_control_now`)
- Duration: 1 decision cycle
- Action: nearest 2 players sprint to ball carrier position (x = ball.x - 5)

### Wall Lane Blocking (`_block_wall_lanes()`)
- When opponent possessor near touchline (y < 8 or y > 32)
- Same-flank defender positions between ball and wall (3m from wall)
- Marks their support runner if present

### Wall Pass Intercept (`_intercept_wall_pass()`)
- Physics-based: mirror receiver across wall → line from possessor to mirror hits wall at contact point
- Position defender on rebound path (4m from wall toward receiver)
- Only same-flank defenders activate

---

## 6. Goalkeeper Intelligence

### Positioning
- **Standard**: Bisector between ball and goal centre (bias 0.62), clamped to defensive fifth (x ≤ 11)
- **Shot incoming** (ball.vx < -0.5): Intercept point = where ball crosses GK x; sprint at speed 1.0
- **1v1** (attacker past last defender, x < 18): Come out to x = 3–6, shade near post
- **Through ball**: Sweep if can reach meeting point before attacker (GK time + 0.3s buffer)
- **Cross**: Stand in corridor (x = 6), shade to cross side

### Distribution (when GK has ball, can_act)
1. **Counter-attack**: Opponent high line (deepest > 35), striker making run behind → throw to striker
2. **Best open teammate**: Score = progress * 1.5 + openness * 1.5 - travel * 0.2 + wide_bonus - mark_penalty
3. **Wall pass** to wide winger (x > 20) if lane clear
4. **Hoof to flank** (y = 8 or 32, x = 4) away from their GK

---

## 7. Implementation Checklist

### Files to Modify
- [ ] `src/policy.py` — Core decision tree, archetype counters, movement
- [ ] `src/opponent.py` — Enhanced archetype detection, counter-tactics
- [ ] `src/tactics.py` — Press triggers, defensive assignments
- [ ] `src/runtime.py` — Wire new archetype data to PolicyInput
- [ ] `src/config.py` — Genome defaults for Deleri FC style
- [ ] `tactics.json` — Deleri FC tactical defaults

### Validation Gates
- [ ] `python -m unittest discover -s tests -v` — all 194 tests pass
- [ ] `football-team build` — Docker build succeeds
- [ ] `football-team validate` — PASS
- [ ] Internal sim vs 7 archetypes — positive GD all
- [ ] Real engine vs 3 bundled — all wins
- [ ] Real engine vs training opponents — win rate > 70%

---

## 8. Testing Protocol

```bash
# 1. Unit tests
python -m unittest discover -s tests -v

# 2. Internal sim (fast, all archetypes)
python -c "
from src.sim import play_match, RuntimeManager
import random
mgr = RuntimeManager()
rng = random.Random(42)
for opp in ['possession','press','direct','defensive','counter','parkbus','wall']:
    r = play_match(mgr, opp, rng, decisions=600)
    print(f'{opp}: {r.score_us}-{r.score_them} (GD {r.goal_diff})')
"

# 3. Real engine vs bundled
for opp in reference reference-strikers slapstick-united; do
  football-team simulate --path . --opponent \$opp --games 3 --duration 60
done

# 4. Real engine vs training pool (sample)
for opp in high_press-elite low_block-elite counter-elite wall-elite; do
  football-team simulate --path . --opponent-path training/opponents/\$opp --games 2 --duration 60
done
```

---

## 9. Success Criteria

| Metric | Target |
|--------|--------|
| Unit tests | 194/194 pass |
| Internal sim vs 7 archetypes | All GD > 0 |
| Bundled opponents (3) | 100% win rate, avg GD ≥ 2 |
| Training elites (12) | Win rate ≥ 65% |
| Training pros (12) | Win rate ≥ 75% |
| No regression vs previous | All previous wins maintained |

---

**This plan is executable. Each section maps directly to code in `src/`. Begin with Phase 1: enhance `OpponentModel.detect_archetype()` and `get_counter_tactics()`, then rewire `_decide_possessor()` priority order.**