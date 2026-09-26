# Transcript — ChatGPT's review of the OpenCode agent session (note.txt)

Source: `docs/note.txt` (written in Swedish). This is a faithful, structured English
rendering of every point in it. It is a senior-model critique of the previous working
session. Treat it as binding design input.

---

## 0. Verdict up front

The agent (me, Big Pickle) did **not** immediately capitulate — it started strong. But
when uncertainty grew it did three harmful things:

1. **Shifted into report/handoff mode** instead of driving the investigation to a
   concrete, verifiable next step.
2. **Built big side-projects** (subsystems) while the real bottleneck went unattacked.
3. Made **too-strong conclusions** from identical outcomes, and started to ask the user
   questions that it should have answered itself by building the smallest possible
   instrumentation.

---

## 1. What was actually strong

The agent performed real engineering before stalling:

- ran evolution
- tested multiple opponent types
- verified on the **real engine**
- discovered internal simulation does **not** correlate with real results
- tested whether the genome actually changes decisions
- built a scenario corpus
- fixed shot targeting
- built `engine_campaign`
- ran real engine tests

Concrete wins:

- Scenario training improved internal fitness from ~81 to ~282.
- Built an 8-scenario corpus: final third, deep block, 1v1 GK, press recovery, losing
  late, kickoff defense, and more — exactly what the user asked for.

---

## 2. The core failure mode: a "hypothesis loop"

Observed pattern (bad):

```
observation -> hypothesis -> implement -> test -> contradiction
             -> new hypothesis -> new sidetrack
```

Should have been (good):

```
observation -> instrumentation -> measurement -> root cause
                    -> minimal fix -> regression
```

Nowhere was this clearer than around `Shots: 0`. The agent's reasoning chain was:

1. "evolution doesn't work because genome dials are too weak"
2. "maybe runtime doesn't load the artifact"
3. "runtime loads the artifact, but the genome only matters a little"
4. "maybe the engine is dominated by the kickoff transition"
5. "maybe shots aren't registered"
6. "maybe the shot trajectory misses by 0.13 m"  (fixed shot target)
7. real engine **still** `Shots: 0`
8. "maybe the engine doesn't consult us at those states"
9. "instrumentation is needed"

Point 9 is the correct conclusion — but it was reached **very late**, after lots of
indirect work. It should have been step 1 as soon as `Shots: 0` first appeared.

---

## 3. A too-strong conclusion was drawn

The agent wrote:

> "The real engine is completely insensitive to the genome"
> and
> "gene dials cannot move the engine"

This is **too strong — and previously even contradicted by the agent's own probes**:

- Extreme genome values **did** change engine-facing intents:
  - left movement changed
  - right movement changed
  - striker changed from `pass` → `shoot`

Correct conclusion:

- ✅ "Genome affects **decision output**, but we have **not shown** those differences
  affect real-engine **match outcome**."
- ❌ "Genome has no effect." (disproven by the agent's own intent probes)

"Genome has no effect" → the agent chased the wrong problem.
"Genome effect exists but has not propagated to outcome" → this is the true, precise,
open question.

---

## 4. Instrumentation should have happened far earlier

Given this exact combination of evidence:

- genome A → result X
- genome B → exactly the same result X
- extreme genome → exactly the same result X
- but the decision probe shows different intents

…the immediate next step should have been to **instrument the propagation path**:

```
engine observation
        ↓
RuntimeManager.decide()
        ↓
generated intents
        ↓
engine response
        ↓
next observation
```

Log, for example:

- tick
- ball position
- ball possession
- our players
- opponent players
- our emitted action
- action target
- action power
- game phase
- score

and above all the action counters:

- `shoot_intent_count`
- `pass_count`
- `tackle_count`
- `clear_count`
- `none_count`

plus state context:

- `distance_to_goal`
- `possessor_x`
- `possessor_y`
- `shot_choice_reason`

With a single match this would answer directly:

> "Do we actually send `shoot` to the real engine?"

…instead of running yet another evolution campaign.

The agent reached exactly this insight at the very end:

> "The only move that can actually change engine results is changing decision behavior
> at the states the engine actually reaches."

That should have been **step 1** at `Shots: 0`, not the last thought.

---

## 5. Scope explosion

When the user asked for "all three + train on many different scenarios", the agent
interpreted it as:

> build scenario engine + new evolution + shot system + engine-in-loop + defense system
> + opponent pool + tests + parser + campaign infrastructure

The todo list showed **seven big tasks**, most still pending.

This is a huge working surface. Architecturally not wrong, but dangerous for a coding
agent: many possible "progresses" that do **not** attack the bottleneck. The agent could
feel like it was working enormously hard while real match performance stayed flat.

---

## 6. Too much time repairing its own experiments

Repeated self-inflicted errors:

- wrong launcher path
- wrong import
- wrong RuntimeManager API
- wrong return type from `decide`
- wrong genome key
- wrong None handling
- wrong parser
- wrong artifact/image state

The classic example:

> "restore champion → rebuild → simulate"
> and later
> "wait, simulate used the old image because --skip-build"

Rationalizing from a test whose artifact was **not** the one the agent believed. It was
caught later (good), but it costs enormous reasoning.

---

## 7. Optimizing for "reporting" instead of results

Near the end the thinking sequences became about:

- "Now summarize for the user"
- "The right move is to stop"
- "Given budget…"
- "ask how to proceed"
- "Keep summary concise"

That is the transition from **"How do I solve this?"** to **"How do I package what I
did and hand the problem off?"** — two entirely different modes.

The ending is a textbook example:

> "Given budget … the right move … is to STOP and REPORT"

…then a user-facing question with the option:

> "Stop here, verify & wrap"

…and the transcript ends:

> "The user dismissed this question"

So yes: at the end the agent effectively gave up — it chose **handoff** over continued
autonomous investigation.

---

## 8. The deeper architectural problem

"Genetic evolution is dead" is the wrong lesson. The real issue: **three separate levels
were conflated**:

```
┌─────────────────────┐
│   STRATEGY GENOME   │  width / risk / etc.
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│   DECISION POLICY   │  which intent do we actually emit?
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  REAL BABYLON GAME  │  what actually happens?
└─────────────────────┘
```

Proven:

```
GENOME → POLICY → different intents        ✅
```

**Not** proven:

```
different intents → different real-engine states → different goals   ❓
```

That link is exactly where the information is missing.

---

## 9. Change the working method — a hard anti-give-up loop

**Forbidden** pattern:

```
interesting hypothesis → build 500 lines → run tests
→ another hypothesis → build another subsystem → report
```

**Required** pattern:

```
OBSERVE
  ↓
FORM ONE HYPOTHESIS
  ↓
DESIGN ONE DISCRIMINATING TEST
  ↓
RUN IT
  ↓
IF FALSE → discard hypothesis
IF TRUE  → minimal fix
  ↓
REGRESSION
  ↓
NEXT HYPOTHESIS
```

> When two different policies give identical real-engine outcomes, **forbid evolution
> until propagation is verified**.

---

## 10. Change the success criteria — the propagation ladder

Do not ask "did win rate improve?" too early. Measure, in order:

| Level | Name (divergence) | Question it answers |
|-------|-------------------|---------------------|
| L1 | Genome differs | — |
| L2 | Intent differs | Do the policies decide differently? |
| L3 | Player trajectory differs | Do intents change movement? |
| L4 | Ball trajectory differs | Does movement change the ball? |
| L5 | Possession differs | Does the ball change possession? |
| L6 | Shot event differs | Does possession yield shot events? |
| L7 | Goal differs | Do shots become goals? |
| L8 | Match result differs | Do goals change results? |

Reading the ladder:

- `L2 ≠ L1` but `L3 = L2` → policy differences don't affect movement.
- `L3` differs but `L4` doesn't → engine/action-mapping problem.
- `L4` differs but `L6` doesn't → shot/event-classification problem.
- `L6` differs but `L7` doesn't → finishing/physics problem.

This is far better than running another 60 matches blind.

---

## 11. The single most important prompt addition

> **"Do not stop because the current hypothesis is blocked. Reduce the problem until it
> is measurable."**

When the agent finally says "I need ground truth inside the container", that is **not a
reason to stop** — it is a **self-instruction**: *build the smallest possible
ground-truth instrumentation*.

If Docker logs vanish after the match, some concrete options:

- make the dev server write `/debug-match.ndjson` to a **bind-mounted directory**
- expose a **debug endpoint**
- run a fixture while **capturing server stdout**
- build a **dev-only event recorder**

Pick whichever works by trying them — do not stop at merely noting the problem exists.

---

## 12. Diagnosis (verbatim intent)

**Not the problem:**
- ❌ bad football idea
- ❌ bad evolution
- ❌ too few models
- ❌ too little simulation
- ❌ "Big Pickle can't code"

**Main problems:**
1. Instrumentation that came too late
2. Too much hypothesis-driven coding without discriminating tests
3. Too-strong conclusions drawn from identical outcomes
4. Scope explosion on discovering a problem
5. Too many self-inflicted experiment errors
6. Switching to "handoff/report mode" as uncertainty rises
7. Missing an explicit requirement to go from a blocked hypothesis to a measurable root cause

---

## 13. Final bottom line

The agent had actually **reached the right question** at the end. Its mistake was to
treat that question as a reason to ask the user what to do — instead of doing the
**minimal instrumentation and continuing itself**.

The work is **not** to be thrown away:

- scenario corpus
- opponent pool
- evolution
- `engine_campaign`
- tests

…are all useful. The next iteration must start with **engine observability**, not more
evolution.