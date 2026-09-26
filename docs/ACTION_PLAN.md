# ACTION PLAN — making ChatGPT's ("Astra's") plan reality

This is the working plan that turns `docs/TRANSCRIPT_OF_NOTE.md` into concrete,
testable steps. The mandate from the note:

> **Instrument first. Measure the propagation ladder. Make minimal fixes. Do not stop
> because a hypothesis is blocked — reduce the problem until it is measurable.**

## Operative rules (bind, re-read before every step)

R1. **Anti-give-up loop.** Every investigation follows:

```
OBSERVE
  → FORM ONE HYPOTHESIS
  → DESIGN ONE DISCRIMINATING TEST
  → RUN IT
  → IF FALSE discard, IF TRUE minimal fix
  → REGRESSION
  → NEXT HYPOTHESIS
```

R2. **Never conclude "the engine is insensitive to the genome".** The proven claim is
L1→L2 (genome→intent). The open question is propagation L2→L8. State only what is
measured.

R3. **Propagation gate.** While two policies give identical real-engine outcomes,
evolution is **forbidden**. Evolution resumes only after a code-level change is shown to
produce divergence somewhere on the ladder AND that divergence reaches outcome.

R4. **Artifact–image integrity.** Every engine experiment starts by asserting the image
digest actually contains the intended artifact hash (see E1). The `--skip-build` class
of errors is unacceptable.

R5. **Smallest possible instrumentation.** When evidence is missing, build the minimal
recorder that answers the question, not a subsystem.

R6. **No handoff mode.** A blocking question that is *discoverable* must be answered by
instrumentation, not by asking the user.

---

## Phase A — Observability (engine ground truth)  [START HERE]

Goal: know exactly what the real engine asks us, what we emit, and what state surrounds
every decision.

- [ ] A1 **Probe stdout surfacing**: add a one-line boot marker (`print("MVTEAM-BOOT …")`)
      in `server.py`, build the image, run one `simulate`, and grep the tool output for
      the marker. Determines whether our container stdout is readable at all.
- [ ] A2 **Decision recorder (dev-only)**: in `server.py`/`runtime`, collect per-decision
      records — NON-mutating w.r.t. decisions (hash-seeded, no time/RNG influence):
      - tick, game phase, score
      - ball position, possession owner
      - our players, opponent players (consignment/roles)
      - emitted action, action target, action power
      - `shoot_intent_count`, `pass_count`, `tackle_count`, `clear_count`, `none_count`
      - `distance_to_goal`, `possessor_x`, `possessor_y`, `shot_choice_reason`
- [ ] A3 **Emission channel** (pick by A1 result):
      - (i) container stdout is surfaced → print JSON summary + N last records at
        `/v1/matches/end`;
      - (ii) otherwise: `docker cp` a debug file written to a known path, or a bind-mount
        dir, or a debug endpoint read via `docker exec`.
- [ ] A4 **Determinism guard**: prove recorder changes nothing — `validate` must replay
      25 fixtures identically and p50 latency unchanged.
- [ ] A5 **Deliverable**: `experiments/engine-ground-truth.md` with one real match fully
      documented (who scored, how many decides, action histogram, max distance-to-goal
      across decisions).

## Phase B — Propagation ladder L1→L8 (measure the chain)

Goal: find the FIRST broken link, with evidence, not guesswork. Two or more genomes with
known L1/L2 divergence (e.g., current champion vs the extreme-shooter build).

- [ ] B1 **L1→L2 (genome→intent)**: run recorder on the real engine, both genomes, same
      seeds. Diff action histograms + intent sequences. (Ground truth: do they diverge on
      the engine at all?)
- [ ] B2 **L2→L3/L4 (intent→movement/ball)**: from recorder fields, diff possessor paths
      and ball positions over ticks for the two genomes. (If identical, policy intents are
      not reaching movement.)
- [ ] B3 **L3/L4→L5 (movement→possession)**: possession% from simulate summaries, plus
      retention from recorder.
- [ ] B4 **L5→L6 (possession→shot events)**: cross-check our `shoot_intent_count` against
      the engine's `Shots:` counter. (If we emit `shoot` but engine shows 0 shots → shot
      classification/positional problem. If we never emit `shoot` → shot-trigger problem.)
- [ ] B5 **L6→L7→L8 (shots→goals→result)**: goals and match results from summaries.
- [ ] B6 **Deliverable**: `experiments/propagation-ladder.md` — empirically fills the L1–L8
      table and names the broken link. Record every metric; no narrative-only conclusions.

## Phase C — Root cause + minimal fix for the broken link

- [ ] C1 One discriminating test that isolates the broken link's mechanism.
- [ ] C2 Minimal fix in exactly one module (no new subsystems, no side-building).
- [ ] C3 Regression: unit suite (87 currently), validate fixtures identical, then repeat
      the Phase B measurement to confirm the ladder now diverges at the intended level.

## Phase D — Gated evolution (only after propagation is verified)

- [ ] D1 Codify the propagation gate (checkbox in the experiment checklist; a small script
      `scripts/hygiene.sh` refuses `engine_campaign`/evolution when the last propagation
      report is stale).
- [ ] D2 Resume with a SMALL gated `engine_campaign` (few genomes, same seeds as B),
      requiring ladder divergence before any promotion. Keep the champion artifact gated.

## Phase E — Experiment hygiene (automate away self-inflicted errors)

- [ ] E1 **Preflight script** `scripts/preflight.py`:
      - assert `DISTILLED_POLICY_JSON` artifact hash == expected (from provenance file);
      - build image and record digest;
      - assert the built image really loads that artifact (start container, read the debug
        endpoint or a build-time injected env echo);
      - only then allow a battery.
- [ ] E2 **State restore script** `scripts/restore-champion.sh`: restore `cb01cb51b59b`,
      rebuild, validate — single command, no manual `cp`+`build` sequences.
- [ ] E3 Move any throwaway probes to `experiments/` with a lo-NOTE that records what image
      digest + artifact hash every result corresponds to.

## Phase F — Working-habits enforcement (the "hard anti-give-up" loop)

- [ ] F1 Always start from "what is the smallest measurement that discriminates?"
- [ ] F2 One hypothesis at a time; a new hypothesis only after the previous test resolved.
- [ ] F3 When blocked: reduce (more granular data), never escalate to a new subsystem.
- [ ] F4 Never ask the user to resolve a discoverable question; present data instead.
- [ ] F5 Keep scope to the bottleneck; park nice-to-haves in TODO backlog, not in the live
      work queue.

---

## Evidence/report files (all under `experiments/`, inside scope)

- `experiments/engine-ground-truth.md`      — Phase A deliverable
- `experiments/propagation-ladder.md`       — Phase B deliverable
- `experiments/experiments.ndjson`          — machine-readable run provenance (existing
                                             `log_experiment` extended with image digest +
                                             artifact hash)
- `experiments/hygiene.md`                  — preflight output log

## Definition of done for this sprint

1. We know, from one instrumented real match, whether we ever emit `shoot`, how often,
   and under what states.
2. The propagation ladder L1→L8 is filled with real-engine data; the broken link is named
   with evidence.
3. Exactly one minimal fix is in, regression passes, and the ladder now shows divergence
   at the intended level.
4. Champion artifact remains `cb01cb51b59b` (or a strictly better, ladder-verified
   candidate), image reproducible via a one-command restore script.