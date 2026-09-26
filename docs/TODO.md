# TODO — Big task list (mirrors ACTION_PLAN.md)

Legend: `[ ]` open · `[x]` done · `[~]` in progress · `[w]` waiting/blocked
Scope rule (AGENTS.md): only files inside `My Teams/my-team-fc/` may change.

---

## Phase A — Observability (engine ground truth)  PRIORITY: FIRST

- [ ] A1 Probe whether `football-team` surfaces our container stdout
  - [ ] A1.1 Add boot marker `print("MVTEAM-BOOT <artifact-hash>")` in `server.py` (dev-only)
  - [ ] A1.2 Build image
  - [ ] A1.3 Run one `simulate` and grep tool output for the marker
  - [ ] A1.4 Record finding in `experiments/engine-ground-truth.md`
- [ ] A2 Decision recorder (dev-only, non-mutating to decisions)
  - [ ] A2.1 Add counters: shoot/pass/tackle/clear/none
  - [ ] A2.2 Add per-decision context: tick, phase, score, ball pos, possession owner,
        our/opp players, emitted action, target, power
  - [ ] A2.3 Add `distance_to_goal`, `possessor_x`, `possessor_y`, `shot_choice_reason`
  - [ ] A2.4 Determinism guard: recorder must not consume hashlib/random/time state
- [ ] A3 Emission channel (choose after A1)
  - [ ] A3.1 stdout route: print JSON summary at `/v1/matches/end`
  - [ ] A3.2 fallback route: debug file in container path + `docker cp`
  - [ ] A3.3 fallback route: bind-mount dir / debug endpoint
- [ ] A4 Determinism/regression proof
  - [ ] A4.1 `validate` replays 25 fixtures identically (True)
  - [ ] A4.2 p50 latency unchanged vs baseline value recorded (0.8 ms)
- [ ] A5 Delphi: `experiments/engine-ground-truth.md` (one full match: who scored,
      decision count, action histogram, max distance-to-goal)

## Phase B — Propagation ladder L1→L8 (measure, don't guess)

Pick 2+ genomes with proven L1/L2 divergence (champion cb01cb51b59b vs extreme-shooter).

- [ ] B1 L1→L2: diff real-engine intent histograms between genomes (same seeds)
- [ ] B2 L2→L3/L4: diff possessor paths & ball positions from recorder data
- [ ] B3 L3/L4→L5: possession% from simulate summaries + recorder retention
- [ ] B4 L5→L6: cross-check `shoot_intent_count` vs engine `Shots:` counter
- [ ] B5 L6→L7→L8: goals/results from summaries
- [ ] B6 Report: `experiments/propagation-ladder.md` — fill L1–L8 table, name broken link

## Phase C — Root cause + minimal fix for the broken link

- [ ] C1 Design ONE discriminating test for the broken link's mechanism
- [ ] C2 Minimal fix in exactly one module
- [ ] C3 Regression: unit suite (87) + validate fixtures identical + re-measure Phase B
      ladder at the intended level

## Phase D — Gated evolution (only after ladder verifies propagation)

- [ ] D1 Propagation gate: forbid evolution/engine_campaign while last propagation report
      is stale
- [ ] D2 Small gated `engine_campaign`: same seeds as B, require ladder divergence
- [ ] D3 Keep champion artifact gated; promote only ladder-verified candidates

## Phase E — Experiment hygiene (kill the --skip-build class of bugs)

- [ ] E1 `scripts/preflight.py` — artifact-hash ↔ image-digest assertion before batteries
- [ ] E2 `scripts/restore-champion.sh` — restore cb01cb51b59b + rebuild + validate in one
      command
- [ ] E3 Extend `log_experiment` to record image digest + artifact hash per run
- [ ] E4 Move throwaway probes under `experiments/` with provenance noted

## Phase F — Work-habit enforcement (anti-give-up loop)

- [ ] F1 Every task starts from: smallest discriminating measurement
- [ ] F2 One hypothesis at a time; no new hypothesis until previous test resolved
- [ ] F3 When blocked: reduce to more granular data (never new subsystem)
- [ ] F4 Never ask user to resolve a discoverable question — present data instead
- [ ] F5 Scope discipline: park nice-to-haves in backlog, keep one bottleneck active

## Housekeeping

- [ ] Restore/verify artifact = cb01cb51b59b as final state (already on disk: confirm)
- [ ] Final unit suite run 87/87 before closing the session
- [ ] Update README? (only if requested — parked unless user asks)

---

## Backlog (parked, intentionally NOT in active queue)

- More scenario types (user liked 8-corpus; expand only after ladder verified)
- Extra scripted opponents (counter/parkbus/wall exist; more later)
- Deeper MCTS search-time budget tuning
- Multiple formations / role specialisation