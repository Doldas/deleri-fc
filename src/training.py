"""Offline training driver: evolution → MCTS demo collection → distillation.

Bridges `evolution.evolve` (genome search over simulated matches) with the
runtime artifact `artifacts/policies/distilled_policy.json` that
`runtime.load_evolved_genome()` consumes at startup (AISTRATEGI §64-66).

Also collects MCTS-overridden decisions (demos) so a supervised approach can
later learn from them, and logs every experiment with full provenance
(AGENTS.md §7). Never imported by the decision path.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import GENOME_KEYS, RuntimeConfig, REWARD_DEFAULTS, default_genome, genome_hash
from .evolution import PoolEntry, evolve
from .light import LightEngine, make_state, LIntent
from .runtime import RuntimeManager
from .sim import BASE_LINEUP, SimResult, mirrored_lineup, OPPONENTS, plan_to_obs, play_match

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
POLICY_DIR = ARTIFACT_DIR / "policies"
DISTILLED_POLICY_JSON = POLICY_DIR / "distilled_policy.json"
CAMPAIGN_POLICY_JSON = POLICY_DIR / "candidate.json"
DEMO_LOG = ARTIFACT_DIR / "demos.ndjson"
EXPERIMENT_LOG = ARTIFACT_DIR / "experiments.ndjson"

ENGINE_VERSION = "1.0"  # protocolVersion we build against
CODE_VERSION = "1.1.0"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_experiment(kind: str, **payload) -> None:
    """Append a provenance-complete experiment record (AGENTS.md §7)."""
    EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "kind": kind,
        "engineVersion": ENGINE_VERSION,
        "codeVersion": CODE_VERSION,
        "at": _now(),
        **payload,
    }
    with EXPERIMENT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


@dataclass
class RunReport:
    summary: dict
    artifact_path: Path | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = dict(self.summary)
        out["artifact_path"] = str(self.artifact_path) if self.artifact_path else None
        out.update(self.extra)
        return out


def run_evolution(
    seed: int = 42,
    generations: int = 12,
    population: int = 8,
    elite: int = 2,
    opponent: str = "possession",
    decisions: int = 300,
    distill_to_artifact: bool = True,
    seed_genome: dict[str, float] | None = None,
) -> RunReport:
    """Full evolution run with experiment provenance + distillation."""
    rng = random.Random(seed)
    summary = evolve(
        rng,
        generations=generations,
        population=population,
        elite=elite,
        opponent=opponent,
        decisions=decisions,
        seed_genome=seed_genome,
    )
    payload = summary.to_dict()
    log_experiment("evolution", seed=seed, opponent=opponent, decisions=decisions, **payload)

    artifact_path = None
    if distill_to_artifact:
        artifact_path = distill(summary.best_genome)
    return RunReport(summary=payload, artifact_path=artifact_path)


def distill(genome: dict[str, float] | None = None, out_path: Path | None = None) -> Path:
    """Write the distilled policy artifact consumed by the runtime.

    The artifact is: the converged genome + reward weights + metadata. When it
    exists, `RuntimeManager` prefers it over the stock genome at startup.
    """
    genome = dict(genome or default_genome())
    out_path = out_path or DISTILLED_POLICY_JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schemaVersion": "1.0",
        "engineVersion": ENGINE_VERSION,
        "codeVersion": CODE_VERSION,
        "generatedAtUtc": _now(),
        "genomeHash": genome_hash(genome),
        "genome": {k: round(float(genome[k]), 6) for k in GENOME_KEYS if k in genome},
        "rewardWeights": dict(REWARD_DEFAULTS),
    }
    out_path.write_text(json.dumps(payload, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    return out_path


def _opp_label(entry: PoolEntry) -> str:
    if isinstance(entry, tuple) and entry[0] == "champion":
        return f"champion:{genome_hash(entry[1])[:8]}"
    return str(entry)


def campaign(
    seed: int = 42,
    generations: int = 12,
    population: int = 8,
    elite: int = 2,
    decisions: int = 300,
    opponents: list[PoolEntry] | None = None,
    styles: list[str] | None = None,
    hof: list[dict[str, float]] | None = None,
    distill_candidate: bool = True,
    seed_genome: dict[str, float] | None = None,
    scenario_lab: list[tuple[str, str]] | None = None,
    scenario_ticks: int | None = None,
) -> RunReport:
    """Evolve against a pool of archetypes AND past champions (§33/§35/§36).

    `opponents` is a list of built-in archetype names (`str`) and/or
    `("champion", genome)` tuple entries for head-to-head self-play. The
    resulting best genome is written to `artifacts/policies/candidate.json`
    (never overwrites the live distilled artifact), so it can be verified on
    the real engine via `football-team simulate --image ... --opponent <name>`
    before `promote_candidate` ships it as the new champion. `scenario_lab`
    switches fitness to the deterministic skill scenarios.
    """
    pool = list(opponents) if opponents else ["possession"]
    rng = random.Random(seed)
    summary = evolve(
        rng,
        generations=generations,
        population=population,
        elite=elite,
        opponent=pool[0] if isinstance(pool[0], str) else "possession",
        decisions=decisions,
        seed_genome=seed_genome,
        opponents=pool,
        styles=styles,
        hof=hof,
        scenario_lab=scenario_lab,
        scenario_ticks=scenario_ticks,
    )
    payload = summary.to_dict()
    log_experiment("campaign", seed=seed, opponents=[_opp_label(o) for o in pool], decisions=decisions, **payload)

    artifact_path = None
    if distill_candidate:
        artifact_path = distill(summary.best_genome, CAMPAIGN_POLICY_JSON)
        log_experiment(
            "candidate_written",
            seed=seed,
            path=str(artifact_path),
            genomeHash=genome_hash(summary.best_genome),
            hof_passed=summary.hof_passed,
            hof_details=summary.hof_details or {},
        )
    return RunReport(summary=payload, artifact_path=artifact_path)


def promote_candidate(candidate_path: Path | None = None, out_path: Path | None = None) -> Path:
    """Promote a real-engine-verified candidate into the live artifact.

    The candidate file (written by `campaign`) must first pass the real-engine
    simulate gate; this just ships the verified genome to the runtime.
    """
    candidate_path = candidate_path or CAMPAIGN_POLICY_JSON
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    out = distill(payload["genome"], out_path)
    log_experiment("promote", source=str(candidate_path), dest=str(out), genomeHash=genome_hash(payload["genome"]))
    return out


def _football_team() -> Path:
    """Locate the `football-team` launcher (repo tool) for engine-in-loop runs."""
    env = os.environ.get("FOOTBALL_TEAM")
    if env:
        return Path(env)
    candidates = []
    here = Path(__file__).resolve()
    # src/training.py -> team root -> starter root
    candidates.append(here.parents[3] / "football-team")
    candidates.append(here.parents[2] / "football-team")
    for c in candidates:
        if c.exists():
            return c
    return Path("football-team")


def _run_tool(team_dir: Path, args: list[str], timeout: int = 900) -> str:
    proc = subprocess.run(
        [str(_football_team()), *args],
        cwd=str(team_dir),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return (proc.stdout or "") + (proc.stderr or "")


def _parse_sim_summary(output: str) -> dict:
    """Extract candidate results from `football-team simulate` output.

    In non-interactive (CI) mode the tool prints one `[candidate i/M] ...` line
    per match plus the rich aggregate block when a TTY is attached; parse the
    per-match lines (always present) and fall back to the aggregate block.
    """
    res: dict = {
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "matches": 0,
        "goals_for": 0,
        "goals_against": 0,
    }
    per_match = re.findall(r"\[candidate \d+/\d+]\s+(\d+)-(\d+)\s+(win|loss|draw)", output)
    if per_match:
        for gf, ga, outcome in per_match:
            res["matches"] += 1
            res["goals_for"] += int(gf)
            res["goals_against"] += int(ga)
            if outcome == "win":
                res["wins"] += 1
            elif outcome == "loss":
                res["losses"] += 1
            else:
                res["draws"] += 1
        res["win_rate_pct"] = round(100.0 * res["wins"] / max(1, res["matches"]), 1)
        res["wdl"] = f"{res['wins']}/{res['draws']}/{res['losses']}"
        m = re.search(r"Possession:\s+([\d.]+)%", output)
        if m:
            res["possession_pct"] = float(m.group(1))
        m = re.search(r"Shots:\s+(\d+)-(\d+)", output)
        if m:
            res["shots_for"], res["shots_against"] = int(m.group(1)), int(m.group(2))
        return res
    # TTY fallback: aggregate summary block.
    for pat, key in (
        (r"W/D/L:\s+([\d/]+)", "wdl"),
        (r"Goals:\s+(\d+)-(\d+)", "goals"),
        (r"Possession:\s+([\d.]+)%", "possession_pct"),
        (r"Shots:\s+(\d+)-(\d+)", "shots"),
    ):
        m = re.search(pat, output)
        if m:
            if key == "wdl":
                w, d, l = (int(x) for x in m.group(1).split("/"))
                res.update(wins=w, draws=d, losses=l, matches=w + d + l)
            elif key == "goals":
                res["goals_for"], res["goals_against"] = int(m.group(1)), int(m.group(2))
            elif key == "shots":
                res["shots_for"], res["shots_against"] = int(m.group(1)), int(m.group(2))
            else:
                res[key] = float(m.group(1))
    if "matches" not in res:
        res["matches"] = 0
    return res


def engine_campaign(
    genomes: list[dict[str, float]],
    label: str = "engine-campaign",
    games: int = 2,
    sides: str = "both",
    opponents: list[str] | None = None,
    seed_prefix: str = "engcamp",
    team_dir: Path | None = None,
) -> list[dict]:
    """Score candidate genomes on the REAL engine via `football-team simulate`.

    This is the engine-in-the-loop fitness that internal-sim evolution cannot
    fake: each candidate is written to the candidate artifact, the image is
    rebuilt, and the simulation gate runs against the generated opponents.
    Slow (~minutes per genome) but authoritative (AGENTS.md §2, §7).

    `opponents` should be a list of generated opponent directory names
    (e.g., "high_press-elite") under the repo's `opponents/` directory.

    Returns one summary dict per genome (goals, possession, shots, wins) and
    logs an experiment record with full provenance.
    """
    # Repo root is the parent of the team directory's parent
    REPO = Path(__file__).resolve().parents[3]
    OPPONENTS_DIR = REPO / "opponents"
    opponents = opponents or ["reference", "reference-strikers"]
    team_dir = team_dir or Path(__file__).resolve().parents[1]
    out: list[dict] = []
    for genome in genomes:
        distill(genome, CAMPAIGN_POLICY_JSON)
        _run_tool(team_dir, ["build", "--no-cache"])
        row: dict = {"genomeHash": genome_hash(genome), "label": label}
        for opp in opponents:
            opp_path = OPPONENTS_DIR / opp
            if not opp_path.is_dir():
                row[f"{opp}.wdl"] = "ERROR"
                row[f"{opp}.goals"] = "N/A"
                continue
            output = _run_tool(
                team_dir,
                ["simulate", "--path", str(team_dir), "--opponent-path", str(opp_path),
                 "--games", str(games), "--duration", "60", "--sides", sides, "--skip-build",
                 "--seed-prefix", seed_prefix],
            )
            s = _parse_sim_summary(output)
            row[f"{opp}.wdl"] = s.get("wdl", "0/0/0")
            row[f"{opp}.goals"] = f"{s.get('goals_for', 0)}-{s.get('goals_against', 0)}"
            row[f"{opp}.matches"] = s.get("matches", 0)
            row[f"{opp}.possession"] = s.get("possession_pct")
            row[f"{opp}.shots"] = (
                f"{s.get('shots_for', 0)}-{s.get('shots_against', 0)}" if "shots_for" in s else None
            )
        out.append(row)
    log_experiment("engine_campaign", label=label, seed_prefix=seed_prefix, games=games, sides=sides, results=out)
    return out


@dataclass
class MCTSCollectReport:
    decisions: int
    mcts_override_count: int
    goal_diff: int
    events: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "decisions": self.decisions,
            "mcts_override_count": self.mcts_override_count,
            "mcts_override_rate": round(self.mcts_override_count / max(1, self.decisions), 4),
            "goal_diff": self.goal_diff,
            "events": self.events,
        }


def collect_mcts_demos(
    seed: int = 7,
    opponent: str = "possession",
    decisions: int = 300,
    max_demos: int = 128,
    max_iterations: int = 8,
) -> MCTSCollectReport:
    """Step the LightEngine, comparing a base manager vs one with MCTS on.

    Decisions where MCTS overrode the policy are the interesting distillation
    targets; a bounded sample is stored to `DEMO_LOG` for supervised training.
    """
    rng = random.Random(seed)
    engine = LightEngine(seed=rng.randint(0, 2**31 - 1))
    st = make_state(
        [(p, r, x, y) for p, r, x, y in BASE_LINEUP],
        mirrored_lineup(),
        ball=(30.0, 20.0),
        possess=("us", "am"),
    )
    base_mgr = RuntimeManager()
    base_mgr.genome_base = dict(base_mgr.genome_base)
    mcts_mgr = RuntimeManager(runtime_config=RuntimeConfig(enable_mcts=True, mcts_iterations=max_iterations))
    opponent_ctrl = OPPONENTS[opponent](rng)

    override_count = 0
    demos = 0
    events: dict[str, int] = {}
    for step in range(decisions):
        obs = plan_to_obs(st, f"demo-{seed}", step)
        base_decision = base_mgr.decide(obs)
        mcts_decision = mcts_mgr.decide(obs)
        overridden = base_decision.get("intents") != mcts_decision.get("intents")
        if overridden:
            override_count += 1
        if overridden and demos < max_demos:
            DEMO_LOG.parent.mkdir(parents=True, exist_ok=True)
            with DEMO_LOG.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "game": f"demo-{seed}",
                            "sequence": step,
                            "ball": [st.ball.x, st.ball.y],
                            "possess": [st.ball.possessing_team, st.ball.possessing_player],
                            "mcts_intents": json.dumps(mcts_decision["intents"]),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            demos += 1

        merged: dict[tuple[str, str], LIntent] = {}
        for intent in mcts_decision.get("intents", []):
            pid = intent["playerId"]
            mv = intent.get("move", {})
            tgt = mv.get("target", {})
            act = intent.get("action", {})
            at = act.get("target")
            merged[("us", pid)] = LIntent(
                tx=tgt.get("x", st.ball.x),
                ty=tgt.get("y", st.ball.y),
                speed=mv.get("speed", 0.5),
                act=act.get("type", "none"),
                action_target=(at["x"], at["y"]) if at else None,
                power=act.get("power"),
            )
        merged.update(opponent_ctrl.decide(st, "them"))
        st = engine.step(st, merged)
        for e in st.events:
            events[e] = events.get(e, 0) + 1

    report = MCTSCollectReport(decisions=decisions, mcts_override_count=override_count, goal_diff=st.score_us - st.score_them, events=events)
    log_experiment("mcts_demos", seed=seed, opponent=opponent, **report.to_dict())
    return report


def baseline_vs_opponents(seed: int = 3, decisions: int = 300) -> dict[str, dict]:
    """Score the current best (or stock) genome against every archetype."""
    rng = random.Random(seed)
    manager = RuntimeManager()
    out: dict[str, dict] = {}
    for opponent in OPPONENTS:
        res, _ = evaluate_genome_for(manager, opponent, rng, decisions)
        out[opponent] = res.summary()
    log_experiment("baseline", seed=seed, opponents=sorted(OPPONENTS), results=out)
    return out


def evaluate_genome_for(manager, opponent: str, rng: random.Random, decisions: int) -> tuple[SimResult, float]:
    """Small helper: play one match with the manager's current genome."""
    from .evolution import fitness_from

    res = play_match(manager, opponent, rng=rng, decisions=decisions)
    return res, fitness_from(res)


def baseline_report(seed: int = 3, decisions: int = 300) -> list[dict]:
    """One result row per scripted opponent (static-suite API: returns a list)."""
    rng = random.Random(seed)
    manager = RuntimeManager()
    rows = []
    for opponent in OPPONENTS:
        res, _ = evaluate_genome_for(manager, opponent, rng, decisions)
        row = res.summary()
        row["opponent"] = opponent
        row["seed"] = seed
        rows.append(row)
    log_experiment("baseline", seed=seed, decisions=decisions, rows=rows)
    return rows


__all__ = [
    "run_evolution",
    "campaign",
    "promote_candidate",
    "engine_campaign",
    "distill",
    "collect_mcts_demos",
    "baseline_vs_opponents",
    "baseline_report",
    "RunReport",
    "MCTSCollectReport",
    "DISTILLED_POLICY_JSON",
    "CAMPAIGN_POLICY_JSON",
]