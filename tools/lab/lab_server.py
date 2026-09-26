#!/usr/bin/env python3
"""Match Lab: a practice viewer that knows about the generated opponents.

Why this exists
---------------
`football-team practice` cannot be pointed at the teams in `opponents/`. Its
opponent list is a closed enum compiled into the shipped .NET CLI
(`--opponent must be reference, reference-strikers, or slapstick-united`), the
three CPUs are compiled into a prebuilt worker image, and the practice viewer is
a compiled JS bundle. This workspace contains no C#, C++ or JS source for any of
those layers, and no .NET SDK or decompiler, so the dropdown cannot be extended
in place. See README.md for the full evidence.

So this serves the same goal by a supported route: it scans `opponents/`, and for
the chosen side it runs the real authoritative engine via
`football-team simulate --opponent-path`, captures the replay the engine
produces, and renders it here. Same engine, same rules, same determinism as the
official practice worker -- only the opponent choice and the rendering are ours.

    python tools/lab/lab_server.py [--port 5177]

Then open http://127.0.0.1:5177/
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPO = ROOT.parent.parent
CLI = REPO / "football-team"
OPPONENTS = REPO / "opponents"
STATIC = pathlib.Path(__file__).resolve().parent / "static"
WORK = ROOT / ".lab"

# The only opponents the kit can run without a source directory.
BUNDLED = {
    "reference": "Reference Athletic",
    "reference-strikers": "Reference Strikers",
    "slapstick-united": "Slapstick United",
}

# One worker: two concurrent `simulate` runs would fight over the same Docker
# image tags and the same `.lab` replay path.
POOL = ThreadPoolExecutor(max_workers=1)

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# Opponent discovery
# --------------------------------------------------------------------------

def read_team_literal(server_py: pathlib.Path) -> dict | None:
    """Pull the `TEAM` dict out of a server.py without executing it.

    `server.py` calls `serve_forever()` at module level, so it cannot be
    imported. The worker reads identity from `/v1/team`, which this literal
    feeds, so it is the authoritative source for a team's display name and kit.
    """
    try:
        tree = ast.parse(server_py.read_text())
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "TEAM":
                try:
                    value = ast.literal_eval(node.value)
                except ValueError:
                    return None
                return value if isinstance(value, dict) else None
    return None


def family_of(team_id: str) -> tuple[str, str]:
    """('low_block', 'elite') for 'low_block-elite'; ('lure', 'mimic') for
    'lure-mimic'. Used only for grouping in the UI."""
    if team_id.startswith("lure-"):
        return "lure", team_id[len("lure-"):]
    if team_id.startswith("learned-"):
        return "learned", team_id[len("learned-"):]
    if "-" in team_id:
        head, _, tail = team_id.rpartition("-")
        return head, tail
    return team_id, ""


BAND_ORDER = {"rookie": 0, "solid": 1, "pro": 2, "elite": 3}


def scan_opponents() -> list[dict]:
    """Every playable opponent: the generated pool plus the three bundled CPUs."""
    found: list[dict] = []
    for team_id, label in BUNDLED.items():
        found.append({
            "id": team_id, "name": label, "shortName": "",
            "family": "bundled", "variant": "", "group": "Bundled CPUs",
            "colors": None, "generated": False, "exists": True,
        })
    if OPPONENTS.is_dir():
        for directory in sorted(OPPONENTS.iterdir()):
            manifest = directory / "football-team.json"
            if not directory.is_dir() or not manifest.exists():
                continue
            try:
                data = json.loads(manifest.read_text())
            except (OSError, ValueError):
                continue
            team = read_team_literal(directory / "server.py") or {}
            colors = (team.get("colors") or {})
            family, variant = family_of(directory.name)
            found.append({
                "id": directory.name,
                "name": team.get("name") or data.get("name") or directory.name,
                "shortName": team.get("shortName", ""),
                "family": family,
                "variant": variant,
                "group": family,
                "colors": {"primary": colors.get("primary"),
                           "secondary": colors.get("secondary")} or None,
                "generated": True,
                "exists": True,
            })
    found.sort(key=lambda o: (0 if o["group"] == "Bundled CPUs" else 1,
                              o["group"],
                              BAND_ORDER.get(o["variant"], 9),
                              o["id"]))
    return found


# --------------------------------------------------------------------------
# Replay trimming
# --------------------------------------------------------------------------

def trim_replay(raw: dict, step: int = 2) -> dict:
    """Compress an engine replay into something a browser can hold.

    A 60 s replay is ~10 MB of verbose per-tick observations. The viewer only
    needs positions, so this flattens to column arrays and halves the frame rate
    (20 Hz -> 10 Hz, still far above what the eye reads on a 60x40 pitch). The
    result is ~100x smaller and parses instantly.
    """
    snapshots = raw.get("snapshots") or []
    if not snapshots:
        return {"meta": {"frames": 0}, "frames": {}}

    first = snapshots[0]["teams"]
    # Teams arrive as a neutral absolute-coordinate pair; team-a defends x=0.
    side_of = {t["id"]: ("a" if t["id"].endswith("-a") else "b") for t in first}

    def player_order(team: dict) -> list[str]:
        return [p["id"] for p in
                sorted(team["players"],
                       key=lambda p: (p["role"] != "goalkeeper", p["id"]))]

    order = {side_of[t["id"]]: player_order(t) for t in first}

    def team_by(snapshot: dict, side: str) -> dict:
        for t in snapshot["teams"]:
            if side_of.get(t["id"]) == side:
                return t
        return {"players": [], "score": 0}

    frames = snapshots[::max(1, step)]
    out: dict[str, list] = {"phase": [], "time": [], "scoreA": [], "scoreB": [],
                             "ball": [], "owner": [], "a": [], "b": [],
                             "facingA": [], "facingB": []}
    for snapshot in frames:
        out["phase"].append(snapshot.get("phase", "openPlay"))
        out["time"].append(round(float(snapshot.get("timeRemainingSeconds", 0)), 2))
        a, b = team_by(snapshot, "a"), team_by(snapshot, "b")
        out["scoreA"].append(a.get("score", 0))
        out["scoreB"].append(b.get("score", 0))

        ball = snapshot.get("ball") or {}
        pos = ball.get("position") or {}
        out["ball"].extend([round(float(pos.get("x", 30)), 3),
                            round(float(pos.get("y", 20)), 3)])
        out["owner"].append(0 if ball.get("possessingTeam") == "team-a"
                            else (1 if ball.get("possessingTeam") == "team-b" else -1))

        for side in ("a", "b"):
            by_id = {p["id"]: p for p in team_by(snapshot, side)["players"]}
            flat: list[float] = []
            for pid in order[side]:
                p = by_id.get(pid) or {}
                ppos = p.get("position") or {}
                flat.append(round(float(ppos.get("x", 30)), 2))
                flat.append(round(float(ppos.get("y", 20)), 2))
                flat.append(1.0 if p.get("canAct", True) else 0.0)
            out[side].extend(flat)
            out[f"facing{side.upper()}"].extend(
                round(float((by_id.get(pid) or {}).get("facingRadians", 0.0)), 3)
                for pid in order[side])

    # Goals are the only thing a viewer must not miss, and they are the one
    # thing the snapshot stream does not label: detect them from score changes
    # so the timeline can mark them even when the event list is empty.
    goals = []
    for i in range(1, len(frames)):
        for side, key, other in (("a", "scoreA", "scoreB"), ("b", "scoreB", "scoreA")):
            if out[key][i] > out[key][i - 1]:
                goals.append({"frame": i, "side": side,
                              "score": [out["scoreA"][i], out["scoreB"][i]],
                              "time": out["time"][i]})

    meta_team = {t["id"]: t for t in first}
    home = meta_team.get("team-a", {})
    away = meta_team.get("team-b", {})
    return {
        "meta": {
            "frames": len(frames),
            "step": max(1, step),
            "opponentId": raw.get("opponentId"),
            "gameId": (raw.get("fixture") or {}).get("gameId"),
            "seed": (raw.get("fixture") or {}).get("seed"),
            "home": {"id": home.get("id"), "name": home.get("name"),
                     "primary": home.get("primaryColor"),
                     "secondary": home.get("secondaryColor")},
            "away": {"id": away.get("id"), "name": away.get("name"),
                     "primary": away.get("primaryColor"),
                     "secondary": away.get("secondaryColor")},
            "goals": goals,
            "order": {"a": order["a"], "b": order["b"]},
        },
        "frames": out,
    }


# --------------------------------------------------------------------------
# Running a match
# --------------------------------------------------------------------------

RESULT_RE = re.compile(r"Goals: (.+?)\s+Average: (\S+)\s+Difference: ([+-]?\d+)")
POSSESSION_RE = re.compile(r"Possession: ([\d.]+)%")
SHOTS_RE = re.compile(r"Shots: (\d+)")
MISSED_RE = re.compile(r"Missed decisions: (\d+)")


def run_match(opponent_id: str, duration: int, step: int) -> dict:
    """Run one authoritative match and return a trimmed replay."""
    if not CLI.exists():
        raise RuntimeError(f"CLI not found at {CLI}")
    WORK.mkdir(parents=True, exist_ok=True)
    replay_path = WORK / f"replay-{uuid.uuid4().hex[:12]}.json"

    # No --skip-build: the opponent image must be rebuilt from disk so identity
    # and strategy edits are actually picked up. Docker layer caching keeps
    # repeat runs fast when nothing changed.
    cmd = [str(CLI), "simulate", "--path", str(ROOT),
           "--games", "1", "--duration", str(duration),
           "--replay", str(replay_path)]
    if opponent_id in BUNDLED:
        cmd += ["--opponent", opponent_id]
    else:
        if not (OPPONENTS / opponent_id / "football-team.json").exists():
            raise RuntimeError(f"no opponent directory for {opponent_id!r}")
        cmd += ["--opponent-path", str(OPPONENTS / opponent_id)]

    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip()[-1200:]
        raise RuntimeError(f"simulate failed (exit {proc.returncode}):\n{tail}")
    if not replay_path.exists():
        raise RuntimeError("engine produced no replay; nothing to show")

    try:
        raw = json.loads(replay_path.read_text())
    finally:
        replay_path.unlink(missing_ok=True)

    trimmed = trim_replay(raw, step=step)
    summary = {"goals": "", "possession": None, "shots": None, "missed": None}
    m = RESULT_RE.search(proc.stdout)
    if m:
        summary["goals"] = m.group(1)
    for key, pattern in (("possession", POSSESSION_RE), ("shots", SHOTS_RE),
                         ("missed", MISSED_RE)):
        found = pattern.search(proc.stdout)
        if found:
            summary[key] = float(found.group(1))
    trimmed["meta"]["summary"] = summary
    trimmed["meta"]["wallTime"] = round(time.time() - started, 1)
    trimmed["meta"]["duration"] = duration
    return trimmed


def job_worker(job_id: str, opponent_id: str, duration: int, step: int) -> None:
    with JOBS_LOCK:
        JOBS[job_id]["state"] = "running"
    try:
        result = run_match(opponent_id, duration, step)
        with JOBS_LOCK:
            JOBS[job_id].update(state="done", result=result, error=None)
    except Exception as exc:  # noqa: BLE001 - surfaced to the browser
        with JOBS_LOCK:
            JOBS[job_id].update(state="error", error=str(exc),
                                trace=traceback.format_exc()[-800:])


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def handler_factory(jobs_lock: threading.Lock):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args, **kwargs):
            """Silence the stdlib access log; the lab prints its own."""

        def send_json(self, status: int, value) -> None:
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def send_file(self, path: pathlib.Path, content_type: str) -> None:
            if not path.is_file():
                self.send_json(404, {"error": "not found"})
                return
            payload = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            route = parsed.path
            try:
                if route in ("/", "/index.html"):
                    self.send_file(STATIC / "index.html", "text/html; charset=utf-8")
                elif route == "/api/opponents":
                    self.send_json(200, {"opponents": scan_opponents(),
                                         "bundled": sorted(BUNDLED)})
                elif route.startswith("/api/job/"):
                    job_id = route.rsplit("/", 1)[-1]
                    with jobs_lock:
                        job = JOBS.get(job_id)
                        if job is None:
                            self.send_json(404, {"error": "unknown job"})
                            return
                        snapshot = {k: v for k, v in job.items() if k != "result"}
                        if job["state"] == "done":
                            snapshot["result"] = job["result"]
                    self.send_json(200, snapshot)
                elif route == "/api/health":
                    self.send_json(200, {"ok": True, "cli": str(CLI),
                                         "cliExists": CLI.exists(),
                                         "opponentsDir": str(OPPONENTS),
                                         "opponentsDirExists": OPPONENTS.is_dir()})
                elif route in ("/lab.js", "/lab.css"):
                    name = route.lstrip("/")
                    self.send_file(STATIC / name,
                                   "application/javascript" if name.endswith("js")
                                   else "text/css")
                elif route == "/README.md":
                    readme = pathlib.Path(__file__).resolve().parent / "README.md"
                    if not readme.is_file():
                        self.send_json(404, {"error": "no README"})
                        return
                    self.send_file(readme, "text/markdown; charset=utf-8")
                else:
                    self.send_json(404, {"error": "not found"})
            except BrokenPipeError:
                pass
            except Exception as exc:  # noqa: BLE001
                self.send_json(500, {"error": str(exc)})

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/match":
                self.send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 8192:
                    raise ValueError("request body too large")
                body = json.loads(self.rfile.read(length) or b"{}")
                opponent_id = str(body.get("opponent", "")).strip()
                duration = max(30, min(600, int(body.get("duration", 60))))
                step = max(1, min(6, int(body.get("step", 2))))
                if not opponent_id:
                    raise ValueError("no opponent selected")
                job_id = uuid.uuid4().hex[:12]
                with jobs_lock:
                    JOBS[job_id] = {
                        "id": job_id, "opponent": opponent_id,
                        "duration": duration, "step": step,
                        "state": "queued", "result": None, "error": None,
                        "created": time.time(),
                    }
                POOL.submit(job_worker, job_id, opponent_id, duration, step)
                self.send_json(202, {"job": job_id})
            except ValueError as exc:
                self.send_json(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                self.send_json(500, {"error": str(exc)})

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=5177)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    if not CLI.exists():
        print(f"warning: CLI not found at {CLI}", file=sys.stderr)
    count = len(scan_opponents())
    server = ThreadingHTTPServer((args.host, args.port), handler_factory(JOBS_LOCK))
    print(f"Match Lab on http://{args.host}:{args.port}/  ({count} opponents)")
    print(f"  engine: {CLI}")
    print("  Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
