# Match Lab

A local visual practice board for My Team FC.

The stock `football-team practice` command only knows three opponents
(`reference`, `reference-strikers`, `slapstick-united`) because that list is
baked into the shipped CLI. It has no flag for a custom opponent, and unknown
flags are ignored rather than rejected. This lab fills the gap: it runs the same
authoritative engine through `football-team simulate`, then replays the result
on a canvas so you can watch a match against any team in the repository.

## Run it

```bash
cd "My Teams/my-team-fc"
python3 tools/lab/lab_server.py --port 5177
```

Then open <http://127.0.0.1:5177/>.

The first match against a given opponent builds that opponent's Docker image.
That can take a minute; later matches reuse the cached layer.

## What you get

- A dropdown of every opponent: the 50 generated teams under `opponents/` plus
  the three bundled CPUs.
- A rendered pitch with both kits, the ball, and the active ball carrier.
- Timeline with goal markers, scrub, step buttons, and 0.25x-4x speed.
- Live summary from the engine: score, possession, shots, missed decisions.

## How it works

`lab_server.py` exposes a small JSON API and serves the static UI in `static/`.

| Route | Purpose |
| --- | --- |
| `GET /api/health` | CLI path and opponents directory check. |
| `GET /api/opponents` | Scans `opponents/` and merges the bundled CPUs. |
| `POST /api/match` | Queues a match, returns a job id. |
| `GET /api/job/<id>` | Job state, then the trimmed replay when done. |

A match job runs:

```bash
football-team simulate --path "My Teams/my-team-fc" --games 1 \
  --duration <seconds> --replay <tmp> [--opponent | --opponent-path] <team>
```

The raw replay is about 10 MB for a 60 s match because it stores 20 Hz absolute
positions. `trim_replay()` reduces it to roughly 0.2 MB by keeping every
`step`-th frame, splitting the player arrays into a named `order`, and hoisting
per-match constants (kits, score, order) into `meta`. The result is a few
hundred frames of plain JSON that the browser can scrub through.

Jobs are serialised behind a lock. `simulate` tags images by opponent, so
overlapping runs would collide on the same Docker tag.

There is deliberately no `--skip-build`. The image is rebuilt from disk so
strategy and identity edits show up immediately.

## Caveats

- This is a viewing tool, not a replacement for the official practice mode. It
  has no live interactive control; you watch a completed simulation.
- Matches are deterministic for a fixed seed, so re-running the same opponent
  and duration gives the same match. Change the duration to get a different one.
- Strength across the generated families is not fully calibrated yet. Many
  opponents share similar scorelines and differ more in style than in results.
  See `artifacts/opponents/FINDINGS.md`.
