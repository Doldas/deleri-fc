"""Tests for the Match Lab tool under tools/lab/.

These cover the two pieces with real logic in them -- opponent discovery and
replay trimming -- plus the identity contract the lab depends on to draw a
distinguishable team. The HTTP layer and the canvas are exercised by hand
because they need a live server and a browser.
"""

import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "lab"))

import training.lab.lab_server as lab  # noqa: E402


def snapshot(index, score_a=0, score_b=0, owner="team-a"):
    """Build one engine-shaped replay snapshot."""
    def team(team_id, side, score):
        return {
            "id": team_id,
            "name": "Team " + side.upper(),
            "score": score,
            "players": [
                {"id": f"{team_id}:{role}", "role": role,
                 "position": {"x": 10.0 + i, "y": 5.0 * i},
                 "canAct": True, "facing": {"x": 1.0, "y": 0.0}}
                for i, role in enumerate(["gk", "d1", "d2", "f1", "f2"])
            ],
        }
    return {
        "phase": "openPlay",
        "timeRemainingSeconds": 60.0 - index,
        "ball": {"position": {"x": 30.0 + index, "y": 20.0},
                 "possessingTeam": owner},
        "teams": [team("team-a", "a", score_a), team("team-b", "b", score_b)],
    }


def raw_replay(count=41):
    return {"snapshots": [snapshot(i, score_a=1 if i >= 20 else 0)
                          for i in range(count)]}


class ScanOpponents(unittest.TestCase):
    def test_finds_bundled_and_generated(self):
        found = lab.scan_opponents()
        self.assertGreaterEqual(len(found), 50)
        ids = {o["id"] for o in found}
        self.assertIn("reference", ids)
        self.assertIn("high_press-elite", ids)

    def test_entries_have_what_the_dropdown_renders(self):
        for entry in lab.scan_opponents():
            with self.subTest(opponent=entry["id"]):
                self.assertTrue(entry["name"])
                self.assertIn("generated", entry)
                if entry["generated"]:
                    self.assertTrue(entry["shortName"])
                    self.assertRegex(entry["colors"]["primary"], r"^#[0-9A-Fa-f]{6}$")

    def test_ids_are_unique(self):
        ids = [o["id"] for o in lab.scan_opponents()]
        self.assertEqual(len(ids), len(set(ids)))


class TrimReplay(unittest.TestCase):
    def test_subsamples_frames(self):
        out = lab.trim_replay(raw_replay(41), step=2)
        self.assertEqual(out["meta"]["frames"], 21)
        self.assertEqual(len(out["frames"]["phase"]), 21)

    def test_column_arrays_are_aligned(self):
        frames = lab.trim_replay(raw_replay(41), step=2)["frames"]
        length = len(frames["phase"])
        for key in ("time", "scoreA", "scoreB", "owner"):
            with self.subTest(column=key):
                self.assertEqual(len(frames[key]), length)
        # Ball and each side's positions are (x, y[, canAct]) per player.
        self.assertEqual(len(frames["ball"]), length * 2)
        self.assertEqual(len(frames["a"]), length * 5 * 3)
        self.assertEqual(len(frames["b"]), length * 5 * 3)
        # Facing is one scalar per player per frame.
        self.assertEqual(len(frames["facingA"]), length * 5)
        self.assertEqual(len(frames["facingB"]), length * 5)

    def test_scores_and_goals_are_tracked(self):
        out = lab.trim_replay(raw_replay(41), step=2)
        self.assertEqual(out["meta"]["goals"][0]["side"], "a")
        self.assertEqual(out["meta"]["goals"][0]["score"], [1, 0])
        self.assertEqual(out["frames"]["scoreA"][-1], 1)

    def test_owner_encoding(self):
        frames = lab.trim_replay(raw_replay(3), step=1)["frames"]
        self.assertEqual(frames["owner"], [0, 0, 0])

    def test_empty_replay_does_not_explode(self):
        out = lab.trim_replay({}, step=2)
        self.assertEqual(out["meta"]["frames"], 0)
        self.assertEqual(out["frames"], {})

    def test_output_is_json_serialisable(self):
        json.dumps(lab.trim_replay(raw_replay(11), step=2))

    def test_shrinks_a_realistic_replay(self):
        big = raw_replay(1201)
        out = lab.trim_replay(big, step=2)
        self.assertEqual(out["meta"]["frames"], 601)
        self.assertLess(len(json.dumps(out)), len(json.dumps(big)))


class GeneratedIdentity(unittest.TestCase):
    """The lab draws two kits, so identity has to actually differ."""

    def _manifests(self):
        for team_json in sorted((lab.OPPONENTS).glob("*/football-team.json")):
            data = json.loads(team_json.read_text())
            if data.get("id") == "null-pointers-fc":
                continue
            yield team_json.parent, data

    def test_names_are_unique(self):
        names = [d["name"] for _, d in self._manifests()]
        self.assertGreaterEqual(len(names), 50)
        self.assertEqual(len(names), len(set(names)))

    def test_short_names_are_unique_and_short(self):
        shorts = []
        for team_dir, data in self._manifests():
            server_py = (team_dir / "server.py").read_text()
            with self.subTest(opponent=data["id"]):
                self.assertIn(f'"name": "{data["name"]}"', server_py)
                code = server_py.split('"shortName": "', 1)[1].split('"', 1)[0]
                shorts.append(code)
        self.assertEqual(len(shorts), len(set(shorts)))
        self.assertTrue(all(1 <= len(s) <= 4 for s in shorts))

    def test_primary_colours_are_varied(self):
        colours = set()
        for team_dir, _ in self._manifests():
            server_py = (team_dir / "server.py").read_text()
            primary = server_py.split('"primary": "', 1)[1][:7]
            colours.add(primary.upper())
        self.assertGreater(len(colours), 8)


if __name__ == "__main__":
    unittest.main()
