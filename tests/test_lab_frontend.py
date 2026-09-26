"""Static checks on the Match Lab frontend.

There is no browser in this environment, so these assert the properties of
`static/lab.js` that broke the page in practice. The dropdown bug was a `const`
declared inside an `if` block and referenced outside it: every opponent after
the first in a group threw `ReferenceError`, the loop aborted, and because
`loadOpponents` was async with no error handling the failure was silent. The UI
showed one option and no error at all.

`test_dropdown_scoping_is_valid` executes the real function under a minimal DOM
stub in `node`, so the assertion is on behaviour rather than on text matching.
"""

import json
import pathlib
import shutil
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training" / "lab"))

STATIC = ROOT / "training" / "lab" / "static"
JS = STATIC / "lab.js"
HTML = STATIC / "index.html"
CSS = STATIC / "lab.css"

NODE = shutil.which("node")

# A DOM stub faithful enough to run loadOpponents(): options is a flat live list
# of every option descendant, exactly like HTMLSelectElement.options.
HARNESS = """
import fs from "fs";
const nodes = {};
function mkNode(tag) {
  const n = { tagName: tag.toUpperCase(), children: [],
    appendChild(c) { this.children.push(c); return c; },
    addEventListener() {}, value: "" };
  Object.defineProperty(n, "options", { get() {
    const out = [];
    const walk = (node) => { for (const c of node.children) {
      if (c.tagName === "OPTION") out.push(c); else walk(c); } };
    walk(n); return out; } });
  Object.defineProperty(n, "innerHTML", {
    get() { return ""; }, set(v) { if (!v) n.children = []; } });
  return n;
}
const select = mkNode("select");
nodes["opponent"] = select;
nodes["opponent-hint"] = mkNode("div");
globalThis.document = { createElement: mkNode,
  getElementById: (id) => nodes[id] };
const el = (id) => nodes[id];
const payload = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
globalThis.fetch = async () => ({ ok: true, status: 200, json: async () => payload });

const src = fs.readFileSync(process.argv[2], "utf8");
// Take the real function straight out of the file, no copy-paste drift.
const fnSrc = src.slice(src.indexOf("async function loadOpponents"),
                        src.indexOf("function updateHint"));
const state = { opponents: [] };
const updateHint = () => {};
const loadOpponents = eval("(" +
  fnSrc.replace(/async function loadOpponents/, "async function") + ")");
await loadOpponents();
const opts = select.options;
console.log(JSON.stringify({
  count: opts.length,
  groups: select.children.filter((c) => c.tagName === "OPTGROUP").length,
  values: opts.map((o) => o.value),
  labels: opts.map((o) => o.textContent),
}));
"""


@unittest.skipIf(NODE is None, "node not available")
class DropdownBehaviour(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile

        import training.lab.lab_server as lab_server

        payload = {"opponents": lab_server.scan_opponents()}
        cls.expected_ids = {o["id"] for o in payload["opponents"]}
        cls.tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(payload, cls.tmp)
        cls.tmp.close()
        cls.script = pathlib.Path(cls.tmp.name).with_suffix(".mjs")
        cls.script.write_text(HARNESS)

    @classmethod
    def tearDownClass(cls):
        for path in (getattr(cls, "script", None), cls.tmp.name):
            if path:
                pathlib.Path(path).unlink(missing_ok=True)

    def _run(self):
        proc = subprocess.run(
            [str(NODE), str(self.script), str(JS), self.tmp.name],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0,
                         f"harness failed: {proc.stderr[-2000:]}")
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_every_opponent_becomes_an_option(self):
        result = self._run()
        self.assertEqual(result["count"], len(self.expected_ids))

    def test_option_values_match_the_api(self):
        result = self._run()
        self.assertEqual(set(result["values"]), self.expected_ids)

    def test_grouped_but_never_empty(self):
        result = self._run()
        self.assertGreaterEqual(result["groups"], 2)
        self.assertFalse([l for l in result["labels"] if "no opponents" in l],
                         "empty-list placeholder was added despite having data")

    def test_labels_show_the_name(self):
        result = self._run()
        self.assertIn("Reference Athletic", result["labels"])


# Runs the real kit helpers from lab.js and reports their decisions as JSON.
KIT_HARNESS = """
import fs from "fs";
const src = fs.readFileSync(process.argv[2], "utf8");
// Take the kit block verbatim out of lab.js.
const mod = src.slice(src.indexOf("const KIT_ALTERNATES"),
                      src.indexOf("function kits()"))
  + "\\nexport { resolveKits, colorDistance, KIT_CONTRAST_MIN };";
fs.writeFileSync(process.argv[4], mod);
const { resolveKits, colorDistance, KIT_CONTRAST_MIN } =
  await import(process.argv[4]);

const opponents = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const HOME = { name: "My Team FC - R4", primary: "#2563EB",
               secondary: "#EFF6FF" };

const out = { min: KIT_CONTRAST_MIN, rows: [], edge: [] };
for (const o of opponents) {
  // The engine's replay meta reports the kit flat (id/name/primary/secondary).
  // /api/opponents reports it nested under `colors`. Feed the flat shape,
  // because that is what actually reaches the renderer at match time.
  const away = { id: o.id, name: o.name,
                 primary: o.colors ? o.colors.primary : null,
                 secondary: o.colors ? o.colors.secondary : null };
  const k = resolveKits({ home: HOME, away });
  out.rows.push({
    id: o.id, generated: o.generated,
    ownPrimary: away.primary, ownSecondary: away.secondary,
    a: k.a, b: k.b, distance: colorDistance(k.a, k.b),
  });
}

// Synthetic pairings that must not be satisfiable by the real pool.
const EDGE = [
  ["identical primary",   { primary: "#2563EB", secondary: "#0f766e" }],
  ["both shades of blue", { primary: "#1E3A8A", secondary: "#1D4ED8" }],
  ["secondary also blue", { primary: "#2563EB", secondary: "#1D4ED8" }],
  ["no identity",         { primary: null, secondary: null }],
  ["undefined identity",  {}],
  ["already distinct",    { primary: "#DC2626", secondary: "#FFFFFF" }],
  ["red home blue away",  { primary: "#2563EB", secondary: "#111111" }],
];
for (const [label, away] of EDGE) {
  const k = resolveKits({ home: HOME, away: { name: label, ...away } });
  out.edge.push({ label, b: k.b, distance: colorDistance(k.a, k.b) });
}
console.log(JSON.stringify(out));
"""


@unittest.skipIf(NODE is None, "node not available")
class KitColours(unittest.TestCase):
    """The two sides must never render the same colour.

    The original code did `base.primary || "#2563eb"`, so the bundled CPUs --
    which report `colors: null` because there is nothing to read out of a
    compiled team -- were drawn in My Team FC's own blue and both teams came
    out identical.
    """

    @classmethod
    def setUpClass(cls):
        import tempfile

        import training.lab.lab_server as lab_server

        opponents = lab_server.scan_opponents()
        cls.payload = tempfile.NamedTemporaryFile("w", suffix=".json",
                                                 delete=False)
        json.dump(opponents, cls.payload)
        cls.payload.close()
        # The harness and the module it extracts must be separate files: the
        # harness writes the module to disk and then imports it, so pointing
        # both at one path makes node overwrite the script it is running.
        cls.harness = pathlib.Path(cls.payload.name).with_suffix(".mjs")
        cls.harness.write_text(KIT_HARNESS)
        cls.mod = pathlib.Path(cls.payload.name).with_suffix(".mod.mjs")
        proc = subprocess.run(
            [str(NODE), str(cls.harness), str(JS), cls.payload.name,
             str(cls.mod)],
            capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            raise AssertionError(f"kit harness failed: {proc.stderr[-2000:]}")
        cls.data = json.loads(proc.stdout.strip().splitlines()[-1])
        cls.by_id = {r["id"]: r for r in cls.data["rows"]}

    @classmethod
    def tearDownClass(cls):
        for path in (getattr(cls, "harness", None), getattr(cls, "mod", None),
                     cls.payload.name):
            if path:
                pathlib.Path(path).unlink(missing_ok=True)

    def test_no_opponent_shares_the_home_colour(self):
        for row in self.data["rows"]:
            with self.subTest(opponent=row["id"]):
                self.assertGreaterEqual(
                    row["distance"], self.data["min"],
                    f"{row['id']}: {row['a']} vs {row['b']} "
                    f"(d={row['distance']:.0f})")

    def test_rendered_colours_are_never_equal(self):
        for row in self.data["rows"]:
            with self.subTest(opponent=row["id"]):
                self.assertNotEqual(row["a"].lower(), row["b"].lower())

    def test_bundled_cpus_without_identity_still_differ(self):
        bundled = [r for r in self.data["rows"] if not r["generated"]]
        self.assertTrue(bundled)
        for row in bundled:
            with self.subTest(opponent=row["id"]):
                self.assertIsNone(row["ownPrimary"])
                self.assertNotEqual(row["a"].lower(), row["b"].lower())

    def test_distinct_opponent_keeps_its_real_primary(self):
        row = self.by_id["counter-elite"]
        self.assertEqual(row["b"].lower(), row["ownPrimary"].lower())

    def test_clashing_opponent_switches_to_its_secondary(self):
        # null-pointers-fc ships a primary byte-identical to ours.
        row = self.by_id["null-pointers-fc"]
        self.assertEqual(row["ownPrimary"].lower(), row["a"].lower())
        self.assertEqual(row["b"].lower(), row["ownSecondary"].lower())

    def test_edge_cases_never_collide(self):
        for row in self.data["edge"]:
            with self.subTest(case=row["label"]):
                self.assertGreaterEqual(row["distance"], self.data["min"])

    def test_contrast_threshold_is_above_the_worst_real_case(self):
        worst = min(r["distance"] for r in self.data["rows"]
                    if r["b"].lower() == (r["ownPrimary"] or "").lower())
        self.assertGreaterEqual(worst, self.data["min"])


class StaticSource(unittest.TestCase):
    """Cheap guards that hold even without node."""

    def test_js_is_syntactically_plausible(self):
        text = JS.read_text()
        self.assertEqual(text.count("{"), text.count("}"),
                         "unbalanced braces in lab.js")
        self.assertNotIn("og is not defined", text)

    def test_load_opponents_reports_failure(self):
        text = JS.read_text()
        self.assertRegex(
            text, r"async function loadOpponents\(\)\s*\{\s*let data;\s*try",
            "loadOpponents must catch fetch errors so the UI never fails silently")

    def test_goalmark_styles_exist(self):
        self.assertIn(".goalmarks", CSS.read_text(),
                      "timeline goal markers are created in JS but were unstyled")

    def test_readme_link_is_served_by_the_server(self):
        html = HTML.read_text()
        self.assertNotIn('href="tools/lab/README.md"', html,
                         "repo-relative link cannot be resolved by the server")
        self.assertIn('href="/README.md"', html)

    def test_frontend_has_no_direct_stdlib_name_clash(self):
        self.assertTrue((ROOT / "training" / "lab" / "lab_server.py").exists())
        self.assertFalse((ROOT / "training" / "lab" / "server.py").exists(),
                         "server.py shadows the stdlib `server` module")


if __name__ == "__main__":
    unittest.main()
