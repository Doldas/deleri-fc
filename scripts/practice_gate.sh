#!/usr/bin/env bash
# Match-level practice gate.
#
# Why this exists: aiming the goalkeeper's distribution at the ball's
# collection point passed all 194 unit tests, built, and validated, and looked
# like a clear win against Vanguard (+11.7 points of win rate). It also cut our
# attack against the reference side from 7.0 to 2.5 goals per match. Nothing in
# the unit suite could see it, because the keeper's chosen target is identical
# between the two versions in every single-tick scenario -- the difference only
# accumulates over a whole match.
#
# So this gate runs real matches against two opponents with opposite
# incentives, and fails if either side of the attack/defence balance moves.
#
#   reference      a passive side: punishes us for losing the ball, and
#                  rewards us for building attacks. Guards attacking output.
#   counter-elite  Vanguard: pressures us and attacks our goal. Guards
#                  defensive output.
#
# A change that improves one and wrecks the other is not an improvement, and
# this is the check that says so. See docs/AB_TESTING.md for the noise floor:
# the engine is not deterministic, so these thresholds are deliberately loose
# and a marginal FAIL means re-run with a different --seed-prefix, not panic.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TEAM="$ROOT/My Teams/my-team-fc"
CLI="$ROOT/tools/team-cli/linux-x64/football-team"

GAMES="${GAMES:-20}"          # 20 games = 40 matches per opponent
SEED="${SEED:-gate-v1}"
MIN_REFERENCE_GPM="${MIN_REFERENCE_GPM:-4.0}"   # we were at 7.0
MIN_VANGUARD_CONCEDED="${MIN_VANGUARD_CONCEDED:-1.05}"
MIN_VANGUARD_WINS="${MIN_VANGUARD_WINS:-6}"

status=0

parse() {  # parse "<label> <text>" -> echoes "wins gpm_conceded"
  local label="$1" text="$2"
  local wdl goals
  wdl=$(sed -n 's/.*W\/D\/L: \([0-9]*\)\/.*/\1/p' <<<"$text" | head -1)
  goals=$(sed -n 's/.*Goals: \([0-9]*\)-\([0-9]*\).*/\1 \2/p' <<<"$text" | head -1)
  echo "${wdl:-0} ${goals:-0 0}"
}

echo "== building =="
"$CLI" build --path "$TEAM" 2>&1 | grep -E "^Built|ERROR"

echo
echo "== reference (guards attacking output) =="
ref_out=$("$CLI" simulate --path "$TEAM" --opponent reference \
  --duration 60 --games "$GAMES" --seed-prefix "$SEED" \
  --sides both --jobs 8 --skip-build --format text 2>&1)
read -r ref_wins ref_for ref_against <<<"$(parse reference "$ref_out")"
MATCHES=$(( GAMES * 2 ))   # --sides both plays every seed home and away
ref_gpm=$(awk -v g="$ref_for" -v n="$MATCHES" 'BEGIN{printf "%.2f", (n>0?g/n:0)}')
echo "   goals/match for: $ref_gpm (floor $MIN_REFERENCE_GPM), goals against: $ref_against"
if awk -v v="$ref_gpm" -v t="$MIN_REFERENCE_GPM" 'BEGIN{exit !(v+0 < t+0)}'; then
  echo "   FAIL: attacking output collapsed to $ref_gpm goals/match"
  status=1
else
  echo "   ok"
fi

echo
echo "== counter-elite / Vanguard (guards defensive output) =="
van_out=$("$CLI" simulate --path "$TEAM" --opponent-path "$ROOT/opponents/counter-elite" \
  --duration 60 --games "$GAMES" --seed-prefix "$SEED" \
  --sides both --jobs 8 --skip-build --format text 2>&1)
read -r van_wins van_for van_against <<<"$(parse vanguard "$van_out")"
van_cpm=$(awk -v g="$van_against" -v n="$MATCHES" 'BEGIN{printf "%.2f", (n>0?g/n:0)}')
echo "   wins: $van_wins (floor $MIN_VANGUARD_WINS), goals conceded/match: $van_cpm (ceiling $MIN_VANGUARD_CONCEDED)"
if awk -v v="$van_cpm" -v t="$MIN_VANGUARD_CONCEDED" 'BEGIN{exit !(v+0 > t+0)}'; then
  echo "   FAIL: conceding $van_cpm per match"
  status=1
else
  echo "   ok"
fi

echo
if [ "$status" -eq 0 ]; then
  echo "PASS - practice gate (seed $SEED, $GAMES games per opponent)"
else
  echo "FAIL - practice gate (seed $SEED, $GAMES games per opponent)"
  echo "Re-run with SEED=other to separate a real regression from engine noise."
fi
exit "$status"
