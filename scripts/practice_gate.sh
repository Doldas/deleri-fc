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
# So this gate runs real matches against several opponents with different
# incentives, and fails if any of them moves.
#
#   reference             passive: punishes us for losing the ball, rewards us
#                         for building attacks. Guards attacking output.
#   reference-strikers    the counter side. This is the one the keeper bug
#                         broke outright -- 0 wins in 20, 0-7 goals per match --
#                         while the gate below still passed. A gate that cannot
#                         see a 0% matchup is not a gate.
#   slapstick-united      same family, different tempo. Guards the easy counter
#                         side we were already winning.
#   counter-elite         Vanguard: pressures us and attacks our goal. Guards
#                         defensive output. This is our worst matchup by a wide
#                         margin and the only side we lose most of the time to,
#                         so its floors are set from measurement, not aspiration.
#
# Every floor is a *regression* guard set slightly above the measured current
# level, so the gate fails on a collapse rather than on engine noise. A change
# that improves one opponent and wrecks another is not an improvement, and this
# is the check that says so. See docs/AB_TESTING.md for the noise floor: the
# engine is not deterministic, so a marginal FAIL means re-run with a different
# SEED, not panic.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# The team directory was renamed my-team-fc -> deleri-fc, and the opponent
# pool was moved under training/opponents. Both paths below used to be stale,
# which made this gate resolve to nothing and silently exercise no matches.
TEAM="$ROOT/My Teams/deleri-fc"
OPPONENTS="$TEAM/training/opponents"
CLI="$ROOT/tools/team-cli/linux-x64/football-team"

GAMES="${GAMES:-20}"           # 20 games = 40 matches per opponent
SEED="${SEED:-gate-v1}"
MATCHES=$(( GAMES * 2 ))       # --sides both plays every seed home and away

# Measured at GAMES=20, SEED=gate-v1 (40 matches each):
#   reference           40/0/0  161-0   4.03 goals/match
#   reference-strikers  40/0/0   60-0
#   slapstick-united    40/0/0  100-0
#   counter-elite        9/0/31   9-51  1.28 conceded/match
MIN_WIN_REFERENCE="${MIN_WIN_REFERENCE:-90}"          # percent
MIN_GPM_REFERENCE="${MIN_GPM_REFERENCE:-3.0}"
MIN_WIN_STRIKERS="${MIN_WIN_STRIKERS:-90}"            # percent
MIN_WIN_SLAPSTICK="${MIN_WIN_SLAPSTICK:-90}"          # percent
MIN_WIN_VANGUARD="${MIN_WIN_VANGUARD:-20}"            # percent
MAX_VANGUARD_CONCEDED="${MAX_VANGUARD_CONCEDED:-1.45}"

status=0

parse() {  # parse "<text>" -> echoes "wins for against"
  local text="$1" wdl goals
  wdl=$(sed -n 's/.*W\/D\/L: \([0-9]*\)\/.*/\1/p' <<<"$text" | head -1)
  goals=$(sed -n 's/.*Goals: \([0-9]*\)-\([0-9]*\).*/\1 \2/p' <<<"$text" | head -1)
  echo "${wdl:-0} ${goals:-0 0}"
}

pct() { awk -v n="$1" -v d="$2" 'BEGIN{printf "%.1f", (d>0?100*n/d:0)}'; }

lt() { awk -v v="$1" -v t="$2" 'BEGIN{exit !(v+0 < t+0)}'; }
gt() { awk -v v="$1" -v t="$2" 'BEGIN{exit !(v+0 > t+0)}'; }

run_gate() {  # run_gate <label> <cli-flag> <path> <min win %> [min gpm] [max conceded]
  local label="$1" flag="$2" path="$3" minwin="$4" mingpm="${5:-}" maxconc="${6:-}"
  local out wins for_ against winpct gpm cpm line
  out=$("$CLI" simulate --path "$TEAM" "$flag" "$path" \
    --duration 60 --games "$GAMES" --seed-prefix "$SEED" \
    --sides both --jobs 8 --skip-build --format text 2>&1)
  read -r wins for_ against <<<"$(parse "$out")"
  winpct=$(pct "$wins" "$MATCHES")
  gpm=$(awk -v g="$for_" -v n="$MATCHES" 'BEGIN{printf "%.2f", (n>0?g/n:0)}')
  cpm=$(awk -v g="$against" -v n="$MATCHES" 'BEGIN{printf "%.2f", (n>0?g/n:0)}')

  line=$(printf "%-18s %6s%% win (floor %s%%)  %s-%s" \
    "$label" "$winpct" "$minwin" "$for_" "$against")
  echo "$line"

  if lt "$winpct" "$minwin"; then
    echo "   FAIL: win rate $winpct% is below the $minwin% floor"
    status=1
  fi
  if [ -n "$mingpm" ] && lt "$gpm" "$mingpm"; then
    echo "   FAIL: attacking output collapsed to $gpm goals/match (floor $mingpm)"
    status=1
  fi
  if [ -n "$maxconc" ] && gt "$cpm" "$maxconc"; then
    echo "   FAIL: conceding $cpm per match (ceiling $maxconc)"
    status=1
  fi
  if [ -z "$mingpm" ] && [ -z "$maxconc" ]; then
    echo "   goals/match for $gpm, conceded $cpm"
  fi
}

echo "== building =="
"$CLI" build --path "$TEAM" 2>&1 | grep -E "^Built|ERROR"

echo
echo "== match-level gates ($MATCHES matches each, seed $SEED) =="
run_gate reference            --opponent      reference                  "$MIN_WIN_REFERENCE"  "$MIN_GPM_REFERENCE"
run_gate reference-strikers   --opponent      reference-strikers         "$MIN_WIN_STRIKERS"
run_gate slapstick-united     --opponent      slapstick-united           "$MIN_WIN_SLAPSTICK"
run_gate counter-elite/Vanguard --opponent-path "$OPPONENTS/counter-elite" "$MIN_WIN_VANGUARD" "" "$MAX_VANGUARD_CONCEDED"

echo
if [ "$status" -eq 0 ]; then
  echo "PASS - practice gate (seed $SEED, $GAMES games per opponent)"
else
  echo "FAIL - practice gate (seed $SEED, $GAMES games per opponent)"
  echo "Re-run with SEED=other to separate a real regression from engine noise."
fi
exit "$status"
