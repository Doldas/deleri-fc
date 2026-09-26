# Tabula RL (novice)

Generated opponent for the Football Babylon kit. **Do not edit by hand** -- run
`python scripts/build_opponent_teams.py` from `My Teams/my-team-fc/` after
changing `src/opponents/engine_brain.py` or the family's tuning table.

* team id: `learned-novice`
* family: `learned`
* behaviour: `src/opponents/engine_brain.py`, inlined into `strategy.py`

## Playing it

```sh
football-team build --team-dir opponents/learned-novice
football-team simulate --team-dir opponents/learned-novice --decisions 1800
```

The bundled `practice` command only offers its own compiled opponents, so use
`simulate --opponent-path` (or the arena) to see this team play.

## Parameters

| key | value |
| --- | --- |
| `compactness` | 0.4 |
| `directness` | 0.45 |
| `gk_aggression` | 0.75 |
| `gk_speed` | 0.9 |
| `learns` | True |
| `line_height` | 0.55 |
| `noise` | 0.28 |
| `pass_power` | 0.4 |
| `press_intensity` | 0.4 |
| `risk` | 0.45 |
| `seed_salt` | 360341166 |
| `shoot_range` | 9.0 |
| `tempo` | 0.6 |
