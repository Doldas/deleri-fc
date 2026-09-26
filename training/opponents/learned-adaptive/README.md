# Tabula RL (adaptive)

Generated opponent for the Football Babylon kit. **Do not edit by hand** -- run
`python scripts/build_opponent_teams.py` from `My Teams/my-team-fc/` after
changing `src/opponents/engine_brain.py` or the family's tuning table.

* team id: `learned-adaptive`
* family: `learned`
* behaviour: `src/opponents/engine_brain.py`, inlined into `strategy.py`

## Playing it

```sh
football-team build --team-dir opponents/learned-adaptive
football-team simulate --team-dir opponents/learned-adaptive --decisions 1800
```

The bundled `practice` command only offers its own compiled opponents, so use
`simulate --opponent-path` (or the arena) to see this team play.

## Parameters

| key | value |
| --- | --- |
| `compactness` | 0.75 |
| `directness` | 0.6 |
| `gk_aggression` | 0.75 |
| `gk_speed` | 0.9 |
| `learns` | True |
| `line_height` | 0.55 |
| `noise` | 0.08 |
| `pass_power` | 0.4 |
| `press_intensity` | 0.85 |
| `risk` | 0.45 |
| `seed_salt` | 4229630901 |
| `shoot_range` | 20.0 |
| `tempo` | 0.9 |
