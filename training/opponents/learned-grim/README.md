# Tabula RL (grim)

Generated opponent for the Football Babylon kit. **Do not edit by hand** -- run
`python scripts/build_opponent_teams.py` from `My Teams/deleri-fc/` after
changing `src/opponents/engine_brain.py` or the family's tuning table.

* team id: `learned-grim`
* family: `learned`
* behaviour: `src/opponents/engine_brain.py`, inlined into `strategy.py`

## Playing it

```sh
football-team build --team-dir opponents/learned-grim
football-team simulate --team-dir opponents/learned-grim --decisions 1800
```

The bundled `practice` command only offers its own compiled opponents, so use
`simulate --opponent-path` (or the arena) to see this team play.

## Parameters

| key | value |
| --- | --- |
| `compactness` | 0.75 |
| `counterpress` | 0.9 |
| `directness` | 0.45 |
| `gk_aggression` | 0.75 |
| `gk_speed` | 0.9 |
| `learns` | True |
| `line_height` | 0.55 |
| `noise` | 0.05 |
| `pass_power` | 0.4 |
| `press_intensity` | 1.0 |
| `risk` | 0.45 |
| `seed_salt` | 1325769270 |
| `shoot_range` | 20.0 |
| `tackling` | 1.0 |
| `tempo` | 0.95 |
