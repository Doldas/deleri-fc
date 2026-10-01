# Red Shift FC (solid)

Generated opponent for the Football Babylon kit. **Do not edit by hand** -- run
`python scripts/build_opponent_teams.py` from `My Teams/deleri-fc/` after
changing `src/opponents/engine_brain.py` or the family's tuning table.

* team id: `high_press-solid`
* family: `high_press`
* behaviour: `src/opponents/engine_brain.py`, inlined into `strategy.py`

## Playing it

```sh
football-team build --team-dir opponents/high_press-solid
football-team simulate --team-dir opponents/high_press-solid --decisions 1800
```

The bundled `practice` command only offers its own compiled opponents, so use
`simulate --opponent-path` (or the arena) to see this team play.

## Parameters

| key | value |
| --- | --- |
| `compactness` | 0.62 |
| `counterpress` | 0.75 |
| `directness` | 0.5 |
| `gk_aggression` | 0.55 |
| `gk_speed` | 0.75 |
| `noise` | 0.12 |
| `press_delay` | 0.2 |
| `press_intensity` | 0.7 |
| `press_trigger` | 0.55 |
| `seed_salt` | 2681249122 |
| `shoot_range` | 17.0 |
| `tackling` | 0.85 |
| `tempo` | 0.85 |
