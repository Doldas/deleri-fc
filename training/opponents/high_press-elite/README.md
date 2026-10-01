# Red Shift FC (elite)

Generated opponent for the Football Babylon kit. **Do not edit by hand** -- run
`python scripts/build_opponent_teams.py` from `My Teams/deleri-fc/` after
changing `src/opponents/engine_brain.py` or the family's tuning table.

* team id: `high_press-elite`
* family: `high_press`
* behaviour: `src/opponents/engine_brain.py`, inlined into `strategy.py`

## Playing it

```sh
football-team build --team-dir opponents/high_press-elite
football-team simulate --team-dir opponents/high_press-elite --decisions 1800
```

The bundled `practice` command only offers its own compiled opponents, so use
`simulate --opponent-path` (or the arena) to see this team play.

## Parameters

| key | value |
| --- | --- |
| `compactness` | 0.85 |
| `counterpress` | 0.75 |
| `directness` | 0.5 |
| `gk_aggression` | 0.9 |
| `gk_speed` | 1.0 |
| `noise` | 0.03 |
| `press_delay` | 0.0 |
| `press_intensity` | 0.95 |
| `press_trigger` | 0.55 |
| `seed_salt` | 4158786448 |
| `shoot_range` | 22.0 |
| `tackling` | 0.85 |
| `tempo` | 1.0 |
