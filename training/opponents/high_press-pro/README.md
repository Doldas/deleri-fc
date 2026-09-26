# Red Shift FC (pro)

Generated opponent for the Football Babylon kit. **Do not edit by hand** -- run
`python scripts/build_opponent_teams.py` from `My Teams/my-team-fc/` after
changing `src/opponents/engine_brain.py` or the family's tuning table.

* team id: `high_press-pro`
* family: `high_press`
* behaviour: `src/opponents/engine_brain.py`, inlined into `strategy.py`

## Playing it

```sh
football-team build --team-dir opponents/high_press-pro
football-team simulate --team-dir opponents/high_press-pro --decisions 1800
```

The bundled `practice` command only offers its own compiled opponents, so use
`simulate --opponent-path` (or the arena) to see this team play.

## Parameters

| key | value |
| --- | --- |
| `compactness` | 0.75 |
| `counterpress` | 0.75 |
| `directness` | 0.5 |
| `gk_aggression` | 0.75 |
| `gk_speed` | 0.9 |
| `noise` | 0.07 |
| `press_delay` | 0.1 |
| `press_intensity` | 0.85 |
| `press_trigger` | 0.55 |
| `seed_salt` | 1413092080 |
| `shoot_range` | 20.0 |
| `tackling` | 0.85 |
| `tempo` | 0.95 |
