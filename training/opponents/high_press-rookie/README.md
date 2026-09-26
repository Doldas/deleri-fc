# Red Shift FC (rookie)

Generated opponent for the Football Babylon kit. **Do not edit by hand** -- run
`python scripts/build_opponent_teams.py` from `My Teams/my-team-fc/` after
changing `src/opponents/engine_brain.py` or the family's tuning table.

* team id: `high_press-rookie`
* family: `high_press`
* behaviour: `src/opponents/engine_brain.py`, inlined into `strategy.py`

## Playing it

```sh
football-team build --team-dir opponents/high_press-rookie
football-team simulate --team-dir opponents/high_press-rookie --decisions 1800
```

The bundled `practice` command only offers its own compiled opponents, so use
`simulate --opponent-path` (or the arena) to see this team play.

## Parameters

| key | value |
| --- | --- |
| `compactness` | 0.45 |
| `counterpress` | 0.75 |
| `directness` | 0.5 |
| `gk_aggression` | 0.3 |
| `gk_speed` | 0.55 |
| `noise` | 0.22 |
| `press_delay` | 0.3 |
| `press_intensity` | 0.5 |
| `press_trigger` | 0.55 |
| `seed_salt` | 3268315231 |
| `shoot_range` | 12.0 |
| `tackling` | 0.85 |
| `tempo` | 0.7 |
