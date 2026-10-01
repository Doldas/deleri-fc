# Blue Wall FC (rookie)

Generated opponent for the Football Babylon kit. **Do not edit by hand** -- run
`python scripts/build_opponent_teams.py` from `My Teams/deleri-fc/` after
changing `src/opponents/engine_brain.py` or the family's tuning table.

* team id: `low_block-rookie`
* family: `low_block`
* behaviour: `src/opponents/engine_brain.py`, inlined into `strategy.py`

## Playing it

```sh
football-team build --team-dir opponents/low_block-rookie
football-team simulate --team-dir opponents/low_block-rookie --decisions 1800
```

The bundled `practice` command only offers its own compiled opponents, so use
`simulate --opponent-path` (or the arena) to see this team play.

## Parameters

| key | value |
| --- | --- |
| `compactness` | 0.45 |
| `directness` | 0.3 |
| `gk_aggression` | 0.3 |
| `gk_speed` | 0.55 |
| `line_height` | 0.28 |
| `noise` | 0.22 |
| `pass_power` | 0.5 |
| `press_delay` | 0.3 |
| `press_intensity` | 0.5 |
| `press_trigger` | 0.3 |
| `risk` | 0.25 |
| `seed_salt` | 1582764933 |
| `shoot_range` | 12.0 |
| `tempo` | 0.7 |
