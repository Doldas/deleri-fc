# Tabula RL (opportunist)

Generated opponent for the Football Babylon kit. **Do not edit by hand** -- run
`python scripts/build_opponent_teams.py` from `My Teams/my-team-fc/` after
changing `src/opponents/engine_brain.py` or the family's tuning table.

* team id: `learned-opportunist`
* family: `learned`
* behaviour: `src/opponents/engine_brain.py`, inlined into `strategy.py`

## Playing it

```sh
football-team build --team-dir opponents/learned-opportunist
football-team simulate --team-dir opponents/learned-opportunist --decisions 1800
```

The bundled `practice` command only offers its own compiled opponents, so use
`simulate --opponent-path` (or the arena) to see this team play.

## Parameters

| key | value |
| --- | --- |
| `compactness` | 0.75 |
| `directness` | 0.85 |
| `gk_aggression` | 0.75 |
| `gk_speed` | 0.9 |
| `learns` | True |
| `line_height` | 0.55 |
| `noise` | 0.1 |
| `pass_power` | 0.4 |
| `press_intensity` | 0.85 |
| `risk` | 0.9 |
| `seed_salt` | 3983563447 |
| `shoot_range` | 20.0 |
| `tempo` | 1.0 |
