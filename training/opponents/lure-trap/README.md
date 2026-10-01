# Serpent Lure FC (trap)

Generated opponent for the Football Babylon kit. **Do not edit by hand** -- run
`python scripts/build_opponent_teams.py` from `My Teams/deleri-fc/` after
changing `src/opponents/engine_brain.py` or the family's tuning table.

* team id: `lure-trap`
* family: `lure`
* behaviour: `src/opponents/engine_brain.py`, inlined into `strategy.py`

## Playing it

```sh
football-team build --team-dir opponents/lure-trap
football-team simulate --team-dir opponents/lure-trap --decisions 1800
```

The bundled `practice` command only offers its own compiled opponents, so use
`simulate --opponent-path` (or the arena) to see this team play.

## Parameters

| key | value |
| --- | --- |
| `compactness` | 0.9 |
| `counterpress` | 1.0 |
| `deception` | 0.15 |
| `directness` | 0.3 |
| `gk_aggression` | 0.9 |
| `gk_speed` | 1.0 |
| `line_height` | 0.2 |
| `noise` | 0.04 |
| `pass_power` | 0.5 |
| `press_intensity` | 0.35 |
| `press_trigger` | 0.3 |
| `risk` | 0.25 |
| `seed_salt` | 2132025381 |
| `shoot_range` | 22.0 |
| `tempo` | 1.0 |
