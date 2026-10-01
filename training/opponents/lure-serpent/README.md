# Serpent Lure FC (serpent)

Generated opponent for the Football Babylon kit. **Do not edit by hand** -- run
`python scripts/build_opponent_teams.py` from `My Teams/deleri-fc/` after
changing `src/opponents/engine_brain.py` or the family's tuning table.

* team id: `lure-serpent`
* family: `lure`
* behaviour: `src/opponents/engine_brain.py`, inlined into `strategy.py`

## Playing it

```sh
football-team build --team-dir opponents/lure-serpent
football-team simulate --team-dir opponents/lure-serpent --decisions 1800
```

The bundled `practice` command only offers its own compiled opponents, so use
`simulate --opponent-path` (or the arena) to see this team play.

## Parameters

| key | value |
| --- | --- |
| `compactness` | 0.85 |
| `counterpress` | 0.9 |
| `deception` | 1.0 |
| `directness` | 0.3 |
| `gk_aggression` | 0.9 |
| `gk_speed` | 1.0 |
| `line_height` | 0.28 |
| `noise` | 0.02 |
| `pass_power` | 0.5 |
| `press_intensity` | 0.95 |
| `press_trigger` | 0.3 |
| `risk` | 0.25 |
| `seed_salt` | 2284878476 |
| `shoot_range` | 22.0 |
| `tempo` | 1.0 |
| `width` | 0.7 |
