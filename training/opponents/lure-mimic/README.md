# Serpent Lure FC (mimic)

Generated opponent for the Football Babylon kit. **Do not edit by hand** -- run
`python scripts/build_opponent_teams.py` from `My Teams/my-team-fc/` after
changing `src/opponents/engine_brain.py` or the family's tuning table.

* team id: `lure-mimic`
* family: `lure`
* behaviour: `src/opponents/engine_brain.py`, inlined into `strategy.py`

## Playing it

```sh
football-team build --team-dir opponents/lure-mimic
football-team simulate --team-dir opponents/lure-mimic --decisions 1800
```

The bundled `practice` command only offers its own compiled opponents, so use
`simulate --opponent-path` (or the arena) to see this team play.

## Parameters

| key | value |
| --- | --- |
| `compactness` | 0.85 |
| `counterpress` | 0.5 |
| `deception` | 0.9 |
| `directness` | 0.3 |
| `gk_aggression` | 0.9 |
| `gk_speed` | 1.0 |
| `line_height` | 0.28 |
| `noise` | 0.22 |
| `pass_power` | 0.5 |
| `press_intensity` | 0.95 |
| `press_trigger` | 0.3 |
| `risk` | 0.25 |
| `seed_salt` | 2046350752 |
| `shoot_range` | 22.0 |
| `tempo` | 0.85 |
| `width` | 0.8 |
