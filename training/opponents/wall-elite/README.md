# Touchline FC (elite)

Generated opponent for the Football Babylon kit. **Do not edit by hand** -- run
`python scripts/build_opponent_teams.py` from `My Teams/deleri-fc/` after
changing `src/opponents/engine_brain.py` or the family's tuning table.

* team id: `wall-elite`
* family: `wall`
* behaviour: `src/opponents/engine_brain.py`, inlined into `strategy.py`

## Playing it

```sh
football-team build --team-dir opponents/wall-elite
football-team simulate --team-dir opponents/wall-elite --decisions 1800
```

The bundled `practice` command only offers its own compiled opponents, so use
`simulate --opponent-path` (or the arena) to see this team play.

## Parameters

| key | value |
| --- | --- |
| `compactness` | 0.85 |
| `directness` | 0.55 |
| `gk_aggression` | 0.9 |
| `gk_speed` | 1.0 |
| `noise` | 0.03 |
| `pass_power` | 0.35 |
| `press_intensity` | 0.95 |
| `risk` | 0.4 |
| `seed_salt` | 3493887935 |
| `shoot_range` | 22.0 |
| `tempo` | 1.0 |
| `width` | 0.95 |
