# Red Hand FC (elite)

Generated opponent for the Football Babylon kit. **Do not edit by hand** -- run
`python scripts/build_opponent_teams.py` from `My Teams/my-team-fc/` after
changing `src/opponents/engine_brain.py` or the family's tuning table.

* team id: `physical-elite`
* family: `physical`
* behaviour: `src/opponents/engine_brain.py`, inlined into `strategy.py`

## Playing it

```sh
football-team build --team-dir opponents/physical-elite
football-team simulate --team-dir opponents/physical-elite --decisions 1800
```

The bundled `practice` command only offers its own compiled opponents, so use
`simulate --opponent-path` (or the arena) to see this team play.

## Parameters

| key | value |
| --- | --- |
| `compactness` | 0.85 |
| `counterpress` | 0.85 |
| `gk_aggression` | 0.9 |
| `gk_speed` | 1.0 |
| `max_pass_travel` | 36.0 |
| `noise` | 0.03 |
| `press_intensity` | 0.95 |
| `seed_salt` | 2990768877 |
| `shoot_range` | 22.0 |
| `tackling` | 1.0 |
| `tempo` | 1.0 |
