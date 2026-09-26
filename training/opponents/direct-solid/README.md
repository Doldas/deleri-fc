# Long Ball FC (solid)

Generated opponent for the Football Babylon kit. **Do not edit by hand** -- run
`python scripts/build_opponent_teams.py` from `My Teams/my-team-fc/` after
changing `src/opponents/engine_brain.py` or the family's tuning table.

* team id: `direct-solid`
* family: `direct`
* behaviour: `src/opponents/engine_brain.py`, inlined into `strategy.py`

## Playing it

```sh
football-team build --team-dir opponents/direct-solid
football-team simulate --team-dir opponents/direct-solid --decisions 1800
```

The bundled `practice` command only offers its own compiled opponents, so use
`simulate --opponent-path` (or the arena) to see this team play.

## Parameters

| key | value |
| --- | --- |
| `compactness` | 0.62 |
| `directness` | 0.95 |
| `gk_aggression` | 0.55 |
| `gk_speed` | 0.75 |
| `max_pass_travel` | 45.0 |
| `noise` | 0.12 |
| `pass_power` | 0.55 |
| `press_delay` | 0.2 |
| `press_intensity` | 0.7 |
| `risk` | 0.65 |
| `seed_salt` | 3690085144 |
| `shoot_range` | 17.0 |
| `tempo` | 0.85 |
| `width` | 0.85 |
