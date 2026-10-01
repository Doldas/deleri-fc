# Vanguard FC (pro)

Generated opponent for the Football Babylon kit. **Do not edit by hand** -- run
`python scripts/build_opponent_teams.py` from `My Teams/deleri-fc/` after
changing `src/opponents/engine_brain.py` or the family's tuning table.

* team id: `counter-pro`
* family: `counter`
* behaviour: `src/opponents/engine_brain.py`, inlined into `strategy.py`

## Playing it

```sh
football-team build --team-dir opponents/counter-pro
football-team simulate --team-dir opponents/counter-pro --decisions 1800
```

The bundled `practice` command only offers its own compiled opponents, so use
`simulate --opponent-path` (or the arena) to see this team play.

## Parameters

| key | value |
| --- | --- |
| `compactness` | 0.75 |
| `directness` | 0.75 |
| `gk_aggression` | 0.75 |
| `gk_speed` | 0.9 |
| `line_height` | 0.25 |
| `max_pass_travel` | 42.0 |
| `noise` | 0.07 |
| `press_delay` | 0.1 |
| `press_intensity` | 0.85 |
| `risk` | 0.6 |
| `seed_salt` | 3976327007 |
| `shoot_range` | 20.0 |
| `tempo` | 0.95 |
| `transition_speed` | 0.95 |
