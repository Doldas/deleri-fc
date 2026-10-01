# Deleri FC

This team was created with the Football Babylon tactics builder. Open it again with
`football-team create` to change the formation, roles, and tactical sliders in Basic mode.

The editable settings live in `team.json` and `tactics.json`. Developers can extend the
generated plan safely in `custom_strategy.py`; the builder never overwrites that file.

```bash
football-team build
football-team validate
football-team practice --opponent reference
football-team export --output deleri-fc.tar.zst
```
