# Null Pointers FC

Edit `strategy.py`; the `decide` function is the only required customization point. The generated strategy is complete and playable.

If Python 3.11+ is installed locally, run `python -m unittest discover tests` for a fast unit test. Python is not required for the team build because `football-team build` runs inside Docker.

```bash
football-team build
football-team validate
football-team practice --opponent reference
football-team export --output null-pointers-fc.tar
```
