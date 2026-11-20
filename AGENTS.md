# Instructions for AI agents working on papertilling

This file is read by coding agents (Antigravity / Gemini, Claude Code and others) when they open this repository.

## Rules

- The author decides. Ask before changing a convention, a definition or the public API; purely technical choices are yours,
  state them when you report.
- Work on a branch; never push or open a pull request without the author's explicit yes.
- The repository is public: before any commit run `python tools/checks/private_scan.py` (no local paths, no personal data).
- When changing Python code or PyO3 bindings, keep the Python package installable without Rust and run `python -m pytest -q`
  (`python -m pytest -q -m slow` when touching slow-tested algorithms).
- Never edit `papertilling/legacy/` (vendored reference code the tests compare against).
- Notebooks are generated from `tools/nb_text.py`: when changing notebook sources or notebook-facing APIs, run
  `python tools/build_notebooks.py` (from `tools/`) and verify with `python tools/run_notebooks.py` (plus
  `python tools/build_colab.py <folder outside the repository>` when updating Colab exports).
- Test by known answers written in the test (hand counts, published formulas), never by values copied from the code under
  test. Report results as measured, including what failed or was not checked.
- Keep modules under 600 lines; prototype on the smallest case first and time it before larger sweeps.

## Layout

`papertilling/` (the package: `lattice.py`, `enumerate.py`, `points.py`, `dsymbol.py`, `groups.py`, `sheet.py`, `tuning.py`,
`widgets.py`, `defects/`, `surfaces/`, `data/`, `legacy/`), `tests/`, `notebooks/` (three notebooks), `tools/` (builders, runners,
`checks/`), `docs/` (`DEVELOPMENT.md`, `RUST_PLAN.md`).
