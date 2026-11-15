# Instructions for AI agents working on papertilling

This file is read by coding agents (Antigravity / Gemini, Claude Code and others) when they open this repository.

## Current task

Read `docs/RUST_PLAN.md`: a plan to build a Rust core and a desktop app for this package. First **refine** the plan with the
author (answer its open questions, check the facts it marks as unverified, propose changes), and only then **execute** it
stage by stage, reporting after each stage. Section 9 of the plan lists the mathematical decisions already made; do not
change them without the author's explicit agreement.

## Rules

- The author decides. Ask before changing a convention, a definition or the public API; purely technical choices are yours,
  state them when you report.
- Never commit, push or open a pull request without the author's explicit yes. Work on a branch.
- The repository is public: before any commit run `python tools/checks/private_scan.py` (no local paths, no personal data).
- Keep the Python package installable without Rust and its tests green: `python -m pytest -q` (fast set) and
  `python -m pytest -q -m slow` before a commit.
- Never edit `papertilling/legacy/` (vendored reference code the tests compare against).
- Notebooks are generated: edit `tools/nb_text.py`, then `python tools/build_notebooks.py` (from `tools/`) and
  `python tools/run_notebooks.py` (execute and save). Run them under the author's Jupyter kernels before saying they work
  (`python tools/run_notebooks.py --kernel <name> --no-save`).
- Colab copies are not part of the repository: `python tools/build_colab.py <folder outside the repository>`, then check them
  with `python tools/run_notebooks.py --dir <that folder> --no-save` from a Python without papertilling installed.
- Test by known answers written in the test (hand counts, published formulas), never by values copied from the code under
  test. Report results as measured, including what failed or was not checked.
- Keep modules under 600 lines; prototype on the smallest case first and time it before larger sweeps.

## Layout

`papertilling/` (the package: `lattice.py`, `enumerate.py`, `points.py`, `dsymbol.py`, `groups.py`, `sheet.py`, `tuning.py`,
`widgets.py`, `defects/`, `surfaces/`, `data/`, `legacy/`), `tests/`, `notebooks/` (three notebooks), `tools/` (builders, runners,
`checks/`), `docs/` (`DEVELOPMENT.md`, `RUST_PLAN.md`).
