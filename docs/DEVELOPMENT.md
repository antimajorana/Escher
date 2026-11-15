# Development notes

## Layout

| path | content |
|---|---|
| `papertilling/` | the package |
| `papertilling/data/` | stored lattice lists: `points.json` (rule `points`), `sheet.json` (rule `sheet`, the named lattices), `minimal_b.json` (rule `bonds`), `named.json` (the 21 named lattices as `(recipe, frame)`) |
| `papertilling/legacy/` | earlier code, used as engines for the defect builder and as regression references; the md5 sums of the originals are in `legacy/SOURCES.txt` |
| `rust/` | Cargo workspace: `papertilling-core` (plane groups, D-symbols, surfaces, defects), `papertilling-py` (optional PyO3 extension `papertilling_rs`), `papertilling-app` (native desktop app) |
| `tests/` | pytest suite |
| `notebooks/` | the three notebooks (saved with outputs) |
| `tools/` | builders of the stored data and notebooks, check scripts (`tools/checks/`), the README pictures, the Colab builder, the export script |

Modules: `_rust` (optional `papertilling_rs` bridge), `dsymbol` (Delaney-Dress symbol, network key), `groups` (ITA Wyckoff tables),
`points` (points-only rule, Voronoi bonds), `tuning` (slider limits and special values), `enumerate` + `kernels*` (bond-counting search),
`lattice` (`Lattice`, catalogue), `sheet` (the named lattices), `widgets` (sliders),
`surfaces/` (`base`, `flat`, `pieces`, `polyhedra`, `genus`, `refusal`, `report` for surface verdicts),
`defects/` (`catalogue`, `walls`, `build`, `config`, `cores`, `draw`, `planarity`, `fast*`).

## Tests

    python -m pytest -q              # fast tests (tests marked slow are skipped)
    python -m pytest -q -m slow      # heavy cases: full defect gate with the legacy engine, larger legacy-vs-fast scans

## Check scripts to run by hand

    python tools/checks/archimedean_control.py   # the search finds all 11 Archimedean tilings (about 1 minute)
    python tools/checks/catalogue_check.py       # every stored lattice is re-checked from its recipe
    python tools/checks/symmetry_check.py        # every stored drawing: group by three routes at several parameter values;
                                                 # no network twice in a list; the network test against a second route
    python tools/checks/points_only_symmetry.py  # symmetry of the point sets of the `points` rule

## Rebuilding stored data and notebooks

    python tools/build_points.py          # papertilling/data/points.json   (rule "points")
    python tools/build_catalogue.py       # papertilling/data/minimal_b.json (rule "bonds")
    python tools/build_notebooks.py       # writes the three notebooks from tools/nb_text.py
    python tools/run_notebooks.py         # executes them top to bottom and saves them with outputs
                                          # (--no-save: only run; --kernel NAME: another Jupyter kernel; N: only notebook N)

## Speed

The lattice search uses numba kernels (`kernels*.py`, each with a pure-Python reference path). The defect builder uses `defects/fast*.py`
(the legacy algorithms with faster primitives). The environment variable `PAPERTILLING_LEGACY=1` switches all defect code back to
the legacy engines (`fast=False` switches only the call it is passed to). The tests check that both paths give identical crystals,
charges and cores; the relaxed picture differs slightly (positions up to 0.1, same graph, energy within 4e-7).

How many lattices: `n` in `minimal_lattices(group, n=n)`. The stored lists hold the first 7 of each group. More than 7 are searched
at once, simplest first, until `n` are found. Measured on one PC (2026-11-14): 12 lattices take 29 s (p6m), 38 s (p1), 39 s (p2),
73 s (p4g); 20 lattices of p6m take 53 s. The search stops after 10 minutes (`time_limit`) and marks the unfinished tier OPEN.

## Rust core, desktop app, and optional Python extension

Root `pyproject.toml` stays pure-Python `setuptools`, so `pip install .` and `python tools/build_colab.py OUTPUT_FOLDER`
require no Rust toolchain. When `papertilling-rs` (`import papertilling_rs`) is installed, Python uses it automatically and
falls back to its own code when `PAPERTILLING_NO_RUST=1` or when the wheel is absent.

    cargo run --manifest-path rust/Cargo.toml -p papertilling-app --release                 # native desktop app
    cargo run --manifest-path rust/Cargo.toml -p papertilling-app --release -- --smoke-test # headless 2D + 3D smoke check
    cargo test --manifest-path rust/Cargo.toml --release --workspace                        # Rust workspace tests
    python tools/build_windows_installer.py                                                 # build dist/papertilling-*-windows-x86_64-setup.exe
    maturin develop --release --manifest-path rust/papertilling-py/Cargo.toml               # install papertilling-rs into active venv
    PAPERTILLING_NO_RUST=1 python -m pytest -q                                              # force pure-Python fallback

Defect configurations can be exported and reopened between the Combine tab of `papertilling-app` and Notebook 3 as portable
`.json` files (`recipe` + `frame` + `defects` + `options` + `summary`) via `save_config` and `load_config` in
`papertilling.defects.config`.

## Publishing the tree

The working copy is kept outside git. `python tools/export_repo.py [TARGET]` copies a clean tree to TARGET (never touching `TARGET/.git`),
rewrites `legacy/SOURCES.txt` without local paths and scans every exported text file for private strings.
