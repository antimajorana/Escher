# Plan: a Rust core and a desktop app for papertilling

Status: refined with the author (2026-11-15); Stages 0–6 implemented and verified on branch `stages-5-and-6`.
Execute stage by stage, reporting after each stage and committing on a branch only after the author's explicit yes.

## 1. Goal

An interactive **desktop** app to explore:
1. every lattice of every wallpaper group (the three list rules, the named lattices, Wyckoff positions, free parameters);
2. the defects of a chosen lattice (dislocations, disclinations, the four wall kinds) drawn on it;
3. defects combined live: place and move several defects, see the crystal, its total charge, its core, and whether it is
   equivalent to a single defect (Volterra and completion cores side by side);
4. the lattice on surfaces (torus, cylinder, Moebius band, Klein bottle, sphere, projective plane, genus g), rotatable in 3D.

It must be designed so that the same core later extends to **3D space groups and their lattices, hyperbolic groups and their
lattices, and more exotic groups**. Nothing 3D, hyperbolic or exotic is built in this plan.

## 2. Decisions (by the author, 2026-11-15)

| question | decision |
|---|---|
| where the app runs | desktop only (native Windows / Linux / macOS) |
| who computes, who draws | a Rust core, with the Python package on top (Python bindings), like moyo / spglib |
| version 1 contents | all four: browse lattices per group, browse defects per lattice, combine defects live, lattices on surfaces |
| licence | GPL-3.0-only, like the Python package |
| GUI and 3D stack | `eframe 0.37` / `egui` + `egui_extras` + `rfd` + custom `egui_wgpu::CallbackTrait` (`wgpu 30`) |
| symmetry back end | 17 plane groups self-contained in `papertilling-core::plane`; reserve `moyo` for the 3D phase (optional dev-test check only) |
| data exchange | JSON (`data/*.json` + `data/named.json` storing `(recipe, frame)`) + PyO3 in-memory bindings (`papertilling-rs`, `abi3-py310`) |
| finding lattices in v1 | stored JSON catalogues (664 drawings across `points`, `sheet`, `bonds`) + 21 named lattices in Stages 1–5; live `points`-rule search (`spade`) optional later |
| exports / imports in v1 | SVG + PNG pictures, and `.json` defect configurations (`recipe` + `frame`) to reopen in Notebook 3 (no Tegula `.ds` in v1) |
| `genus g` in the app | add a rotatable 3D embedding for `genus g` in the Rust app while matching Python's `V, E, F`, cones, and 2D double-sheet net invariants |

Earlier decisions about the mathematics are in section 9; they hold for the Rust core too.

## 3. What exists already (survey and verification of 2026-11-15)

No tool doing all four of the goals together was found (searches of the web and arXiv from five angles plus a completeness
check; "not found", not "does not exist"; some entries rest on search snippets). Closest relatives and what to reuse:

- **Tegula** (Huson, Delgado-Friedrichs; GPL-3.0, Java): interactive browser of 2D periodic tilings of the plane, sphere and
  hyperbolic plane by Delaney-Dress symbol (arXiv 2007.10625). Idea to borrow: ordering by Dress complexity and a searchable
  database. (`.ds` export/import verified compatible with `dsymbol.py`, deferred past v1.)
- **Gavrog / Systre** (Apache-2.0, Java): 3D nets and the Systre canonical key; the 3D counterpart of the D-symbol key.
- **moyo** (`0.22.0`, Apache-2.0 / MIT, Rust, with Python and WASM bindings): checked in crate source: **no standalone 17
  plane-group API or table**; provides the 230 space groups (`MoyoDataset`), 80 layer groups (`MoyoLayerDataset`, matching all
  17 polar layer groups of `groups.spglib_layer_check`), and 1651 magnetic space groups. Candidate base for the 3D phase and a
  model for "Rust core + Python bindings".
- **Bilbao Crystallographic Server**: reference Wyckoff positions of the plane groups (an independent check of the ITA tables).
- **GAP Cryst / CARAT / CrystCat**: group back end in any dimension (normalisers, conjugacy classes).
- **HyperCells / hypertiling / EPINET**: hyperbolic cells, regular tilings, hyperbolic tilings to 3D nets (EPINET data is
  CC BY-NC-ND: do not reuse without permission).
- Defect software (Atomsk, OVITO DXA, graphene defect builders) builds or detects defects in 3D atomistic crystals; none was
  found that derives every defect class of a lattice from its group, combines defects or compares cores.
- Theory already in print (cite it): holonomy / conjugacy-class classification of point defects (arXiv 1311.0496,
  arXiv 2106.06699), Mermin, Rev. Mod. Phys. 51, 591 (1979); Kleman and Friedel, "Disclinations, dislocations and continuous
  defects: a reappraisal" (arXiv 0704.3055); Bowick and Giomi, Adv. Phys. 58, 449 (2009) for defects on curved surfaces.
  The new part of papertilling is the code: enumeration, the defect catalogue of each lattice, combination and equivalence,
  and surfaces, in one framework.

## 4. Architecture

One Cargo workspace in `rust/`:
- `papertilling-core`: pure Rust, no GUI, no Python, no C/Fortran build dependencies (`serde`, `serde_json`, `faer`
  `sparse-linalg` for AMD + sparse Cholesky, `petgraph` for VF2 labelled core isomorphism and articulation points, `svg`;
  `spade` if live `points` search is added later).
- `papertilling-py`: PyO3 (`0.30`, `abi3-py310`) bindings, built with `maturin` (`1.16+`) as optional distribution
  `papertilling-rs` (`import papertilling_rs`). **Optional**: root `pyproject.toml` stays pure-Python `setuptools` so
  `pip install .`, git installs, CI, and `build_colab.py` keep working without a Rust toolchain; Python uses `papertilling_rs`
  when importable (unless `PAPERTILLING_NO_RUST=1`) and falls back to its own code otherwise.
- `papertilling-app`: native desktop app on `eframe 0.37` / `egui 0.37` (`wgpu 30` default, `App::ui` API), `egui_extras 0.37`
  (`TableBuilder`), `rfd 0.18` (`xdg-portal` pure-Rust native dialogs), `png 0.18`, and a custom `egui_wgpu::CallbackTrait`
  3D surface viewport (two-sided `|n·L|` shading, depth-sorted translucent faces for self-intersecting immersions, antipodal
  sphere double cover for the projective plane, and a 3D handlebody embedding for `genus g`).

Inside the core, keep the parts that will extend to other geometries free of 2D matrices: Delaney-Dress symbols and their
canonical keys, "pieces + pairings" surfaces, holonomy classes of defects, defect configurations as graph surgery with a
holonomy on each bond. Keep everything specific to the Euclidean plane in a `plane` module: the 17 groups and their 72-row
ITA tables, Wyckoff positions, parameterised families, Voronoi neighbours, wedge cuts, the drawing layer.

Do **not** port `papertilling/legacy/` or `papertilling/defects/fast.py`. Write the defect builder from the definitions:
cut-and-reglue as surgery on the bond graph (each bond carries its plane vector and the holonomy accumulated across cuts,
with multi-sheeted wedge insertion for negative Frank angles), and compute positions in a separate drawing layer (Gauss-Newton
relaxation via symbolic-once sparse Cholesky; Tutte embedding with a pinned rim as fallback/warm-start; checks for crossings,
folds, overlaps and crowded sites).

## 5. Stages (each ends with a report to the author and a commit on a branch only after their yes)

**Stage 0. Toolchain and facts.** Verified: `rustc` / `cargo 1.100.0` + MSVC BuildTools present; `pyo3 0.30.1` + `maturin 1.16.0`
support Python 3.10–3.15 (`abi3-py310`); `moyo 0.22.0` has 230 space + 80 layer groups (no standalone 17 plane groups);
`eframe 0.37.0` defaults to `wgpu 30.0` (`egui_wgpu::CallbackTrait`). Update `.gitignore`, `private_scan.py` and `export_repo.py`
for `rust/`, scaffold the 3-crate workspace, add a CI job building and testing the crates, and verify a native smoke window
(`--smoke-test`) rendering both 2D `egui::Painter` bonds and a 3D `egui_wgpu` mesh. Report what works.

**Stage 1. Core: groups, lattices, D-symbols.** The 17 wallpaper groups (exact rational operations, 72-row ITA Wyckoff tables,
split `G+` centre classes `(i)`/`(ii)`), reading the stored lists (`papertilling/data/*.json`: 664 entries across `points`,
`sheet`, `bonds`, plus `data/named.json` with `(recipe, frame)` for the 21 named lattices), the Delaney-Dress symbol, its
canonical key and the network key. **Gate:** Rust recomputes the stored key, network key and symmetry index of all 664 catalogue
entries and all 21 named lattices identically to Python; the Wyckoff tables equal `papertilling.groups`.

**Stage 2. App: the gallery.** Groups, then lattices (three rules, list length N), named lattices, Wyckoff table with the
occupied positions marked, the lattice drawn (plus SVG/PNG export), sliders for the free parameters that stop at crossings and
mark special values (port `papertilling/tuning.py` with `Family`'s orthogonal mirror-line base $p - \frac{p\cdot d}{d\cdot d} d$).
**Gate:** slider limits and special values equal Python's on `p4g #1` and `p1 #1` under both `rule="bonds"` (`tests/test_widgets.py`)
and `rule="points"`.

**Stage 3. Surfaces.** Flat quotients (torus, cylinder, Moebius band, Klein bottle), the pieces + pairings engine (sphere,
projective plane), genus g by branched double covers, the IMPOSSIBLE / NOT FOUND / OK verdicts (`surface_table`), a rotatable
3D view (including a 3D handlebody embedding for `genus g`) and the flat net. **Gate:** `V, E, F`, cone lists and verdicts
equal Python's on the first lattice of every group (the 17 x 8 table) and on the known answers in `tests/test_sphere.py`
(C60, cube, ...) and `tests/test_genus.py`.

**Stage 4. Defect catalogue.** Dislocation classes, disclination classes (centre classes, allowed angles), the four wall
kinds, built by graph surgery and drawn by the drawing layer. **Gate:** class lists equal `DefectCatalogue` (counts, labels,
Burgers vectors, angles, wall lists) on the 12 lattices of `drawing_check.py`; every picture passes the drawing checks under
the documented rules of `tests/test_drawing.py` (no crossings, overlaps, crowded sites or folds, except Moebius seam stubs,
locally repaired glide-breaking walls with `crowded <= 1` noted, and un-openable walls falling back to `"rigid"` with a note).

**Stage 5. Live combining.** Click to place a defect (snapped to an allowed centre), drag to move it, see the crystal update,
its total charge, its core graph, and the comparison with the single defect (Volterra and completion cores), plus `.json`
configuration export/import (`recipe` + `frame` + `defects` + `options` + `summary`) for Notebook 3. **Gate:** total charges
and verdicts equal `equivalent_single` on the Notebook 3 configurations and the square/honeycomb controls in `tests/test_cores.py`
and `tests/test_config.py`. Target: an update in under 100 ms for a 500-site crystal.

**Stage 6. Python on top.** Expose the core through `papertilling-py` (`papertilling-rs`); the Python package calls it when
present and falls back cleanly when `PAPERTILLING_NO_RUST=1` or not installed. **Gate:** the full Python test suite (`pytest -q`
and `pytest -q -m slow`) passes with and without the extension, at the oldest supported versions too.

## 6. How to test the Rust core against Python

Compare invariants, never drawn positions: D-symbol keys, V/E/F, cone lists, total charges, core graphs up to isomorphism,
class lists. Every stage starts with a known-answer control on the smallest case, and a negative control where possible.
The Python package (branch where stage 0 starts) is the reference; if the two disagree, find the cause, do not pick one.

## 7. Working rules for the executing agent

- Read `AGENTS.md` (repository root) first.
- Work on a new branch per stage; never commit, push or open a pull request without the author's explicit yes.
- Keep the Python package installable and its tests green throughout (`python -m pytest -q`; `-m slow` before a commit).
- Before any commit: `python tools/checks/private_scan.py` must report no hits (the repository is public).
- Short reports after each stage: what works, the gate numbers, what is not done or not verified.

## 8. Refinement questions (resolved with the author, 2026-11-15)

1. GUI stack: **`eframe 0.37` / `egui` + `egui_extras` + `rfd` + custom `egui_wgpu::CallbackTrait` (`wgpu 30`)** (see section 2).
2. Symmetry back end: **17 plane groups self-contained in `papertilling-core::plane`**; reserve `moyo` for the 3D phase.
3. Data exchange: **JSON** (`data/*.json` + `data/named.json`) and in-memory PyO3 bindings (`papertilling-rs`).
4. Finding new lattices: **stored JSON lists + 21 named lattices in Stages 1–5**; live `points`-rule search optional later.
5. Export/import formats in v1: **SVG + PNG pictures, and Notebook 3 `.json` configurations only** (no Tegula `.ds` in v1).
6. `genus g` surface in the app: **add a rotatable 3D embedding in Rust** while matching Python's `V, E, F`, cones, and 2D net.

## 9. Mathematical decisions already made (2026-11-13 to 2026-11-15; they hold in Rust too)

- A lattice of G: planar, connected, exactly G (with G's translations); every site at least 3 bonds; no bond through a site.
- Lists: one entry per network per group (network = D-symbol with the symmetry forgotten), by its simplest drawing.
  Default rule `points`: the points alone have exactly the symmetry G, fewest kinds of site first, then fewest sites per
  cell, bonds = Voronoi neighbours. Also `sheet` (same condition on the points, fewest kinds of bond: reproduces the
  author's hand-made sheet of named lattices) and `bonds` (bonds may lower the symmetry). Lattices whose network has more
  symmetry than G are kept.
- Defect classes: homotopy classification, i.e. conjugacy classes of the holonomy in the lifted proper group, Frank angle
  kept unreduced (+180 and -180 differ); disclinations between -180 and +180 deg.
- Mirror- and glide-breaking defects: all four kinds kept "to see which is correct": (a) domain walls from the Euclidean
  normaliser, (b) twin walls, (c) Moebius seams, (d) glide-breaking walls.
- Equivalence of configurations: total charge + core graph ("in the same way a +90 vertex disclination and a -90 face
  disclination equal a dislocation on the square lattice").
- Surfaces: flat ones exactly; the sphere with the lattice's elementary disclinations at its own rotation centres
  (12 x 60, 8 x 90, 6 x 120, 4 x 180), size (h, k); projective plane; genus g. A lattice that cannot have a surface is flagged
  IMPOSSIBLE (proved from its symmetry) or NOT FOUND (this construction failed at the sizes tried).
- Not decided yet: whether completion cores count as the same defect as Volterra cores; which wall kind is "correct";
  the default corner class for the square lattice on a sphere (face centres give 8 triangles, `centre="site"` the cube);
  whether to try a projective plane for p4g with corners alternating between the two 4-fold classes.

## 10. State of the Python package at handover

Branch `surfaces-and-drawing` (not yet committed at the time of writing: commit it first, with the author's yes, so stage 0
starts from a committed reference). It adds the surfaces (`papertilling/surfaces/`), the drawing layer for defect pictures
(`papertilling/defects/draw.py`, `planarity.py`), the core gallery (Volterra and completion cores), the surface verdicts, the
drawing and private-data checks (`tools/checks/`), the Colab builder (`tools/build_colab.py OUTPUT_FOLDER`; the Colab copies are
kept outside the repository), and the notebook text in `tools/nb_text.py` (rebuild with `tools/build_notebooks.py`, run with
`tools/run_notebooks.py`).
