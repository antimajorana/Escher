# Changelog

## Unreleased

- Lattices on surfaces (`papertilling.surfaces`): cylinder, torus, Moebius band and Klein bottle (exact quotients of the plane by
  symmetries of the lattice); sphere and projective plane with the lattice's elementary disclinations at its own rotation centres
  (icosahedron, cube, octahedron, pillow; size (h, k)); closed surfaces of genus g by branched double covers of a torus. Each surface
  has `checks()` (Euler characteristic, Gauss-Bonnet, no loops or double bonds, plane coordination away from cones) and a flat-net and a
  3D picture (`S.show()`).
- Defect pictures: `build(..., picture="relaxed")` and the Notebook 2 entries are redrawn when bonds would cross, fold or overlap
  (`papertilling/defects/draw.py`); raw pictures are unchanged. Notebook 3 now draws relaxed pictures.
- `tools/checks/drawing_check.py`: crossings, overlaps and folded faces of a drawn crystal.
- Rust workspace (`rust/`): `papertilling-core` (17 wallpaper groups, D-symbols, parameter tuning, flat/polyhedral/genus-g surfaces, defect catalogues, surgery, planar drawing, $\tilde{G}^+$ equivalence, Volterra and completion core gallery), `papertilling-app` (native `eframe` + `wgpu` 2D/3D desktop explorer with live defect combining studio and `.json` config exchange with Notebook 3), `papertilling-py` (optional `papertilling-rs` PyO3 acceleration wheel with automatic `PAPERTILLING_NO_RUST=1` fallback), and Windows/macOS/Linux release installers.

## 0.2.0

- Lists of the most Bravais-like lattices of each of the 17 wallpaper groups under three rules: `points` (default), `sheet`, `bonds`.
- 18 named lattices (Square, Kagome, Shastry-Sutherland, ...) from a hand-made sheet: `named`, `named_lattices`, `gallery`.
- Defect catalogue of a lattice (dislocations, disclinations, walls), a builder that puts defects on it, and a comparison of configurations.
- Slider explorer for the free parameters of a lattice (needs `ipywidgets`).
- Three notebooks, a test suite, and check scripts under `tools/checks/`.
