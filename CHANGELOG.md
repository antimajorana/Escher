# Changelog

## Unreleased

- Fix: `DefectCatalogue` now standardises a lattice to the ITA setting through its reduced (right-handed) primitive cell, so its Wyckoff labels agree with `wyckoff_rotocentres`, `fusion_channels` and `build` on every named lattice. Before, a lattice given in a left-handed cell was standardised through a reflection: the catalogue swapped 1b and 1c in `chiral_triangular` (p3), and named other classes differently from `wyckoff_rotocentres` in `oblique`, `straight`, `rectangular` and `sutherlands_brick`. Positions and geometry are unchanged; only names change.
- Fix: `fusion_channels` of two disclinations whose Frank angles sum to a whole number of turns (+-360 degrees) returned no channels. The holonomy is then a translation, so it now returns the Burgers channels (law `D×D (turn)`, each with `frank_angle_deg`, `contains_vacuum` False). Totals of 360 degrees or more that are not a whole number of turns still return no channels.
- Licence: changed from GPL-3.0-only to AGPL-3.0-only on 2026-11-25. Versions published before this change remain available under GPL-3.0-only. Anyone who runs a modified version as a network service must offer its source to that service's users.
- Correctness pass (C1): unified 0-based seeded layer-by-layer growth (`vertex`, `edge`, `face`, `cell`, `Dirichlet`) across 2D plane lattices (opening on `p6m` honeycomb with 4 layers), $S^2 / H^2$ curved tilings, $S^3 / E^3 / H^3$ honeycombs (touching and face-shell rules up to 120,000 cells, plus radius-$R$ ball truncation for paracompact $H^3$), 230 space-group `points3d` catalogue (at least 7 entries per group by shells-until-connected bonds), 80 layer-group slabs with symmetry-checked stacking, and Dirichlet-seeded growth of defected crystals (disclinations, dislocations, Stone-Wales); 2D boundary conditions per lattice (open, 1-pair cylinder, 2-pair torus, 3-pair Wigner-Seitz hexagonal torus, Moebius, Klein) with cell-symmetry verdicts and explicit non-isometric 3D picture notes; adjacent-bond vertex angle-shear overlay; and quarantine/hiding of postponed Analysis, AO elastic, 3D manifold embeddings, and un-reglued 3D line defects.
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
