# Desktop audit and live-combining performance

Working branch: `app-audit-and-combine-speed`. Commit and feature-branch push authorized; manual review pending.

## Scope and invariants

No frozen mathematical decision in `RUST_PLAN.md` §9 was changed. Sphere/RP2 still tries face centres before
sites; p4g's single-class RP2 construction remains NOT FOUND. Charges keep the unreduced Frank angle in
the lifted proper group, including the distinction between +180° and -180°. Core equivalence still uses
degree-labelled graphs and exact VF2 isomorphism, not the drawing or a hash alone.

The author approved additive technical Rust helper APIs during this audit. Existing public signatures
are retained. The root build backend remains pure-Python setuptools; `papertilling/legacy/` is untouched.
Three already-oversized Python modules were split mechanically: graph helpers are re-exported from their
old modules, and reader tests moved to `tests/test_build_readers.py`. No Python algorithm was changed.

## Findings and regression coverage

Line references below refer to the audit working tree; test names identify the durable regression coverage.
Correctness-only fixes have no meaningful millisecond before/after measurement.

### Gallery and cross-tab state

- `gallery_tab.rs`, `live_plane/tuned.rs`: parameter edits only changed a private value list; `active_item()`
  continued exporting the original recipe, so all three other tabs used the untuned lattice. The active
  item now contains the tuned recipe/frame/readout. Tests:
  `slider_moves_reach_active_item_and_bump_revision_but_view_changes_do_not`,
  `tuned_named_lattice_keeps_its_frame_and_draws_what_it_exports`,
  `tuned_gallery_geometry_reaches_surfaces_defects_and_combine`.
- `gallery_tab/ui.rs`: egui's slider formatting/clamping changed values even without input, e.g. a stored
  value 1.154 became 1.15 on the first frame. Rendering no longer mutates the parameter; step rounding and
  snapping happen only on an edit. Test: `drawing_the_panel_never_moves_a_parameter`.
- `plane/tuning/sliders.rs`, `plane/tuning.rs`: the controller now keeps the base recipe's nominal ranges,
  applies the Python coupled-slider refresh, and rejects NaN edits. Cell-parameter `current_ok` now agrees
  with Python independently of site-parameter validity. The fixture-backed `gallery_sliders` tests compare
  limits, forbidden spans and special values; Gallery readout tests also cover the raised symmetry at special loci.
- `app.rs`: every Gallery selection eagerly rebuilt hidden Defects/Combine state on the UI thread.
  Reloads are now lazy on tab activation, keyed by the actual recipe and frame, not just the lattice label.
  Test: `tabs_reload_lazily_for_the_current_gallery_lattice`.

### Surfaces

- `live_ops.rs`, `surfaces_tab.rs`: h/k controls silently substituted another size, or ignored the controls
  entirely for small genus surfaces. There are now explicit table-default and custom-size modes; custom
  sizes are used exactly as shown and target-specific ranges are labelled. Typed refusals preserve
  IMPOSSIBLE versus NOT FOUND. Tests: `custom_sizes_are_used_as_shown_with_hand_counted_tori`,
  `auto_size_builds_the_surface_of_the_table_cell_and_refusals_carry_their_verdict`.
- `surfaces_tab.rs`: a refused build left the previous surface's 2D net visible. Both viewports now clear.
  Test: `net_and_3d_follow_every_control_change_and_refusals_clear_both`.
- `mesh3d/rp2.rs`: the RP2 quotient view joined face corners from inconsistent antipodal copies, tearing
  triangles across the sphere. A consistent hemisphere representative now draws each face once.
  Test: `rp2_double_cover_is_a_sphere_and_the_quotient_view_has_no_torn_faces`, including Petersen and hemi-cube controls.
- `view3d.rs`, `mesh3d/sort.rs`: translucent faces were submitted in buffer order. They are now sorted
  back-to-front when the view changes, with depth-tested opaque markers/bonds and two-sided shading.
  Test: `back_to_front_order_is_a_permutation_sorted_by_view_depth`.
- `view3d.rs`: yaw grew indefinitely; it now wraps. Pitch remains clamped. Test:
  `orbit_wraps_yaw_clamps_pitch_and_reset_restores_the_default_view`.
- `mesh3d.rs`, `scene2d.rs`: 2D and 3D used different ring colours. They now share the palette, including
  chart-local net faces; faces across quotient seams are not filled across the chart.
  Tests: `face_palette_distinguishes_rings_and_matches_the_2d_ring_palette`,
  `net_rings_use_the_3d_palette_without_filling_across_seams`.
- `backend.rs`: the 17×8 verdict matrix blocked its first display (~1.3 s). It now prefetches on a worker;
  the UI checks readiness without waiting. Matrix clicks select the first points-rule lattice, rather
  than silently selecting the first lattice of the current Gallery rule. Per-active-lattice verdicts
  are separately keyed by the complete recipe and frame.

### Defects and exports

- `defect/walls.rs`: several generic Euclidean-normaliser index constants disagreed with Python's ITA table
  (including p2, p3, p3m1, cmm and p4m). The constants now match the reference. The 38-lattice
  `defect_catalogue_fixture` gate checks the complete catalogue, labels, classes, vectors and wall metadata.
- `live_ops.rs`: Möbius/glide wall direction readouts were always (1,0), irrespective of the mirror/glide
  direction. They now use the catalogue's primitive basis and line period.
- `scene_crystal.rs`, `core_panel/card.rs`: ghost-stub endpoints are absolute positions, but the renderer added
  the site position again; it also drew the full ghost bond underneath. Both renderers now use the absolute
  endpoints, omit full ghost bonds, and do not fill faces through ghost edges.
  Tests: `every_mobius_seam_of_38_lattices_keeps_stubs_and_draws_no_bond_across`,
  `core_cards_draw_absolute_ghost_endpoints_and_never_the_full_ghost_bond`.
- `scene_crystal.rs`: translated crystals were coloured by distance from the world origin, not from their
  own centroid. Core-face colouring now follows the translated crystal.
- `live_defects_cache.rs`: every Defects selection rebuilt the whole lattice catalogue. A bounded cache
  now reuses the cell, catalogue and core-detection data. Invalid recipes give a diagnostic rather than a panic.
- `defects_tab.rs`: the full 1,271-build sweep found repairs lasting up to 5.59 s. Catalogue preparation
  and subsequent repairs now use a generation-counted worker too. Superseded results cannot overwrite
  the current lattice; old pictures/readouts are explicitly marked stale, and exports wait for completion.
  Test: `slow_repairs_submit_without_blocking_and_newest_lattice_wins`, including exact backend-result equality.
- `export.rs`, `export_raster.rs`: PNG omitted text, arrows and reliable polygon fills; unknown extensions
  silently selected a format, and dimensions were unchecked. Exports now include annotations, pins, arrows,
  even-odd polygon fill, clipped raster operations, explicit extension handling and allocation bounds.
  Tests cover all scene variants, XML escaping/parsing, PNG decode/dimensions, alpha, dashes, text, empty and non-finite scenes.
- `canvas2d.rs`, `app.rs`: Combine pins were absent from exported pictures. They now accompany the exported
  scene, and exporting an in-flight/mid-drag crystal is refused rather than mixing old geometry with new pins.

### Combine correctness and interaction

- `canvas2d/hit.rs`: sparse marker IDs were treated as vector offsets; clicking a pin in placement mode
  also placed a duplicate. Picking now uses marker IDs, distinguishes pin clicks from empty-canvas clicks,
  and preserves screen-pixel hit radii. Pointer-anchored zoom no longer moves the world point under a dragged pin.
  Headless pointer-event tests cover picking, pan/drag arbitration, clicks, zoom and release.
- `live_combine.rs`: catalogue Burgers coefficients were converted using the drawn cell basis, not the
  catalogue's reduced primitive basis. Likewise, a dislocation with `at: null` was displayed at a different
  default position from the surgical builder. Both now use the corresponding primitive basis.
  Fixture-backed `live_combine/parity_tests.rs` checks these conversions and filtered split-class snaps.
- `combine_tab.rs`: a refused surgery's defect-free placeholder was still compared and reported as
  equivalent to the perfect lattice. A refused build cannot be compared or exported with an equivalence
  summary. Test: `refused_surgery_is_not_reported_as_equivalent` (confirmed to fail before the guard).
- `combine_tab.rs`, `combine_ui.rs`: importing a Python patch radius silently clamped it twice (import ≤16,
  slider ≤13). The import and slider now share 2.4–30; any remaining range adjustment is reported explicitly.
  Test: `json_import_keeps_python_patch_radius_through_a_ui_frame` (old slider changed 17.25 to 13.0).
- `combine_types.rs`, core manifest: unsupported config versions were accepted, optional/null fields and
  Python aliases were inconsistently handled, and default JSON float parsing could lose round-trip precision.
  Parsing now validates the version, supports Python's missing/null metadata and aliases, and enables
  serde_json `float_roundtrip`. Cross-language fixtures include non-zero `frame.origin`, cuts and missing `at`.
- `live_combine.rs`: imported redundant group metadata could override the actual recipe group; Python does
  not do this. The recipe is now authoritative. Degenerate frames are refused before state changes.
  Test: `recipe_controls_the_imported_group_and_unknown_formats_do_not_mutate_state`.
- `surgery.rs`: non-finite charges/positions/cuts, degenerate cells and unrepresentable disk index ranges
  could reach unchecked integer/index operations. They now return errors. Test:
  `malformed_numeric_specs_and_patch_extents_refuse_without_panicking`.
- `core_panel.rs`: completion-card caps could hide matches. Selection now follows Python: all Volterra
  cards, matching completions first, a default five completions when matched, and a thirty-card unmatched
  gallery cap, with an explicit show-all option. Tests also cover empty/error/many-card states.
- `core_panel.rs`: a newly added card cache initially fingerprinted only the first site and array lengths.
  It now hashes all drawing inputs, so a moved later site cannot leave a stale picture.
  Test: `changing_any_card_position_invalidates_the_cached_picture`.

## Combine architecture and performance

- `combine_worker.rs`: independent build, lattice-preparation and comparison workers keep egui responsive.
  Jobs carry generations; queued obsolete jobs and old results are dropped. Errors/panics are returned as
  diagnostics. Results request repaint. The comparison worker owns its bounded single-defect cache.
- `combine_tab.rs`: placement/preset/release controls only submit work. During dragging, pins update
  immediately and in-flight previews are allowed to finish; further moves coalesce to the newest positions.
  Without coalescing, discarding every preview older than the newest pointer event could starve the canvas.
  Release requests a final build and then a comparison. Previous verdicts/cards are explicitly labelled stale.
  Tests: `latest_worker_drops_superseded_jobs_and_recovers_from_panics`,
  `drag_previews_then_compares_the_released_configuration_once`,
  `lattice_reload_drops_obsolete_builds_and_keeps_the_ui_responsive`.
- `defect/single_cache.rs`: the scan and gallery no longer depend on the multi-pin positions. Keys retain
  the exact cell and Python-compatible six-decimal Burgers value, radius, grid and drawing mode. Candidate
  degree-labelled graphs are retained for WL-prefiltered exact VF2 matching. Hash equality alone never
  establishes equivalence. Tests compare cold and warm results, translated configurations and bounded eviction.
- `equivalence.rs::CentreSnapper`: centre classes are prepared once, not rebuilt on every hover/drag snap.
- `plane/relax_normal.rs`: sparse triplet ordering, sparsity construction and symbolic Cholesky analysis
  were repeated for every factorisation. They are now reused, including pinned and Levenberg-shifted paths.
  Four regression tests compare positions, costs, iteration counts and solves bit-for-bit with the old solver.
- `plane/relax_preview.rs`: a bounded matrix-free, block-preconditioned conjugate-gradient preview changes
  positions/display stubs only; full relaxation/planarity/core inspection is deferred to the final build.
  Preview tests check finite coordinates, decreasing energy and unchanged mathematical graph data.
- `core_panel.rs`: card primitives are cached and compiled only when scrolled into view.
- `live_combine.rs`: enormous accidental pin separations are refused by an explicit desktop-view budget
  (~20,000 estimated sites), not called mathematically impossible. No charge/equivalence is invented for
  the stand-in picture. The core's valid mathematical construction and Python package remain unrestricted.
  Test: `extreme_pin_separation_is_a_view_budget_refusal_not_a_false_charge`.

The suspected all-pairs planarity and Volterra welding paths were inspected: normal crossing, crowding
and site-on-bond checks already use spatial grids, and Volterra chart welding already uses keyed maps.
They were not replaced merely on suspicion. The measured dominant cost was relaxation, then repeated
single-defect scans/drawing. Giant-segment planarity retains its existing conservative fallback.

### Measurements and reproduction

Release builds, same machine, wall time, `# time` printed by the harnesses. These are observations, not
hard real-time guarantees. Timings can vary with other test/build processes.

Baseline before the relaxation/cache/worker changes:

| Case | Sites | Build/readout total | Relaxation | Planarity |
|---|---:|---:|---:|---:|
| square ±90 pair | 497 | 176.0 ms | 128.4 ms | 4.2 ms |
| square pair, larger disk | 2,007 | 641.8 ms | 494.8 ms | 25.1 ms |
| honeycomb pair | 475 | 105.6 ms | 79.2 ms | 3.3 ms |
| honeycomb pair, larger disk | 1,896 | 1,940.3 ms | 1,683.3 ms | 13.5 ms |
| kagome pair | 505 | 117.5 ms | 77.5 ms | 3.9 ms |
| kagome pair, larger disk | 2,014 | 629.9 ms | 490.8 ms | 19.8 ms |

Original drawn single-defect comparisons at grid 3: square pair 180.9 ms, square dislocation 191.3 ms,
honeycomb pair 429.5 ms, honeycomb dislocation 421.1 ms. These ran synchronously in UI event handlers.
The original 484-site Stage 5 benchmark failed at 153.32 ms; it passes after symbolic reuse at 75.22 ms.
Unfiltered snap baselines were square 1.34 ms, kagome 1.61 ms, Shastry–Sutherland 4.91 ms and chiral
triangular 6.82 ms per call; cached snapping is about 0.002 ms per call.

Post-change measurements of the same cases:

| Case | Final build/readout total | Relaxation alone | In-drag preview total |
|---|---:|---:|---:|
| square pair, 497 sites | 51.4 ms | 29.6 ms | 19.8 ms |
| square pair, 2,007 sites | 308.9 ms | 132.7 ms | 96.0 ms |
| honeycomb pair, 475 sites | 41.3 ms | 25.8 ms | 15.9 ms |
| honeycomb pair, 1,896 sites | 557.5 ms | 413.8 ms | 47.6 ms |
| kagome pair, 505 sites | 59.8 ms | 39.6 ms | 17.3 ms |
| kagome pair, 2,014 sites | 254.8 ms | 139.4 ms | 58.2 ms |

Each stage is timed independently; stage timings are not a decomposition of the separate total sample.
All the above work now runs off the UI thread. Preview wall times include surgery and charge readout,
not just the approximate solver's eight-millisecond hint (the hint is not a hard deadline).

Warm drawn comparisons measured 15.7–24.6 ms end-to-end in the worker, versus 180.9–429.5 ms original
uncached drawn comparisons. The remaining warm work includes configuration-core extraction and drawing;
it is **not** claimed to be under one millisecond. Candidate WL/VF2 matching is timed separately by the
cache at **0.047–0.081 ms warm**, and submission is measured separately from completion.

The headless held-pointer harness measures both side-panel and canvas work while worker builds run:

| Lattice | Drag-frame median | p95 | Maximum |
|---|---:|---:|---:|
| square | 1.419 ms | 3.677 ms | 7.714 ms |
| honeycomb | 1.173 ms | 2.300 ms | 7.160 ms |
| Shastry–Sutherland | 1.695 ms | 4.785 ms | 8.993 ms |
| chiral triangular | 1.302 ms | 2.455 ms | 7.157 ms |

Placement submission measured 0.026–0.033 ms, release 0.007–0.011 ms, comparison submission
0.167–0.345 ms, and lattice-reload submission 0.304–1.604 ms. These are not worker completion times.
The harness warms the UI and holds the primary pointer through every synthetic drag frame; a prior
unheld-pointer harness was corrected because it exercised release on each frame instead of a continuous drag.

Run the same app paths, including full drag-frame and cached/cold comparisons:

```text
cargo test --release --manifest-path rust/Cargo.toml -p papertilling-app combine_bench -- --ignored --nocapture --test-threads=1
cargo run --release --manifest-path rust/Cargo.toml -p papertilling-core --example combine_bench -- live big reps=3
```

## Verification status

- Latest-tree workspace clippy, all targets, `-D warnings`: passed.
- Latest-tree release workspace: **141 passed**, no failures; five intentionally ignored profiling/sweep tests.
  The four Combine profiling tests and the full Defects sweep were also run separately and passed.
  The panic printed by the worker-recovery test is deliberate and caught; that test passes.
- Full Defects sweep: **1,271 builds across 38 lattices**, 110.29 s; slowest repair 5.59 s.
  After backgrounding Defects, worst-repair submission measured **11.334 ms** in the integrated run;
  its completed crystal and readout match the synchronous backend exactly. The timing includes compiling
  the previous scene for display, not the repair itself.
- Pure-Python default pytest (`PAPERTILLING_NO_RUST=1`): **342 passed, 1 skipped, 137 deselected** in 315.98 s.
  The skipped direct-binding check requires the optional extension.
- Fresh release ABI3 wheel rebuilt and extracted to a separate test location; bridge activation was
  asserted before the extension-enabled full default pytest run: **343 passed, 137 deselected**,
  no skips or failures, in **248.71 s**. Existing spglib deprecation warnings and the deliberate
  invalid-lattice warning remain.
- Follow-up full pure-Python slow suite (`PAPERTILLING_NO_RUST=1 python -m pytest -v -m slow`):
  **137 passed, 343 deselected** in **226.62 s**, no failures. The legacy-gate timing pilots passed
  in 0.369 s (square) and 0.857 s (Shastry–Sutherland); the full 104-case gate fixture completed
  in 17.24 s with its existing process pool. Logs and JUnit results are retained outside the repository.
  With the fresh Rust extension enabled, the same slow suite also passed: **137 passed,
  343 deselected** in **259.77 s**, no failures or skips. Across the disjoint fast and slow
  selections, this is **480 passed with Rust**, and **479 passed plus one expected
  direct-binding skip without Rust**.
- Rebuilt native release executable: smoke passed in **2.680 s**, exit 0:
  `PAPERTILLING_SMOKE_OK adapter=AMD Radeon(TM) Graphics backend=Vulkan frames=5 uploads=5`.
  The smoke visits the surface rendering states; it is not a manual pixel-level UI review.
- Private scan: no hits. `git diff --check`: clean. All **177** non-vendored Python/Rust modules are
  strictly below 600 actual lines. The frozen plan, vendored reference tree and root Python build backend
  were not changed.

### Deliverables

`python tools/build_windows_installer.py 0.2.1` produced the standalone .NET Windows installer.
Its embedded application resource was decompressed without running the installer and SHA-256 checked
against the freshly built executable.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `dist/papertilling-app.exe` | 23,327,744 | `96A31548969FCCB920ABFD7E117D73A98EA993232EBC9063BCF51582D0E29B13` |
| `dist/papertilling-0.2.1-windows-x86_64-setup.exe` | 8,171,008 | `C394F1EC8E98A5F7A72E906C0C685BA94BDC80599DDD5B640F156FBE07313D7F` |

The installer assembly version is **0.2.1.0**. Package/core/app version strings deliberately remain
**0.2.0**; the installer argument is not a package-version bump. The native executable is also retained
at `rust/target/release/papertilling-app.exe`. No installation/uninstallation was performed. A follow-up
read-only check found an existing registered installation whose executable differs from this rebuild;
the author explicitly chose to preserve it rather than temporarily replace/remove it for a lifecycle test.

About **9.2 GiB** of generated build intermediates were removed from the main and three worktree caches.
Cargo refused the older untagged main cache; after inspecting it, only its known generated subdirectories
and library/debug outputs were removed. The tagged worktree caches were cleaned with Cargo. All source
worktrees, finished artifacts, the fresh wheel/test extension and validation logs were retained; the two
retained executable copies still have identical SHA-256 values after cleanup.

### Remaining limitations

- Some complex Gallery slider refreshes remain synchronous (measured example ~0.33 s). Custom Surfaces
  builds remain synchronous; the offered-size sweep measured a maximum of ~0.89 s. The reference-table
  prefetch, Defects repairs and Combine workers do not block those interactions.
- Preview relaxation is approximate and time-bounded only by a hint plus iteration limits; the final
  released configuration uses the unchanged full mathematical solver.
- No manual native-pixel inspection was possible because computer-use tools were unavailable. Coverage
  consists of headless egui interaction tests, independently checked drawing primitives, decoded exports,
  Python fixtures and the real Vulkan smoke run.
- PNG text uses a small bitmap font and a limited set of symbol/accent substitutions, not the full native
  font stack. SVG retains its text as Unicode.
- Full installer install/uninstall testing remains unverified: the author chose to preserve the existing
  installation, and the current uninstaller requires native confirmation dialogs. Embedded payload
  integrity is verified, but this is not a substitute for the lifecycle test.
- The author authorized committing and pushing this feature branch after the automated checks passed.
  Manual inspection remains pending; no default-branch merge or release publication was authorized.

## Source locations

Current working-tree entry points (all paths are repository-relative):

| Finding | Location |
|---|---|
| Tuned recipes and revision | `rust/papertilling-app/src/gallery_tab.rs:103,223` |
| Input-only slider mutation | `rust/papertilling-app/src/gallery_tab/ui.rs:295` |
| Coupled limits / cell-parameter status | `rust/papertilling-core/src/plane/tuning/sliders.rs:123,160`; `rust/papertilling-core/src/plane/tuning.rs:246` |
| Lazy cross-tab reload / matrix lattice / export guard | `rust/papertilling-app/src/app.rs:112,152,240` |
| Exact surface sizes / wall directions / cached defect building | `rust/papertilling-app/src/live_ops.rs:116,188,387` |
| Refusal clearing / active-lattice verdict key / nonblocking matrix | `rust/papertilling-app/src/surfaces_tab.rs:152,163,296` |
| RP2 lifts | `rust/papertilling-app/src/mesh3d/rp2.rs:26` |
| Mesh invalidation / orbit / sorted GPU upload | `rust/papertilling-app/src/view3d.rs:70,91,332` |
| Background verdict table | `rust/papertilling-app/src/backend.rs:349,406` |
| Normaliser indices | `rust/papertilling-core/src/defect/walls.rs:17` |
| Ghost endpoints / translated face colouring | `rust/papertilling-app/src/scene_crystal.rs:12,222` |
| Per-lattice defect context | `rust/papertilling-app/src/live_defects_cache.rs:31` |
| Background Defects builds and stale-result handling | `rust/papertilling-app/src/defects_tab.rs:98,117,143,168` |
| SVG/PNG export and validation | `rust/papertilling-app/src/export.rs:30,237`; `rust/papertilling-app/src/export_raster.rs:21,47,150` |
| Exported pins / hit testing / duplicate click prevention | `rust/papertilling-app/src/canvas2d.rs:95,181,206`; `rust/papertilling-app/src/canvas2d/hit.rs` |
| Primitive Burgers basis / default pin position / view budget / import geometry | `rust/papertilling-app/src/live_combine.rs:249,267,347,469` |
| Build submission / refused-build comparison / preview coalescing / JSON import | `rust/papertilling-app/src/combine_tab.rs:172,196,264,445` |
| JSON version/null/alias handling | `rust/papertilling-core/src/defect/combine_types.rs:203`; `rust/papertilling-core/src/defect/surgery.rs:19` |
| Numeric and index-range validation | `rust/papertilling-core/src/defect/surgery.rs:57,117,208` |
| Card cache / matching completion selection / clipped rendering | `rust/papertilling-app/src/core_panel.rs:56,90,200`; `rust/papertilling-app/src/core_panel/card.rs:12` |
| Generation-counted workers | `rust/papertilling-app/src/combine_worker.rs:50,159,172` |
| Single-defect gallery and candidate cache | `rust/papertilling-core/src/defect/single_cache.rs:59,174` |
| Cached centre snapping | `rust/papertilling-core/src/defect/equivalence.rs:122` |
| Reused symbolic factorisation | `rust/papertilling-core/src/plane/relax_normal.rs:36` |
| Matrix-free drag preview | `rust/papertilling-core/src/plane/relax_preview.rs:172` |
