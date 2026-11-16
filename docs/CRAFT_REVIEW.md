# CraftAnything feature integration review

This follow-up reviews the uncommitted CraftAnything changes built on audit commit `3f31d5c`.
The earlier parity/performance audit remains in `RUST_AUDIT.md`. The author approved reviewing,
fixing, testing, committing and pushing this combined feature branch, and updating the installed
Windows app. No default-branch merge or release publication is part of this work.

## Scope

- Persistent lattice aliases and name propagation through tabs, image exports and Notebook 3 JSON.
- Pointer-anchored viewport zoom and enlarged core-card inspection.
- Defects and Combine 3D elastic-sheet visualization, alongside the unchanged 2D mathematical crystal.
- Regression safety for asynchronous builds, comparison, import/export and tab transitions.

The frozen conventions in `RUST_PLAN.md` §9, `papertilling/legacy/`, and the root pure-Python
setuptools backend are unchanged. Elastic-sheet heights and energies are approximate visualization
outputs, not alternative definitions of holonomy, charge, core equivalence or surface feasibility.
Craft workspace state, logs and capture directories are kept locally and excluded from version control.

## Reproduced integration regressions

Seven added tests failed against the incoming feature implementation before the fixes:

| Problem | Fix and regression coverage |
|---|---|
| Renaming in Gallery reset hidden Combine pins and Defects selections. | Tab identity uses the stable alias key plus group, recipe and frame, never the mutable display name. `app/rename_tests.rs::review_rename_preserves_hidden_combine_pins_and_defects_selection`. |
| Saving an unchanged name replaced a non-Gallery scene with Gallery geometry. | Alias refresh only redraws when the display name changes. `review_saving_an_unchanged_name_never_changes_another_tabs_scene`. |
| A colliding alias identified Honeycomb as Triangular during JSON import. | Import prefers exact recipe/frame, then structural D-symbol matches; a name cannot establish identity. `review_import_identity_uses_geometry_not_a_colliding_alias`. |
| Renaming an unrelated Gallery lattice erased an imported experiment name. | Imported overrides are renamed only when their stable identity matches the renamed lattice. `review_renaming_an_unrelated_lattice_keeps_an_imported_experiment_name`. |
| Changing bending stiffness superseded a pending Combine clear with an old crystal. | A stiffness-only job may reuse geometry only when no newer geometry is pending and the completed picture is final. `features_gate_tests.rs::review_kappa_change_preserves_a_pending_combine_clear`. |
| Changing bending stiffness likewise discarded a pending Defects selection. | Reuse is disabled while a newer build is pending. `review_kappa_change_preserves_a_pending_defects_selection`. |
| Switching lattices temporarily showed the old lattice's 3D sheet. | Preparing a new lattice clears the 3D viewport; retained same-lattice results are explicitly labelled stale. `review_lattice_switch_never_displays_the_previous_lattices_sheet`. |

Additional safeguards:

- Failed worker builds cannot compare their previous crystal against new specifications.
- Preview warm-start identity includes the complete cell, including basis and bonds, rather than only
  the two cell vectors. Spatial bucket coordinates use overflow-safe neighbouring keys.
- Notebook JSON methods were mechanically separated into `combine_json.rs` to keep modules below
  600 lines without changing their signatures or serialized format.
- Native smoke coverage now visits five surface states and both Defects/Combine sheet viewports,
  waiting for pending geometry before declaring success.

## Parallel review areas & final verification fixes

The sheet solver, alias/zoom implementation and combined UI integration received dedicated reviews and
scoped fixes:

- **3D elastic-sheet unfolding & hinges (`sheet3d_energy.rs`, `sheet3d_topology.rs`, `sheet3d.rs`, `sheet3d_opt.rs`, `sheet3d_tests.rs`, `sheet3d_review_tests.rs`):**
  - Removed 2D `align_sign` flips in `build_hinges` that locked folded surgical triangles into permanent 0-degree creases, and added deterministic chart unfolding (`prepare_planar_chart`: `relax_preview` + `local_tutte` + `global_tutte`) instead of a wall-clock `12.0` ms timeout.
  - Updated `gate4_dipole_dislocation_elastic_screening` to check the physical Seung-Nelson screening metrics (`rms|z|`: `0.409 < 0.5 * 1.224`, `E_2d`: `0.0932 << 1.5165`, `E_bend`: `0.0317 < 0.0454`, total 3D energy `0.0456 < 0.0684`, and `>85%` stretch relief `0.0139 < 0.25 * 0.0932`), plus `sheet3d_review_raw_folded_dipole_unfolds_without_creasing` and `!nlen.is_finite()` protection in `center_and_align_horizontal`.
  - Physical sheet topology, display-only ghost annotations, principal-plane alignment, numerical input validation, and explicit optimization/topology diagnostics.
- **UI & state synchronization (`combine_tab.rs`, `combine_ui.rs`, `combine_worker.rs`, `defects_tab.rs`, `features_gate_tests.rs`):**
  - Submitting a build or `kappa_b` update now marks `view3d_dirty` immediately so settled 3D viewports show `"previous result — stale"` while background work runs (`review_pending_builds_mark_settled_3d_caption_stale`).
  - `reload_for_item` clears cached 3D sheets (`CombineSheet3dBundle::empty()` / `empty_sheet3d()`), failed lattice loads guard all `CombineTabState` build/preset/placement actions (`lattice_unavailable()`) and render `"3D unavailable"` (`review_failed_lattice_reload_never_resurrects_previous_sheet`), and non-reload `DefectsTabState` build errors update both the 2D note and 3D caption.
- **Native smoke hardening & Clippy cleanup (`view3d.rs`, `surfaces_tab.rs`, `app.rs`, `app/smoke.rs`, `main.rs`, `core_panel/inspector/tests.rs`, `mesh3d/sheet.rs`):**
  - Replaced cumulative paint counting with `painted_generation`/`painted_primitives` tracking (`is_current_mesh_painted()`, `has_geometry()`), checked stage refusals/errors and stale captions, and added `smoke_outcome` exit-code gating (`app::smoke::tests`).
  - Fixed `egui::Id::unique` and `clippy::let_and_return` warnings under `-D warnings`.
- **Alias storage & cursor-anchored zoom (`aliases.rs`, `canvas2d/hit.rs`, `core_panel/inspector.rs`, `view3d.rs`):**
  - Atomic alias-file replacement via sibling staging + sidecar lock, concurrent independent stores, keyboard shortcut ownership, modifier/Shift-wheel handling, HUD-crossing drags and cursor-ray-centred 3D zoom.

## Verification & measurements

All checks below passed on the final verified tree (every non-legacy `.rs`/`.py` module `< 600` lines; `private_scan.py` `0` hits; `papertilling/legacy/`, `pyproject.toml` and `docs/RUST_PLAN.md` unchanged):

- `cargo clippy --manifest-path rust/Cargo.toml --workspace --all-targets -- -D warnings`: **0 warnings/errors** (`25.39 s`).
- `cargo test --release --manifest-path rust/Cargo.toml --workspace`: **197 passed, 0 failed, 6 ignored** (`206.28 s`):
  - `papertilling-app`: **113 passed** (`19.25 s`; `6` timing/sweep benchmarks ignored by default)
  - `papertilling-core`: **57 unit + 23 integration = 80 passed** (`19.49 s` unit + integration targets)
  - `papertilling-py`: **4 passed** (`0.08 s`)
- Combine UI & 3D sheet worker benchmarks (`cargo test --release -p papertilling-app combine_bench_ -- --ignored --nocapture --test-threads=1`):
  - UI `place`: `0.024-0.026 ms`; drag frames median `0.85-1.13 ms` (p95 `2.58-4.68 ms`, max `7.63-8.24 ms`); UI `release`: `0.005-0.006 ms`; `compare` submit: `0.11-0.21 ms` (warm cache `14.3-16.6 ms`); lattice reload: `0.93-1.53 ms`.
  - 2D+3D worker (`N=497` square / `N=475` honeycomb): `124.1 ms` / `99.1 ms` full (`57.4 ms` / `45.3 ms` preview).
- Python test matrix (`pytest`):
  - With rebuilt `papertilling-rs` (`0.2.0`) wheel (`190.39 s` build): **343 passed** fast (`251.35 s`) + **137 passed** slow (`254.58 s`).
  - Without Rust (`PAPERTILLING_NO_RUST=1`): **342 passed, 1 skipped** fast (`239.92 s`; 137 slow also passed in `324.93 s` prior to final Rust-only UI fixes).
- Release binary, 7-stage Vulkan smoke test, and Windows installer (`0.2.1`):
  - Release build `papertilling-app`: `76.29 s` (`SHA256: 569F58B39AB98F5120C62A9B0854FCA529A93201643A29C31FF83E85C2B6A4CA`).
  - Native Vulkan `--smoke-test` (`AMD Radeon(TM) Graphics`): **passed** (`2.03 s`, `frames=11`, `uploads=8`, `stages=7`).
  - Windows installer (`dist/papertilling-0.2.1-windows-x86_64-setup.exe`, `8.34 MB`, `2.83 s`): `SHA256: E4730C85B414AF41AE754E4580B25B0D3539C8A82C7BDA5D3319EF65B442BCF4`.
  - Silent update of local Windows installation verified (`2.296 s`, matching app/uninstaller/licence hashes, registry `0.2.1`, Start Menu + Desktop shortcuts, and installed native Vulkan smoke test `2.938 s`).

## Known limitations

- The sheet optimizer is a finite-budget, local numerical model. Read convergence/topology
  diagnostics before interpreting heights, energies or a particular buckling branch.
- SVG and JSON preserve Unicode aliases. PNG text retains the existing limited bitmap character set.
- Automated headless UI interaction tests and real native GPU smoke are not a substitute for a
  manual inspection of every control and exported picture.
- Full uninstall testing requires native confirmation dialogs and is not performed by the silent
  update verification.
