<img align="right" src="assets/escher.png" width="112" height="112" alt="Circular Escher logo with a central pentagon and three hyperbolic growth layers on black">

# Escher

Escher lets you build lattices and explore how their defects combine and bend a sheet. The logo shows a plaquette-centered hyperbolic `{5,5}` tiling with three layers around its central pentagon.

## Install

Run `*-windows-x86_64-setup.exe` on 64-bit Windows. The release also includes portable Windows, macOS, and Linux builds. [Latest downloads](https://github.com/antimajorana/Escher/releases/latest)

## Try it on the honeycomb

1. Use **Find lattice** to select **Honeycomb**. In **Defects → Fuse & compare**, choose **+α + −α → Dislocation**.
2. The pair has two local rotations of +60° and −60°. Its enclosing loop measures zero net rotation and a nonzero translation, the Burgers vector. Compare its pentagon–heptagon core with the single dislocation.
3. Choose **Open pair in Place & fuse**, then **Split view**. Set **Sheet colour** to **Angle deficit** to visualize curvature in the sheet model.

A match requires both the loop charge and the core bond pattern, within the search limits. [Scientific scope and mirror/glide examples](docs/SCIENTIFIC_SCOPE.md)

Browse 17 wallpaper groups and surface models under **Lattice**. Compare space, layer, and regular hyperbolic groups under **Advanced → All Views → Forge**.

## Python

Run `python -m pip install '.[notebooks]'` from this checkout with Python 3.10+, then open the worked notebooks.

[Notebooks](notebooks/) · [Build from source and run tests](docs/DEVELOPMENT.md#rust-core-desktop-app-and-optional-python-extension) · [Changelog](CHANGELOG.md) · [GPL-3.0-only](LICENSE)

Copyright © 2026 antimajorana
