# Papertilling

Explore lattices and defects in the Euclidean plane, 3D crystals, layers, and regular hyperbolic tilings and honeycombs.

## Desktop

Download a [Windows, macOS, or Linux release](https://github.com/antimajorana/papertilling/releases/latest), or build from this checkout with Rust 1.95+.

```sh
cargo run --locked --release --manifest-path rust/Cargo.toml -p papertilling-app
```

Browse the 17 wallpaper groups and surfaces under **Lattice**. In **Defects**, use **Place & fuse** or **Fuse & compare** for planar defects. Open **Advanced → All Views → Forge** for space-group, layer-group, and hyperbolic charges.

## Python

With Python 3.10+, run `python -m pip install .` from this checkout. Install `'.[notebooks]'` to run the [three notebooks](notebooks/).

```python
import matplotlib.pyplot as plt
import papertilling as pt

lattice = pt.named("Kagomé")
b = pt.DefectCatalogue(lattice).dislocations[0]["b"]
pt.build(lattice, dict(kind="dislocation", b=b, at=(0.37, 0.41))).plot()
plt.show()
```

## Scope

Planar core equivalence compares the measured topological charge and the local bond graph. Catalogue and core searches are bounded. **Not found** reports a failed construction within the attempted settings. Elastic sheets and curved 3D surface previews are models with numerical and geometric limits.

[Scientific scope and wall examples](docs/SCIENTIFIC_SCOPE.md) · [Development and tests](docs/DEVELOPMENT.md) · [Design and conventions](docs/RUST_PLAN.md) · [Changelog](CHANGELOG.md)

[GPL-3.0-only](LICENSE) · Copyright © 2026 antimajorana
