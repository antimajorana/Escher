"""Draw the pictures used in README.md into docs/img/ (from the package itself).

    python tools/make_readme_images.py [OUTDIR]        default OUTDIR: docs/img next to this tools/ folder

gallery.png: the named lattices (papertilling.gallery()); groups.png: the first lattice of each of the 17 wallpaper
groups (stored "points" lists); defect.png: a dislocation in the Kagome lattice.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import papertilling as pt

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / name, dpi=70, bbox_inches="tight")
    plt.close(fig)
    print(name, (OUT / name).stat().st_size, "bytes")


save(pt.gallery(ncols=6, n=3, size=2.2), "gallery.png")

cat = pt.load_catalogue()
fig, axes = plt.subplots(3, 6, figsize=(13, 6.8))
for ax in axes.flat:
    ax.axis("off")
for ax, (group, res) in zip(axes.flat, cat.items()):
    res.lattices[0].plot(n=2, ax=ax, title=False)
    ax.set_title(group, fontsize=11)
save(fig, "groups.png")

L = pt.named("Kagomé")
dc = pt.DefectCatalogue(L)
conf = pt.build(L, dict(kind="dislocation", b=dc.dislocations[0]["b"], at=(0.37, 0.41)))
fig, ax = plt.subplots(figsize=(5.5, 5.5))
conf.plot(ax=ax, title="A dislocation in the Kagome lattice")
save(fig, "defect.png")
