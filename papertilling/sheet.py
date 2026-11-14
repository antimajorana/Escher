"""The named lattices: the hand-made sheet "Lattices of the 17 wallpaper groups" and three variants.

    from papertilling import named, named_lattices, gallery
    L = named("Shastry-Sutherland")          # a papertilling Lattice: L.plot(), L.parameters(), DefectCatalogue(L), ...
    named_lattices()                          # the table: name, group, sites, kinds of bond, network
    gallery()                                 # all of them drawn

Names are matched loosely: "Kagomé", "kagome", "Zig Zag", "zigzag", "Sutherland's Brick" all work.

Each lattice is stored as drawn on the sheet (on a neat grid). The group listed here is the MEASURED symmetry
of that drawing (three routes, lattice.check_family). For three of them it is larger than the label on the sheet:
Straight is p2 (sheet: p1), Shastry's Ladder is pmg (sheet: pm) and Brick is cmm (sheet: pg). The variants
"straight_p1", "shastrys_ladder_pm" and "brick_pg" are deformed versions that do have the labelled group.
"""
from __future__ import annotations

from .legacy.wallpaper_lattices import available_lattices, make_cell

# (name on the sheet, key, group written on the sheet), in the order of the sheet
SHEET = [
    ("Oblique", "oblique", "p2"), ("Elongated Triangular", "elongated_triangular", "cmm"), ("Straight", "straight", "p1"),
    ("Square", "square", "p4m"), ("Triangular", "triangular", "p6m"), ("Kagomé", "kagome", "p6m"), ("Honeycomb", "honeycomb", "p6m"),
    ("Rectangular", "rectangular", "pmm"), ("Fractal Triangular", "fractal_triangular", "p31m"),
    ("Breathing Kagomé", "breathing_kagome", "p3m1"), ("Shastry's Ladder", "shastrys_ladder", "pm"),
    ("Shastry-Sutherland", "shastry_sutherland", "p4g"), ("Brick", "brick", "pg"), ("Zig Zag", "zigzag", "pmg"),
    ("Sutherland's Brick", "sutherlands_brick", "pgg"), ("Chiral Square", "chiral_square", "p4"),
    ("Chiral Kagomé", "chiral_kagome", "p6"), ("Chiral Triangular", "chiral_triangular", "p3"),
]
# deformed versions that have the group written on the sheet: (name, key, group, the sheet lattice it comes from)
VARIANTS = [("Straight (p1 version)", "straight_p1", "p1", "straight"),
            ("Shastry's Ladder (pm version)", "shastrys_ladder_pm", "pm", "shastrys_ladder"),
            ("Brick (pg version)", "brick_pg", "pg", "brick")]
_ALIASES = {"zig_zag": "zigzag", "shastry_sutherlands": "shastry_sutherland", "snub_square": "shastry_sutherland"}


def _key(name: str) -> str:
    k = name.strip().lower()
    for a, b in ((" ", "_"), ("'", ""), ("’", ""), ("-", "_"), ("é", "e"), ("(", ""), (")", "")):
        k = k.replace(a, b)
    k = _ALIASES.get(k, k)
    if k not in available_lattices():
        raise KeyError(f"no lattice named {name!r}; the names are: " + ", ".join(n for n, _, _ in SHEET)
                       + "; variants: " + ", ".join(k for _, k, _, _ in VARIANTS))
    return k


def named(name: str, **params):
    """The lattice with this name, as a papertilling Lattice. Extra keyword arguments go to the stored recipe
    (for the lattices that have one, e.g. a second bond length)."""
    from .lattice import Lattice
    k = _key(name)
    L = Lattice.from_cell(make_cell(k, **params))
    sheet = {key: (nm, grp) for nm, key, grp in SHEET}
    var = {key: (nm, grp) for nm, key, grp, _ in VARIANTS}
    nm, grp = sheet.get(k) or var.get(k) or (k, None)
    L.meta.update(name=nm, key=k, sheet_group=grp)
    return L


def named_lattices(show: bool = True, variants: bool = True):
    """The table of the named lattices: list of dicts (name, key, sheet_group, group, kinds_of_site, sites,
    kinds_of_bond, bonds_per_site, faces, network_symmetry). `group` is the measured group of the stored drawing."""
    rows = []
    for nm, k, grp in SHEET + [(n, key, g) for n, key, g, _ in (VARIANTS if variants else [])]:
        L = named(k)
        f = L.family
        deg = sorted({sum(1 for b in f.bonds() if b[0] == i) + sum(1 for b in f.bonds() if b[1] == i) for i in range(f.m)})
        rows.append(dict(name=nm, key=k, sheet_group=grp, group=L.group, kinds_of_site=L.rank[0], sites=f.m,
                         kinds_of_bond=len(f.orbits), bonds_per_site=deg, faces=L.vertex_types(), network_symmetry=L.symmetry_index))
    if show:
        print(f"{'name':30s} {'key':22s} {'sheet':5s} {'group':5s} {'kinds of site':>13s} {'sites':>5s} {'kinds of bond':>13s}  "
              f"{'bonds/site':10s} faces round a site")
        for r in rows:
            flag = "" if r["sheet_group"] == r["group"] else "  <- the drawing has more symmetry than the sheet's label"
            print(f"{r['name']:30s} {r['key']:22s} {r['sheet_group'] or '':5s} {r['group']:5s} {r['kinds_of_site']:13d} {r['sites']:5d} "
                  f"{r['kinds_of_bond']:13d}  {str(r['bonds_per_site']):10s} {' | '.join(r['faces'])}{flag}")
    return rows


def gallery(ncols: int = 6, n: int = 3, size: float = 3.0, variants: bool = False):
    """All the named lattices drawn in the order of the sheet. Returns the matplotlib figure."""
    import math
    import matplotlib.pyplot as plt
    items = SHEET + [(nm, k, g) for nm, k, g, _ in (VARIANTS if variants else [])]
    rows = int(math.ceil(len(items) / ncols))
    fig, axs = plt.subplots(rows, ncols, figsize=(size * ncols, 1.1 * size * rows), squeeze=False)
    for ax, (nm, k, grp) in zip(axs.flat, items):
        L = named(k)
        L.plot(n=n, ax=ax, title=False)
        ax.set_title(f"{nm}\n({L.group}" + ("" if grp == L.group else f"; sheet: {grp}") + ")", fontsize=9)
    for ax in list(axs.flat)[len(items):]:
        ax.axis("off")
    fig.tight_layout()
    return fig
