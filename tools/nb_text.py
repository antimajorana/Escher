"""Cell text of the three notebooks (read by build_notebooks.py)."""

BOOT = '''# Setup: find the papertilling package (it sits in the folder above this notebook) and check what it needs.
import sys, pathlib, importlib.util
for _p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:
    if (_p / "papertilling" / "__init__.py").exists():
        sys.path.insert(0, str(_p)); break
_kernel = "uv pip install --python '" + sys.executable + "' "
if importlib.util.find_spec("papertilling") is None:
    raise ImportError("papertilling not found. Keep this notebook inside the papertilling/notebooks folder, or install the "
                      "package into this kernel:  " + _kernel + "-e <path to the papertilling folder>")
_missing = [m for m in ("numpy", "scipy", "networkx", "matplotlib", "spglib") if importlib.util.find_spec(m) is None]
if _missing:
    raise ImportError("this kernel lacks " + ", ".join(_missing) + ". Install with:  " + _kernel + " ".join(_missing)
                      + "   (or run  %pip install " + " ".join(_missing) + "  in a cell)")
import papertilling
print("papertilling", papertilling.__version__, "| Python", sys.version.split()[0], "| numba:", importlib.util.find_spec("numba") is not None,
      "| sliders:", importlib.util.find_spec("ipywidgets") is not None)'''

SETUP = '''import warnings; warnings.filterwarnings("ignore", category=DeprecationWarning)
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
from IPython.display import Markdown, display
from papertilling.lattice import load_catalogue, minimal_lattices, Lattice
from papertilling.groups import wyckoff_markdown
from papertilling.widgets import explore
from papertilling import named, named_lattices, gallery'''

# ------------------------------------------------------------------------------------------------
NB1 = [
("md", r'''
# 1. Lattices of a wallpaper group

For a chosen wallpaper group G, this notebook lists its **most Bravais-like lattices**, lets you pick one, and move its free parameters without leaving G.

**The rule: judge a lattice by its points alone.** A Bravais lattice is one point copied by a group of translations: every site is equivalent to every other, and nothing else is chosen. The same idea for a wallpaper group G:

1. **Points.** Take one point and copy it with all the operations of G (one *orbit*; the possible starting points are the Wyckoff positions of G). If one orbit is not enough, take two, then three.
2. **Exactly G, judged on the points.** The set of points, with no bonds drawn, must have the symmetry G and no more, with G's translations. Example: one orbit of p1 is a Bravais lattice, which always has a 2-fold rotation, so it is a lattice of p2 and not of p1.
3. **Simplest first.** Fewest kinds of site (orbits), then fewest sites per primitive cell.
4. **Bonds are not chosen.** Two sites are joined when their Voronoi cells share an edge (the Voronoi cell of a site is the region of the plane closer to it than to any other site). When four or more sites lie on one empty circle, which symmetry can force, they are left as one face (a square is not cut into two triangles). This network uses only distances, so it has exactly the symmetry of the points: bonds can neither add nor remove symmetry. It is always planar and connected, and every site has at least 3 bonds.
5. **One entry per network.** If a second set of points of G gives a network already listed, it is kept aside (`res.other[k]`).

**What to expect.**
* For p2, pmm, cmm, p4m and p6m the first entry is the Bravais lattice itself (one site per primitive cell). By rule 4 the oblique lattice is drawn as a triangular network (6 Voronoi neighbours), the rectangular and square ones with 4 bonds per site.
* p6m then gives the honeycomb, the kagome and the other classical uniform tilings, in order of sites per cell.
* For p1, pm, pg and cm no single orbit works: any single orbit of these groups has an extra 2-fold rotation. Their lists start with two or three kinds of site.
* The Voronoi network of a set of points can change when a parameter moves (four sites pass through a common circle). An entry shows the network met most often; the list says when there are others.

**Each list line also compares the network with the group:** "its own symmetry is this group", or "its own symmetry is N x larger" (the network, with the positions of its sites forgotten, has more symmetry than G; such lattices are kept in the lists).

**Status of the result: computational evidence.** Every choice of Wyckoff positions is tried, simplest first, at 48 random parameter values; a network met at fewer than 4 % of them is ignored, and one met at none is not found. The group of every listed lattice is checked by three routes (count of symmetry operations, the group finder of `wallpaper_lattices`, spglib). Two Wyckoff positions that give the same lattice (for p2: 1a, 1b, 1c, 1d) are listed once.

**Two other rules are available** (`RULE` below):
* `RULE = "sheet"`: the same condition on the points, but the bonds are chosen: among short bonds, the **fewest kinds of bond** that give a planar connected lattice with at least `MIN_DEGREE = 3` bonds at every site. Rank: kinds of site, kinds of bond, sites per cell, bonds per cell. This is the rule behind the hand-made sheet of named lattices (section 7): the Oblique lattice keeps 4 bonds per site, the Chiral Square is the first lattice of p4.
* `RULE = "bonds"`: as `"sheet"` but without the condition on the points, so the bonds may lower the symmetry (a lattice of p1 can then have 2 sites per cell).
'''),
("code", SETUP),
("md", r'''
## 1. Choose a group and list its lattices

`N` is the number of lattices to list. The stored lists hold the first 7 of every group and open instantly; with `N` above 7 the search runs now, going on to more kinds of site and more sites per cell until it has `N` (see the timings in `README.md`; it stops after 10 minutes and says so). Set `RECOMPUTE = True` to redo the search for one group (10 to 30 s for 7). With `RULE = "sheet"` or `"bonds"`, a `MIN_DEGREE` other than 3 searches again by itself (up to a minute).

Each entry: rank (kinds of site, kinds of bond, sites per cell, bonds per cell), the Wyckoff positions of the sites (ITA letter and site symmetry), the number of bonds per cell in each kind of bond, the free parameters; then its network (the face sizes round each kind of site, how the network's own symmetry compares with the group, whether other sets of points give the same network, and whether the Voronoi network changes with the parameters); then the one-line recipe that rebuilds it.
'''),
("code", '''GROUP = "p4g"          # any of p1 p2 pm pg cm pmm pmg pgg cmm p4 p4m p4g p3 p3m1 p31m p6 p6m
N = 7                  # how many lattices to list. Up to 7 is instant (stored); more than 7 searches now.
RULE = "points"        # "points": points alone + Voronoi bonds (default); "sheet": points alone + fewest kinds of bond;
                       # "bonds": fewest kinds of site and bond, bonds may lower the symmetry
MIN_DEGREE = 3         # only for RULE = "sheet" or "bonds": fewest bonds per site, 3 (default) or 2
RECOMPUTE = False
if RECOMPUTE or N > 7 or (RULE != "points" and MIN_DEGREE != 3):
    res = minimal_lattices(GROUP, n=N, rule=RULE, min_degree=MIN_DEGREE)
else:
    res = load_catalogue(rule=RULE)[GROUP].take(N)
res.show()'''),
("code", '''print(res.coverage)'''),
("md", r'''
## 2. The Wyckoff positions of the group

A **Wyckoff position** is a set of points with the same site symmetry (the operations of G that fix the point). Its **letter** is the one of the International Tables (ITA Vol. A), its **multiplicity** is the number of such points per conventional cell. A site on a rotation centre cannot move; a site on a mirror line can slide along it (one free parameter); a site in general position has two free parameters.

These tables were checked two independent ways: against the group's own operations, and against spglib's layer-group database (`tests/test_groups.py`).
'''),
("code", '''display(Markdown(wyckoff_markdown(GROUP)))'''),
("md", r'''
## 3. Pick a lattice

`PICK` is the number in the list above. The table marks which Wyckoff positions it uses.
'''),
("code", '''PICK = 1
L = res.lattices[PICK - 1]
print(L)
display(Markdown(wyckoff_markdown(GROUP, occupied=L.family)))
L.info()'''),
("code", '''fig, axs = plt.subplots(1, 2, figsize=(12, 6))
L.plot(n=3, ax=axs[0]); axs[0].set_title(f"{GROUP} #{PICK}")
L.family.plot(n=2, ax=axs[1], symmetry=True); axs[1].set_title("its symmetry elements")
plt.show()'''),
("md", r'''
## 4. Its free parameters

These are the coordinates G leaves free (a site on a mirror line slides along it; a site in general position moves in two directions) and the cell shape G leaves free (b, and gamma for oblique cells). Moving them **cannot lower** the symmetry below G. It can only **raise** it, on special values listed by `L.special()`: there the lattice gains symmetry and is no longer a lattice of G.
'''),
("code", '''L.parameters()
print("\\nspecial values (the group jumps up there):")
L.special();'''),
("md", r'''
### Move them with sliders

Each slider moves one free parameter, with the others held. Under each slider a bar shows its whole range:
* **hatched red = forbidden**: there the bonds would cross, a bond would pass through a site, two sites would coincide, or the lattice would become another tiling. The slider cannot be dragged into a forbidden part: its ends are the edges of the allowed stretch around the current value. The allowed stretch of one parameter depends on the others, so all the limits are recomputed after every move.
* **blue tick with a group name = a special value**, where the symmetry jumps up to that group (for instance, a value where a rectangle of sites becomes a square). A slider close to a special value snaps onto it, and the status line says which group the lattice has there.
* The cell parameters (`b`, `gamma`) are never forbidden (stretching or shearing the cell cannot make bonds cross); they only have special values.

The status line under the bars re-checks every move by the slow, independent route (group by two finders, planarity, bonds through sites, same tiling). For a lattice of the points-only rule a second line says whether the bonds are still the Voronoi-neighbour bonds of the points; if not, the drawing is still planar but the nearest-neighbour network of those points is another one.

`box.lattice()` gives the lattice at the slider values; `box.limits` and `box.special` hold the limits and the special values.
'''),
("code", '''box = explore(L, n=2)'''),
("code", '''# the same by hand: a new Lattice; a warning is printed if the group, planarity or the tiling changes
name = L.family.names[-1] if L.family.names else None
if name:
    L2 = L.set(**{name: L.values[name] + 0.05})
    print(name, "->", L2.values[name], "| group still", L2.status["group"], "| same tiling:", L2.status["same_key"])'''),
("md", r'''
## 5. Hand it to Notebooks 2 and 3

The line below rebuilds this lattice anywhere `papertilling` is imported. Paste it into the `LATTICE` cell of Notebooks 2 and 3.
'''),
("code", '''L_now = box.lattice() if box is not None else L
print(L_now.code())'''),
("code", '''# grow a finite piece of any size
from papertilling.legacy.wallpaper_lattices import WallpaperLattice
piece = WallpaperLattice(L_now.cell(), shape="disk", radius=6)
piece.plot(title=False); plt.show()
print(len(piece.positions), "sites")'''),
("md", r'''
## 6. All 17 groups at a glance

The most Bravais-like lattice of every group (its rank, Wyckoff positions, and the face sizes round each kind of site).
'''),
("code", '''cat = load_catalogue()
for g, r in cat.items():
    e = r.lattices[0]
    print(f"{g:5s} rank {str(e.rank):15s} sites {'; '.join(e.wyckoff()):26s} faces round a site: {' | '.join(e.vertex_types())}")'''),
("md", r"""
## 7. The named lattices

Eighteen well-known lattices (Square, Kagomé, Shastry-Sutherland, ...) are stored by name, drawn on neat grids. `named("Kagomé")` gives one as a lattice of this package, so everything above and Notebooks 2 and 3 work on it; names are matched loosely (`"zig zag"`, `"kagome"`).

The table gives, for each: the group written on the original sheet, the **measured** group of the drawing, the kinds of site and of bond, and the faces round a site. Three drawings have more symmetry than their label (Straight, Shastry's Ladder, Brick); the last three rows are deformed versions that do have the labelled group.
"""),
("code", """rows = named_lattices()"""),
("code", """fig = gallery(); plt.show()"""),
("code", """K = named("Shastry-Sutherland")
print(K)
print(K.code())          # paste this line into the LATTICE cell of Notebooks 2 and 3, or write L = named("Shastry-Sutherland") there"""),
]

# ------------------------------------------------------------------------------------------------
LATTICE = '''# LATTICE: paste a line printed by Notebook 1 (L.code()), or take entry PICK of a group's list.
from papertilling.lattice import load_catalogue, Lattice
L = load_catalogue()["p4g"].lattices[0]        # p4g #1 (points-only rule: the Shastry-Sutherland network)
# L = Lattice.from_recipe(dict(group="p4m", sites=[(0, 0)], bonds=[((0, 0), (1, 0))], a=1))   # square lattice
# from papertilling import named; L = named("Kagomé")                                        # a named lattice
print(L)'''

NB2 = [
("md", r'''
# 2. The defects a lattice can carry

For one lattice from Notebook 1, this notebook lists every **crystalline-symmetry-breaking defect** it can carry and draws what each one looks like on the lattice.

**Where defects come from.** Cut the crystal along a line, and glue the two sides back after moving one of them by a symmetry operation g of the crystal. Far from the end of the cut nothing looks wrong, because g maps the crystal onto itself. Near the end, something must give: that is the defect. Walking once round it, a chain of bonds does not close: it is off by g, the **holonomy** of the defect.
* g a **translation** by **b**: a **dislocation** with Burgers vector **b**;
* g a **rotation** by alpha about a rotation centre: a **disclination** of Frank angle alpha (alpha > 0: a wedge removed; alpha < 0: a wedge inserted);
* g a **reflection or glide**: the sheet comes back mirrored. A point defect of this kind cannot exist on an ordinary (orientable) sheet: it needs a Moebius seam. Reflections give instead **walls** (lines).

**Which defects are the same (the homotopy classification).** Two point defects are the same type when one can be turned into the other by moving it and deforming the crystal far away. The holonomy then changes only by conjugation, g -> h g h^-1 with h a rotation or translation of G (reflections are excluded on an orientable sheet), and the rotation angle is kept as a real number (+180 and -180 are different: one removes half the plane, the other inserts it). So:
* dislocations: **b** and R **b** are the same type when R is a rotation of G;
* disclinations: same angle, and centres mapped onto each other by rotations and translations of G. A G-orbit of centres can split into two such classes (p4g: 2a(i) and 2a(ii)): they are mirror images, so their cores look identical, but they are different types.

**Walls and the other mirror- and glide-breaking defects** (four kinds):
* **(a) domain walls**: between the lattice and a copy moved by an isometry sigma that is not in G but maps G's pattern of symmetry elements onto itself (sigma in the Euclidean normaliser N(G)). The copy has the same group, but is a different domain: a mirror image (chirality domain), a rotated copy, or a shifted copy (antiphase domain). Their number is [N(G):G].
* **(b) twin walls**: the lattice on one side, its mirror image across the wall line on the other. On a mirror line of the lattice the wall is invisible.
* **(c) Moebius seams**: one per class of mirror or glide line; the sheet becomes non-orientable.
* **(d) glide-breaking walls**: on a glide line, the pure reflection instead of the glide: the two sides are off by the glide translation along the wall (a stacking fault).

**Status.** Classes: proved by a one-line argument from the definitions (formal verification not performed). Lists and pictures: computational evidence. For p1, pm, pg and cm the normaliser is continuous: the list (a) is a set of representatives, marked incomplete.
'''),
("code", '''import warnings; warnings.filterwarnings("ignore", category=DeprecationWarning)
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
from papertilling.defects.catalogue import DefectCatalogue'''),
("code", LATTICE),
("code", '''fig, ax = plt.subplots(figsize=(6, 6)); L.plot(n=3, ax=ax); plt.show()
cat = DefectCatalogue(L, bmax=2.0)        # Burgers vectors up to 2 x the shortest
cat.summary();'''),
("md", r'''
## 1. Dislocations: the shortest Burgers vector in every direction

For each lattice direction (a primitive lattice vector), the shortest Burgers vector along it, its length, its class (the rotations of G that map it to others), whether **b** and **-b** are the same type, and its mirror partner. Each picture is one dislocation, built by cut and reglue (the cut ends at a generic point of the cell).
'''),
("code", '''cat.show("dislocations"); plt.show()'''),
("md", r'''
## 2. Disclinations between -180 and +180 degrees

At every class of rotation centre (labelled by its ITA Wyckoff position; (i)/(ii) when a G-orbit splits), every allowed angle k x 360/n. The picture shows where the centre sits: on a site (the site loses or gains bonds), in a face (the ring round it shrinks or grows), or on a bond.
'''),
("code", '''cat.show("disclinations"); plt.show()'''),
("md", r'''
## 3. Mirror and glide lines, and Moebius seams (c)
'''),
("code", '''for m in cat.mirrors:
    print(m["id"], m["kind"], "direction", np.round(m["direction"], 3), "glide", round(m["glide"], 4), "through a site:", m["through_site"])
if cat.mobius_seams:
    cat.show("mobius_seams"); plt.show()
else:
    print("no mirror or glide: no Moebius seam (a chiral lattice)")'''),
("md", r'''
## 4. Domain walls (a)

The other domains with the same group: `index` = [N(G):G] counts them (with the lattice itself). Each wall here is a straight line with the lattice on one side and the moved copy on the other; bonds across the wall are cut (the bonding across a wall is not fixed by symmetry).
'''),
("code", '''print(cat.normaliser["note"] if "note" in cat.normaliser else cat.normaliser)
if cat.domain_walls:
    cat.show("domain_walls"); plt.show()'''),
("md", r'''
## 5. Twin walls (b) and glide-breaking walls (d)
'''),
("code", '''if cat.twin_walls:
    cat.show("twin_walls"); plt.show()
if cat.glide_breaking:
    cat.show("glide_breaking"); plt.show()'''),
("md", r'''
## 6. One defect up close

Every entry has `build(R=...)` (a larger patch with `R`) and `plot(ax)`. Its `spec` is what Notebook 3 takes.
'''),
("code", '''e = cat.disclinations[0]
print(e["id"], e["spec"])
fig, ax = plt.subplots(figsize=(7, 7)); e.plot(ax=ax, R=7); plt.show()'''),
]

# ------------------------------------------------------------------------------------------------
NB3 = [
("md", r'''
# 3. Several defects close together, and which configurations are equivalent

Put any list of dislocations and disclinations on the lattice of Notebook 1, as close as you like, look at the lattice, and decide which configurations are the same.

**Equivalence: total charge + core graph.**
1. **Total charge.** Walk once round all the defects at once. The holonomy of that big loop is the product of the individual holonomies; its class (as in Notebook 2) is the total charge. Example: on the square lattice, +90 at a site c1 and -90 at the neighbouring face centre c2 compose to a pure translation: (R90, (I-R90)c1)(R-90, (I-R-90)c2) = (I, (R90 - I)c2) (c1 = 0), so b = (I - R90)c2 = (1, 0) (up to the class): together they are **one unit dislocation**.
2. **Core graph.** The sites whose neighbourhood (out to a few bonds) differs from the perfect lattice, with the bonds between them: compared as graphs (isomorphism), not as pictures. In the example, the +90 site has 3 bonds and the -90 face is a pentagon: exactly the core of the unit dislocation. So the pair **is** a dislocation.
Two configurations are equivalent when both agree. `equivalent_single(config)` finds the single defect with the same total charge and says whether the cores match.

**Status.** Total charge: read from the built graph (computational evidence; the algebra is proved). Core comparison: exact graph isomorphism of the core balls. "No match" for a dislocation means none among the cores found by scanning cut positions (`grid`), not a proof.
'''),
("code", '''import warnings; warnings.filterwarnings("ignore", category=DeprecationWarning)
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
from papertilling.defects.catalogue import DefectCatalogue
from papertilling.defects.build import build
from papertilling.defects.config import equivalent_single, compare, classify'''),
("md", r'''
## 0. Control: a known example on the square lattice

+90 at the site (0, 0) and -90 at the face centre (1/2, 1/2) of the square lattice.
'''),
("code", '''sq = "square"
pair = build(sq, dict(kind="disclination", angle=90, at=(0, 0)), dict(kind="disclination", angle=-90, at=(0.5, 0.5)), picture="raw")
pair.inspect()
single = build(sq, dict(kind="dislocation", b=(1, 0), at=(0.371, 0.413)), picture="raw")
print({k: v for k, v in compare(pair, single).items() if k != "details"})
fig, axs = plt.subplots(1, 2, figsize=(13, 6.5))
pair.plot(ax=axs[0], title="+90 site, -90 face"); single.plot(ax=axs[1], title="unit dislocation")
plt.show()'''),
("md", r'''
## 1. Your lattice and its defect classes
'''),
("code", LATTICE),
("code", '''cat = DefectCatalogue(L)
for c in cat.centre_classes:
    print(f"{c['label']:8s} order {c['order']}  point {np.round(c['point'], 4).tolist()}  {c['location']}  angles {c['angles']}")
for t in cat.dislocations:
    print(t["id"], "b =", np.round(t["b"], 4).tolist(), "|b| =", round(t["length"], 4))'''),
("md", r'''
## 2. Build configurations

A defect is a dict:
* `dict(kind="disclination", angle=90, at=(x, y))`: `at` must be a rotation centre that allows the angle (or add `snap=True` to `build` to move it to the nearest one);
* `dict(kind="dislocation", b=(bx, by), at=(x, y))`: `b` a lattice vector, `at` the end of the cut (any point; the core changes as it moves inside a cell).

Points are Cartesian, in the frame of the plot. `centre(label, shift)` below gives the centre of a class moved by a lattice vector `shift` (in cells).
'''),
("code", '''A = cat.cell.A
def centre(label, shift=(0, 0)):
    c = next(c for c in cat.centre_classes if c["label"] == label)
    return c["point"] + A @ np.array(shift, float)

top = max(c["order"] for c in cat.centre_classes)
alpha = 360 / top
labels = [c["label"] for c in cat.centre_classes if c["order"] == top]
print("highest-order centres:", labels, "angle", alpha)

configs = {
    f"+{alpha:g} {labels[0]} / -{alpha:g} {labels[0]} next cell": [dict(kind="disclination", angle=alpha, at=centre(labels[0])),
                                                                  dict(kind="disclination", angle=-alpha, at=centre(labels[0], (1, 0)))],
    f"+{alpha:g} {labels[0]} / -{alpha:g} {labels[-1]}": [dict(kind="disclination", angle=alpha, at=centre(labels[0])),
                                                          dict(kind="disclination", angle=-alpha, at=centre(labels[-1]))],
    "shortest dislocation": [dict(kind="dislocation", b=cat.dislocations[0]["b"], at=A @ np.array([0.371, 0.413]))],
}
built = {name: build(L, *specs, picture="raw") for name, specs in configs.items()}
fig, axs = plt.subplots(1, len(built), figsize=(6.5 * len(built), 6.5))
for ax, (name, c) in zip(np.atleast_1d(axs), built.items()):
    c.plot(ax=ax, title=name)
plt.show()'''),
("code", '''for name, c in built.items():
    print("==", name); c.inspect()'''),
("md", r'''
## 3. Which are equivalent?

`classify` groups the configurations by total charge and core graph. `equivalent_single` compares one configuration with the single defect of the same total charge (for a dislocation it scans cut positions: `grid`).
'''),
("code", '''labels_eq = classify(list(built.values()), names=list(built.keys()))'''),
("md", r'''
**Reading the two answers.** One total charge (one dislocation class) can have **several cores**: the same Burgers vector can sit on different rings, depending on where the cut ends inside the cell and on which bonds close the core. So two configurations can have the same charge but different cores (different classes in the table above), and still **each** be equivalent to *a* single dislocation of that charge: `equivalent_single` says which of the single dislocation's cores it matches. A "Volterra core" comes from cut and reglue alone; a "completion core" also has bonds added between under-coordinated core sites, without crossings and without exceeding the perfect coordination. Whether completion cores count as the same defect is your call; the verdict says which kind matched.
'''),
("code", '''for name, c in built.items():
    r = equivalent_single(c, grid=2)
    print(f"{name:40s} -> {r['verdict']}")'''),
("md", r'''
## 4. Your own configuration

Edit the list and run. `picture="relaxed"` relaxes the drawing (slower; the graph is the same).
'''),
("code", '''mine = build(L,
             dict(kind="disclination", angle=alpha, at=centre(labels[0])),
             dict(kind="disclination", angle=-alpha, at=centre(labels[-1], (1, 1))),
             picture="relaxed")
mine.inspect()
mine.plot(); plt.show()'''),
]
