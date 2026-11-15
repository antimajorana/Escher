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
("md", r"""
## 8. The lattice on other surfaces

**Which surfaces can carry a lattice without any defect?** Exactly the **flat** ones: the plane, the cylinder, the torus, the Moebius band and the Klein bottle. Each is the plane with some points glued together, and the gluing must be a symmetry of the lattice, so that every site keeps exactly the neighbourhood it had in the plane: same bonds, same lengths, same angles.
* **cylinder**: points a lattice vector **C** apart are glued (like the (n, m) of a nanotube); open at both ends;
* **torus**: points **T1** apart and points **T2** apart are glued (a finite crystal with periodic boundary conditions);
* **Moebius band** and **Klein bottle**: the gluing is a **glide** (reflect across a line, then slide along it), so the sheet comes back mirrored. That needs a mirror or glide line in the group: impossible for p1, p2, p3, p4, p6 (chiral lattices).

**Every other closed surface needs disclinations.** Walking round a disclination of Frank angle alpha turns a direction by 360 - alpha degrees instead of 360. The theorem of Gauss and Bonnet says that on a closed surface the Frank angles always add up to **360 deg x chi**, where chi = V - E + F (sites minus bonds plus faces) is the Euler characteristic:

| surface | chi | Frank angles must add up to |
|---|---|---|
| torus, Klein bottle | 0 | 0 (no defect needed) |
| sphere | 2 | +720 deg (for example 12 x 60 deg: the 12 pentagons of a football) |
| projective plane | 1 | +360 deg |
| surface with g holes (genus g) | 2 - 2g | -720 deg x (g - 1) |

Each picture below has two parts: the **flat piece** (the region of the plane that is glued up; edges with the same arrow are glued) and a **3D shape**. The flat piece is exact. The 3D shape is exact for the cylinder, but a flat torus has no undistorted shape in 3D (the doughnut stretches the bonds outside and squeezes them inside), and a Klein bottle must pass through itself in 3D.

`S.checks()` verifies each surface: no bond from a site to itself, no doubled bond, every site with its plane number of bonds (away from an open edge), and V - E + F = chi. Sizes are integer vectors in units of the primitive cell vectors of `L.cell()`.
"""),
("md", r"""
**First, which surfaces this lattice can have.** `surface_table(SURF)` tries all eight and says, for each:
* **OK**: built, and every check passed;
* **IMPOSSIBLE**: proved from the lattice's symmetry, for every lattice of its group: no mirror or glide line means no Moebius band, Klein bottle or projective plane (those need a gluing that reverses orientation, and it must be a symmetry); no rotation centre (p1, pm, pg, cm) means no sphere or projective plane (a disclination must be a rotation of the lattice, so without rotations every Frank angle is 0 or negative, while Gauss-Bonnet needs +720 or +360 deg);
* **NOT FOUND**: this construction failed at every size and centre tried (for example, a corner would fold a bond onto itself). That is not a proof that the surface is impossible; the line says what was tried.
"""),
("code", """from papertilling.surfaces import surface_table
SURF = L_now                         # the lattice chosen above (or e.g. named("Kagomé"))
table = surface_table(SURF)          # about 10-60 s: sizes are retried for NOT FOUND"""),
("code", """from papertilling.surfaces import torus, cylinder, mobius_band, klein_bottle, glides, sphere, projective_plane, genus_surface
SURF = L_now                        # the lattice chosen above (or e.g. named("Kagomé"))

for make in (torus, cylinder):      # sizes: torus(SURF, T1, T2), cylinder(SURF, wrap, length); default about 220 sites
    S = make(SURF)
    S.show(); plt.show()
    S.checks()"""),
("code", """if glides(SURF):
    for make in (mobius_band, klein_bottle):
        S = make(SURF)
        S.show(); plt.show()
        S.checks()
else:
    print(f"{SURF.group} has no mirror or glide line: no Moebius band and no Klein bottle for this lattice.")"""),
("md", r"""
### The sphere: the lattice folded onto a polyhedron

The lattice is cut into congruent flat pieces whose corners sit at its own rotation centres of the highest order n, and the pieces are folded onto a polyhedron: an **icosahedron** for n = 6 (12 corners, each a disclination of +60 deg), a **cube** for n = 4 (8 x +90), an **octahedron** for n = 3 (6 x +120), a **pillow** for n = 2 (4 x +180). These are the smallest disclinations the lattice allows, and they add up to +720 deg, as Gauss-Bonnet requires. Away from the 12 (8, 6, 4) corners, every site has exactly its plane neighbourhood.

* `size=(h, k)`: the edge of a piece is the lattice vector h b1 + k b2 between two centres (like the (h, k) of fullerenes and virus capsids). Honeycomb (1, 1) is C60: 60 sites, 12 pentagons, 20 hexagons.
* `centre=`: which class of rotation centres carries the corners (`"1a"`, `"site"`, `"face centre"`). By default a face centre, so that no site loses a bond (honeycomb: the disclinations are pentagons). On the square lattice, `centre="site"` gives the cube (8 three-bond sites); the default, the plaquette centres, gives 8 triangles instead.
* Every gluing between pieces is tested: it must be a symmetry of the lattice, otherwise the construction is refused (a p3 lattice cannot fold onto an icosahedron, because a 60 deg rotation is not one of its symmetries; it folds onto an octahedron). A corner that would fold a bond onto itself, or shrink a face below 3 sides, is refused too. p1, pm, pg and cm have no rotation centre: no sphere.

The 3D shape is the polyhedron pushed onto a sphere, so bonds near the corners look shorter; the flat net on the left is exact.
"""),
("code", """try:
    S = sphere(SURF)                 # e.g. sphere(SURF, size=(1, 1)), sphere(SURF, centre="site")
    S.show(); plt.show()
    S.checks()
    kinds = sorted({(c["label"], c["location"], c["angle"]) for c in S.cones})
    print(len(S.cones), "disclinations:", "; ".join(f"{a:+g} deg at {lab} ({loc})" for lab, loc, a in kinds))
except ValueError as err:
    print("no sphere for this lattice:", err)"""),
("md", r"""
### The projective plane

A closed surface with chi = 1: a sphere with every point glued to the opposite point. Here: half of the polyhedron (a hemi-icosahedron, hemi-cube, hemi-octahedron or hemi-pillow), with opposite edges glued after a **reflection**. So it carries half as many disclinations (+360 deg in total), and it needs a reflection or glide among the lattice's symmetries that matches the gluing: refused for chiral lattices (p2, p3, p4, p6) and for groups without rotations. Honeycomb (1, 1) gives 30 sites, 6 pentagons and 10 hexagons. The 3D picture shows the double cover (a sphere): the two sites with the same colour are the same site of the projective plane.
"""),
("code", """try:
    S = projective_plane(SURF)       # default size (1, 1); try (1, 0) or (2, 0) if refused
    S.show(); plt.show()
    S.checks()
except ValueError as err:
    print("no projective plane at this size:", err)"""),
("md", r"""
### Surfaces with g holes (genus g)

They need negative disclinations adding up to -720 deg x (g - 1). The construction used here works for every lattice, including p1: take two copies (sheets) of a torus of the lattice, cut both along g - 1 slits, and glue each side of a slit to the opposite side on the other sheet. Going once round an end of a slit takes you to the other sheet; going round twice brings you back, after 720 deg instead of 360 deg. So each slit end is a disclination of **-360 deg**, sitting in a face, which becomes one face with twice as many sides. Every site keeps its plane neighbourhood. There are 2(g - 1) slit ends, and 2(g - 1) x (-360) = -720 (g - 1), as required.

The picture is the two sheets side by side with the slits (red), their ends (stars) and the bonds that change sheet; a genus-g surface with this geometry has no simple 3D shape.
"""),
("code", """for g in (2, 3):
    try:
        S = genus_surface(SURF, g=g)             # torus_size=(6, 6) cells by default; enlarge it for more holes
        S.plot_net(); plt.show()
        S.checks()
    except ValueError as err:
        print(f"genus {g}: refused:", err)"""),
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

**The cut-and-glue recipe for each kind.** Every defect here is made by the same Volterra procedure: cut the crystal, move one side by an isometry sigma, glue it back. What differs is the shape of the cut and the kind of sigma:

| cut | sigma | defect |
|---|---|---|
| half-line ending at a point | translation by a lattice vector **b** | dislocation (point) |
| half-line ending at a rotation centre | rotation by alpha about that centre | disclination (point) |
| half-line ending at a point | reflection or glide | impossible on an ordinary sheet: going once round the end would bring you back mirrored, and a small loop on a sheet cannot reverse orientation |
| whole line, glued back to itself across the cut | reflection or glide of the lattice | (c) Moebius seam: the sheet becomes non-orientable |
| whole line, one side replaced by its mirror image across the line | reflection that is not a symmetry | (b) twin wall (invisible on a mirror line of the lattice) |
| whole line on a glide line, one side reflected | the bare reflection instead of the glide | (d) glide-breaking wall: the sides are off by the glide translation along the wall |
| whole line, one side replaced by a moved copy | an isometry that keeps the pattern of symmetry elements but is not a symmetry (normaliser) | (a) domain wall: chirality, rotation or antiphase domains |

**Can a wall end inside the crystal?** Walk once round the would-be end: you cross the wall once, so you come back moved by the mismatch sigma between the two sides. But sigma is only defined up to the lattice's own symmetries g: side B is sigma(lattice), and that equals sigma g (lattice) for every g in G. The end must be a point defect carrying one of these sigma g, and on an ordinary sheet a point defect can only carry an orientation-preserving one (a translation or a rotation). So: **a wall can end exactly when some sigma g preserves orientation.**
* **Antiphase wall** (sigma a translation): it can end, at a partial dislocation with that Burgers vector.
* **Rotation domain wall** (sigma a rotation): it can end, at a partial disclination.
* **Glide-breaking wall**: sigma = the bare reflection, and the lattice has the glide g on that line; sigma g^-1 is a translation by the glide's slide (half a period along the line). It can end, at a partial dislocation whose Burgers vector is that slide.
* **Twin wall** (sigma a reflection across the wall): if the lattice has any mirror or glide line m, then sigma m is a rotation (two reflections make a rotation), so the twin wall is really a **tilt boundary** and can end at a partial disclination whose angle is twice the angle between the wall and that mirror line. Only on a **chiral** lattice (p1, p2, p3, p4, p6), where every sigma g reverses orientation, can a twin wall not end.
* **Chirality domain wall** (a mirror image of a chiral lattice): every sigma g reverses orientation, so it cannot end. It runs to the edge of the crystal, closes into a loop, or meets other walls.

(Argument from the holonomy: proved result; formal verification not performed. These endings are not computed by the package.)

**Status.** Classes: proved by a one-line argument from the definitions (formal verification not performed). Lists and pictures: computational evidence. For p1, pm, pg and cm the normaliser is continuous: the list (a) is a set of representatives, marked incomplete.

**Pictures.** Each defect is first built with the sites in their rigid cut-and-reglue positions. Where that would make bonds cross or overlap (near a wall line, or at some dislocation cores), the picture is redrawn: the bonds relax like springs, locally around the bad region first, and only if needed over the whole patch. The bonds themselves are never changed. A Moebius seam is the exception: a non-orientable seam cannot lie flat, so the bonds across it are drawn as dotted stubs.
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

**How the crystals are drawn.** `build` first places the sites by cutting and regluing the plane once per defect (the "raw" picture, `picture="raw"`). With two or more disclinations the raw picture is not a faithful drawing: each disclination stretches the angles around its own centre, and for a +alpha/-alpha pair the two stretches do not cancel, so the far field covers more than 360 deg and folds over itself along the cut (for +-90 deg it covers about 384 deg). The graph (who is bonded to whom) is right; only the positions are wrong. The default `picture="relaxed"` therefore lets the bonds relax like springs, and if the result still has crossing bonds or folded faces it redraws the crystal: first only around the bad region, and if needed the whole crystal with its rim pinned to a circle and each site placed at the average of its neighbours (by Tutte's theorem that drawing has no crossing bonds when the crystal's graph is 3-connected, which most crystals are). If no stage gives a clean picture, the least bad one is kept and the plot title says how many crossings or overlaps remain. `c.defect.info["drawing"]` says which stage was used.

A configuration whose Frank angles do not add up to 0 cannot have an undistorted far field in the plane: a net +alpha makes the plane a cone, and the drawing must squeeze it (or, for a net negative angle, crowd it). Only net-zero configurations look like the perfect lattice far away.
'''),
("code", '''sq = "square"
pair = build(sq, dict(kind="disclination", angle=90, at=(0, 0)), dict(kind="disclination", angle=-90, at=(0.5, 0.5)))
pair.inspect()
single = build(sq, dict(kind="dislocation", b=(1, 0), at=(0.371, 0.413)))
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
built = {name: build(L, *specs) for name, specs in configs.items()}
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
### See the cores

`equivalent_single(c, show=True)` draws the configuration's own core (first panel) next to every core of the single dislocation with the same charge: **Volterra cores** (cut and reglue only; the orange ring is the face that is not in the perfect lattice, red circles mark sites that lost a bond) in one row, and, when they exist, **completion cores** (a Volterra core healed by extra bonds, drawn as dashed magenta lines) in the next. The core that matches the configuration is framed in green. When the match is a completion core, the Volterra cores are always shown too, so you can see what was healed; at most 5 completion cores are drawn when there is a match (`r["gallery"]` holds them all).

The list of cores comes from a finite scan of cut positions (`grid`): a larger grid can find more cores, and the numbering ("completion core 7") belongs to that scan. The ring sizes in a title belong to the build that first reached that core; the same core can be reached with an extra perfect-size ring.
'''),
("code", '''pair = built[f"+{alpha:g} {labels[0]} / -{alpha:g} {labels[-1]}"]
r = equivalent_single(pair, grid=2, show=True)'''),
("md", r'''
## 4. Your own configuration

Edit the list and run. `picture="relaxed"` (the default) gives a faithful drawing; `picture="raw"` the unrelaxed cut-and-reglue positions (faster; with several disclinations they overlap along the cuts). The graph is the same in both.
'''),
("code", '''mine = build(L,
             dict(kind="disclination", angle=alpha, at=centre(labels[0])),
             dict(kind="disclination", angle=-alpha, at=centre(labels[-1], (1, 1))),
             picture="relaxed")
mine.inspect()
mine.plot(); plt.show()'''),
]
