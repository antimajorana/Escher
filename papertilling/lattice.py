"""Lattice: one planar periodic lattice of a wallpaper group G, as a family with free parameters, its
D-symbol key (rule B) and its rank (option b). minimal_lattices(group): the first n by rank.

    from papertilling.lattice import minimal_lattices, load_catalogue, Lattice
    res = load_catalogue()["p4g"]          # instant: built by tools/build_catalogue.py
    res = minimal_lattices("p4g")          # or search now (seconds; see res.coverage)
    res.show()
    L = res.lattices[0]; L.parameters(); L2 = L.set(t1=0.3); L2.plot(n=3); L.special(); L.code()

Labels: the definitions are the author's (docs/DEVELOPMENT.md); planarity is found by sampling (computational
evidence); group by three routes on final entries (operation count, legacy identify_wallpaper_group,
spglib layer route): computational evidence.
"""
from __future__ import annotations

import json
import math
import os
import time
import warnings

import numpy as np

from .dsymbol import dsymbol as _dsymbol, DSymbolError, network_key, symmetry_index
from .legacy import minimal_lattices as ML
from .legacy.lattice_families import Family, has_crossings, clearance, min_separation, spg_group, degrees_of
from .legacy.wallpaper_lattices import periodic_connected, _symmetry_ops

DATA = os.path.join(os.path.dirname(__file__), "data", "minimal_b.json")
DATA_POINTS = os.path.join(os.path.dirname(__file__), "data", "points.json")
DATA_SHEET = os.path.join(os.path.dirname(__file__), "data", "sheet.json")


class LatticeWarning(UserWarning):
    """Raised (as a warning) when moving a parameter changes the group, planarity or the tiling."""


def ops_in_cell(fam):
    """G's operations (Mf, t) in the fractional basis of fam.cell(): Family._ops are in G's primitive
    basis with the origin of G; fam.cell() is drawn in the frame (Q, o0), so its fractional coordinates
    are x' = F + s with s = (Q B)^-1 o0, and x -> M x + t becomes x' -> M x' + t + (I - M) s."""
    Q, o0 = fam.frame
    B = Q @ fam._Ac() @ fam._U
    s = np.linalg.solve(B, o0)
    return [(M, np.asarray(t, float) + (np.eye(2) - M) @ s) for M, t in fam._ops]


def family_key(fam, **values):
    c = fam.cell(**values)
    return _dsymbol(c, ops_in_cell(fam))["key"]


def family_rank(fam):
    return (len(fam.gens), len(fam.orbits), fam.m, len(fam.bonds()))


def _key_tuple(k):
    return None if k is None else tuple(tuple(int(x) for x in row) for row in k)


def check_family(fam, key=None, **values):
    """Status of the lattice at the given values: group (legacy route), spglib, exact?, planar?,
    bond through a site?, key and whether it equals `key`."""
    f = fam.set(**values) if values else fam
    c = f.cell()
    out = {}
    sep = min_separation(c)
    out["sites_coincide"] = sep < 1e-6
    try:
        g, mp = f.group_at() if not out["sites_coincide"] else (None, None)
    except ValueError as err:       # the legacy group finder can fail within about 1e-6 of a special value
        g, mp = None, None
        out["group_error"] = str(err)
    out["group"], out["primitive_sites"] = g, mp
    out["exact"] = (g == f.group and mp == f.m)
    out["spglib"] = spg_group(c)
    out["planar"] = not has_crossings(c)
    out["clearance"] = clearance(c) if not out["sites_coincide"] else 0.0
    out["bond_through_site"] = out["clearance"] < 1e-6
    try:
        out["key"] = family_key(f)
    except (DSymbolError, RuntimeError) as err:
        out["key"] = None; out["key_error"] = str(err)
    out["same_key"] = None if key is None else (out["key"] == key)
    return out


class Lattice:
    """A lattice of G: Family (network + free parameters) + D-symbol key + rank + planar values."""

    def __init__(self, family, key=None, rank=None, planar_values=None, meta=None):
        self.family = family
        self.group = family.group
        if key is None:
            try:
                key = family_key(family)
            except (DSymbolError, RuntimeError) as err:
                warnings.warn(f"no D-symbol at these values ({err}): not a planar tiling here", LatticeWarning, stacklevel=2)
        self.key = _key_tuple(key)
        self.rank = tuple(rank) if rank is not None else family_rank(family)
        self.planar_values = list(planar_values or [family.values])
        self.meta = dict(meta or {})

    # --- construction -----------------------------------------------------------------------
    @classmethod
    def from_recipe(cls, recipe, key=None, **kw):
        """From dict(group, sites, bonds, a, b, gamma) (a Family recipe)."""
        fam = Family.from_recipe(recipe)
        return cls(fam, key=key, **kw)

    @classmethod
    def from_cell(cls, cell, group=None):
        """A catalogue lattice (name or UnitCell) as a Lattice of its own group (or of `group`)."""
        return cls(Family.from_cell(cell, group=group))

    # --- parameters ---------------------------------------------------------------------------
    @property
    def values(self):
        return self.family.values

    def parameters(self, show=True):
        """Free parameters: list of (name, value, meaning); moving them keeps G (except on special loci)."""
        return self.family.parameters(show=show)

    def set(self, **params):
        """A new Lattice with some parameters changed. Re-checks the group, planarity, bonds through
        sites and the D-symbol key, and WARNS (LatticeWarning) if any of them changed. The new object's
        .key is the key at the new values (None if there is no D-symbol); the old key is meta['parent_key'];
        .status holds every check."""
        fam = self.family.set(**params)
        ref = self.key if self.key is not None else _key_tuple(self.meta.get("parent_key"))   # the tiling we started from
        st = check_family(fam, key=ref)
        msgs = []
        if st["sites_coincide"]:
            msgs.append("two sites coincide: not a lattice of this network here")
        elif not st["exact"]:
            extra = "" if st["group"] != self.group else " with more translations"
            msgs.append(f"now {st['group']}{extra}: you are on a special locus (group larger than {self.group})")
        if not st["planar"]:
            msgs.append("bonds now cross")
        if st["bond_through_site"]:
            msgs.append("a bond now passes through a site")
        if st["key"] is not None and st["same_key"] is False and st["planar"] and not st["bond_through_site"]:
            msgs.append("different tiling: D-symbol key changed (another rule-B lattice)")
        for m in msgs:
            warnings.warn(m, LatticeWarning, stacklevel=2)
        new = Lattice(fam, key=st["key"] if st["key"] is not None else ref, rank=self.rank,
                      planar_values=self.planar_values,
                      meta=dict(self.meta, parent_key=None if ref is None else [list(r) for r in ref]))
        if st["key"] is None:
            new.key = None                                  # no D-symbol here (not a planar tiling)
        new.status = st
        return new

    def special(self, show=True):
        """Supergroup loci of the family (legacy Family.loci): where the group jumps up."""
        out = []
        for L in self.family.supergroup_loci():
            out.append(dict(equations=L.equations(), group=L.group, spglib=L.spglib, sites=L.sites,
                            free_left=L.dim, example=L.example))
        if show:
            if not out:
                print("no special locus found")
            for d in out:
                print(f"  {', '.join(d['equations']):45s} -> {d['group']}"
                      + ("" if d["sites"] == self.family.m else f" ({d['sites']} sites per primitive cell)")
                      + f"; {d['free_left']} free parameter(s) left")
        return out

    # --- recipes --------------------------------------------------------------------------------
    def recipe(self):
        return self.family.recipe()

    def code(self):
        """One line that rebuilds this lattice."""
        r = self.recipe()
        pt = lambda p: "(" + ", ".join(_fmt(x) for x in p) + ")"
        s = ", ".join(pt(p) for p in r["sites"])
        b = ", ".join(f"({pt(p)}, {pt(q)})" for p, q in r["bonds"])
        lat = "".join(f", {k}={_fmt(r[k])}" for k in ("a", "b", "gamma") if k in r)
        return f'Lattice.from_recipe(dict(group="{r["group"]}", sites=[{s}], bonds=[{b}]{lat}))'

    # --- the network, with the symmetry forgotten ---------------------------------------------------
    @property
    def network(self):
        """Key of the network alone (dsymbol.network_key): equal for two drawings of the same network."""
        return None if self.key is None else network_key(self.key)

    @property
    def symmetry_index(self):
        """How many times larger the network's own (combinatorial) symmetry is than G. 1: G is all of it;
        > 1: this is a less symmetric drawing of a more symmetric network."""
        return None if self.key is None else symmetry_index(self.key)

    def vertex_types(self):
        """Face sizes round a site, one string per kind of site, e.g. ['4.4.4.4'] (read from the cell)."""
        return _vertex_types(self.cell())

    # --- views ----------------------------------------------------------------------------------
    def cell(self, **values):
        return self.family.cell(**values)

    def dsymbol(self):
        return _dsymbol(self.cell(), ops_in_cell(self.family))

    def wyckoff(self):
        """Wyckoff position of each site orbit: ITA letters from groups.label_family if available,
        else the legacy label (multiplicity per primitive cell, site symmetry, kind)."""
        try:
            from .groups import label_family
            return [f"{d['name']} {d['site_symmetry']}" for d in label_family(self.family)]
        except Exception:
            return [f"{g['mult']} {g['sym']} ({g['kind']})" for g in self.family.gens]

    def info(self):
        f = self.family
        b = f.bonds()
        deg = degrees_of(f.m, b)
        L = f.bond_lengths()
        return dict(group=self.group, rank=self.rank, site_orbits=self.wyckoff(),
                    sites_per_cell=f.m, bond_orbits=[dict(bonds_per_cell=len(o), length=round(L[k], 4),
                                                          joins=sorted({f.site_orbit[o[0][0]] + 1, f.site_orbit[o[0][1]] + 1}))
                                                     for k, o in enumerate(f.orbits)],
                    bonds_per_cell=len(b), coordination=sorted(set(deg)), maxwell=2 * f.m - len(b),
                    connected=periodic_connected(f.m, b), free_parameters=f.names,
                    dsymbol_size=None if self.key is None else len(self.key), **{k: v for k, v in self.meta.items() if k in ("routes", "tier", "open")})

    def plot(self, n=2, ax=None, title=True, **kw):
        return self.family.plot(n=n, ax=ax, title=title, **kw)

    def __repr__(self):
        return (f"Lattice({self.group}, rank {self.rank}: {self.rank[0]} site orbit(s) {self.wyckoff()}, "
                f"{self.rank[1]} bond orbit(s), {self.rank[2]} sites and {self.rank[3]} bonds per primitive cell)")

    def to_dict(self):
        return dict(recipe=self.recipe(), key=None if self.key is None else [list(r) for r in self.key], rank=list(self.rank),
                    planar_values=self.planar_values[:20], meta=self.meta)

    @classmethod
    def from_dict(cls, d):
        return cls.from_recipe(d["recipe"], key=d["key"], rank=d["rank"], planar_values=d.get("planar_values"),
                               meta=d.get("meta"))


# ----------------------------------------------------------------------------------------------
# from a search entry to a Lattice with good display values
# ----------------------------------------------------------------------------------------------
def _angle_ratio(cell):
    """Smallest angle between neighbouring bonds at a site, divided by the even angle 360/degree of that
    site (1 = bonds evenly spread everywhere, 0 = two bonds on top of each other)."""
    A = cell.A
    X = np.asarray(cell.basis, float)
    ang = {}
    for (i, j, n1, n2, _t) in cell.bonds:
        v = X[j] + A @ np.array([n1, n2], float) - X[i]
        ang.setdefault(i, []).append(math.atan2(v[1], v[0]))
        ang.setdefault(j, []).append(math.atan2(-v[1], -v[0]))
    worst = 1.0
    for lst in ang.values():
        lst.sort()
        gaps = [(lst[(k + 1) % len(lst)] - lst[k]) % (2 * math.pi) for k in range(len(lst))]
        if len(lst) > 1:
            worst = min(worst, min(gaps) * len(lst) / (2 * math.pi))
    return worst


def _quality(fam, v, loci):
    """How good a drawing is, between 0 and 1: the smallest of
    the clearance (site to a bond it is not on, average spacing 1), shortest / longest bond,
    the bond-angle evenness (_angle_ratio), and 2 x the distance to the nearest supergroup locus
    (so that the drawing is clearly a lattice of G and not almost one of a larger group). -1: bonds cross."""
    c = fam.cell(**v)
    if has_crossings(c):
        return -1.0
    L = fam.set(**v).bond_lengths()
    dist = min((Lc.residual(v) for Lc in loci), default=1.0)
    return min(clearance(c), min(L) / max(L), _angle_ratio(c), 2 * dist)


_CELL_RANGE = dict(b=(0.55, 1.8, 0.06), gamma=(55.0, 125.0, 4.0))        # (low, high, step) of the cell search


def _best_values(fam, key, candidates, rng, polish=300, accept=None):
    """Default parameter values of a lattice (technical choice): start from the best of `candidates` (dicts of
    values, all with D-symbol key `key`) by _quality, then climb: `polish` random small moves of the site
    parameters and of the free cell parameters, each kept only if the quality goes up, the lattice is still
    planar with the SAME key, and accept(values) is true when given (points rule: the bonds are still the
    Voronoi ones). Finally rounded to 3 decimals if that keeps planarity, clearance, the key, the exact group
    (check_family) and accept; if the climbed point fails that check, the starting candidate is used."""
    key = _key_tuple(key)
    try:
        loci = fam.supergroup_loci()
    except Exception:
        loci = []
    best = None
    for c in candidates:
        v = dict(fam.values, **c)
        q = _quality(fam, v, loci)
        if q >= 0 and (best is None or q > best[0]):
            best = (q, v)
    if best is None:
        raise RuntimeError(f"{fam.group}: no candidate value is planar by the legacy crossing test "
                           "(the two crossing tests disagree: report it)")
    start = best[1]
    q, v = best
    names = [n for n in fam.names if n in fam.site_names or n in _CELL_RANGE]
    for it in range(polish if names else 0):
        shrink = 1.0 - 0.8 * it / polish
        w = dict(v)
        move = [n for n in names if rng.random() < 0.5] or [names[int(rng.integers(len(names)))]]   # a random subset
        for n in move:
            if n in _CELL_RANGE:
                lo, hi, st = _CELL_RANGE[n]
                w[n] = float(min(hi, max(lo, v[n] + rng.normal(0.0, st * shrink))))
            else:
                w[n] = float(v[n] + rng.normal(0.0, 0.04 * shrink))
        qw = _quality(fam, w, loci)
        if qw > q:
            try:
                same = _key_tuple(family_key(fam, **w)) == key and (accept is None or accept(w))
            except (DSymbolError, RuntimeError, ValueError):
                same = False
            if same:
                q, v = qw, w

    def good(vals):
        try:
            st = check_family(fam.set(**vals), key=key)
        except Exception:
            return False
        return (st["planar"] and st["clearance"] > 0.02 and st["same_key"] and st["exact"]
                and (accept is None or accept(vals)))

    vr = {k: round(x, 3) for k, x in v.items()}
    if good(vr):
        return vr
    if good(v):
        return v
    sr = {k: round(x, 3) for k, x in start.items()}
    return sr if good(sr) else start


def _display_values(e, rng, max_try=64, polish=300):
    """Default values of a search entry: _best_values on its planar samples."""
    fam = e.family()
    return _best_values(fam, e.key, [e.cd.values(e.cd.U[s]) for s in e.samples[:max_try]], rng, polish)


def lattice_from_entry(e, rng):
    v = _display_values(e, rng)
    fam = e.family(v)
    st = check_family(fam, key=e.key)
    if not st["exact"]:                                  # the default cell happens to be special
        for _ in range(5):
            from .enumerate import random_cell_params
            fam = fam.set(**random_cell_params(fam, rng))
            st = check_family(fam, key=e.key)
            if st["exact"]:
                break
    cnt = len(_symmetry_ops(fam.cell(), use_types=False))
    routes = dict(count=fam.group if cnt == len(fam._ops) else f"{cnt} operations, not {len(fam._ops)}",
                  legacy=st["group"] if st["exact"] else f"{st['group']} ({st['primitive_sites']} sites)",
                  spglib=st["spglib"])
    agree = cnt == len(fam._ops) and st["exact"] and st["spglib"] == fam.group and st["same_key"] and st["planar"]
    planar = [e.cd.values(e.cd.U[s]) for s in e.samples[:20]]
    meta = dict(tier=list(e.tier), structures=e.structures, n_planar_samples=len(e.samples), routes=routes,
                routes_agree=bool(agree), open=bool(getattr(e, "open", False)), wyckoff_search=e.cd.wyckoff)
    L = Lattice(fam, key=e.key, rank=e.rank, planar_values=planar, meta=meta)
    L.status = st
    return L


class MinimalResult:
    """Result of minimal_lattices: .lattices (first n), .all, .tiers, .coverage, .notes, .show()."""

    def __init__(self, group, lattices, tiers, coverage, notes=(), n=7, seconds=None, bounds=None, counts=None,
                 gaps=(), same=None, other=None):
        self.group, self.drawings, self.tiers, self.coverage = group, list(lattices), list(tiers), coverage
        self.gaps = list(gaps)
        self.notes, self.n, self.seconds, self.bounds, self.counts = list(notes), n, seconds, bounds or {}, counts or {}
        self.same = same or self.bounds.get("same", "network")
        # .drawings: every drawing found (distinct D-symbol under G), by rank.
        # .all: one entry per network, its simplest drawing (same == "network"), or every drawing (rule B).
        # .other[k]: the other drawings of the network of .all[k]. .lattices: the first n of .all.
        self.rule = self.bounds.get("rule", "bonds")
        self.all, self.other, first = [], [], {}
        if other is not None:                    # points rule: entries given, with their other Voronoi networks
            self.all, self.other = list(self.drawings), [list(o) for o in other]
            self.drawings = self.all + [L for o in self.other for L in o]
        for L in ([] if other is not None else self.drawings):
            net = L.network if self.same == "network" else L.key
            if net in first and net is not None:
                self.other[first[net]].append(L)
            else:
                first[net] = len(self.all); self.all.append(L); self.other.append([])
        self.lattices = self.all[:n]
        self.ties = [L for L in self.all[n:] if self.lattices and L.rank == self.lattices[-1].rank]
        self.open = [L for L in self.all if L.meta.get("open")]

    def take(self, n):
        """The same result showing only its first n lattices (n at most the number it was built with:
        further entries were found but not finished; search again with minimal_lattices(group, n=...))."""
        if n > self.n:
            raise ValueError(f"this result was built for the first {self.n} lattices; for more, run "
                             f"minimal_lattices({self.group!r}, n={n}, rule={self.rule!r})")
        import copy
        r = copy.copy(self)
        r.n, r.lattices = n, self.all[:n]
        r.ties = [L for L in self.all[n:] if r.lattices and L.rank == r.lattices[-1].rank]
        return r

    def __getitem__(self, k):
        return self.lattices[k]

    def __len__(self):
        return len(self.lattices)

    def table(self):
        rows = []
        for k, L in enumerate(self.lattices, 1):
            rows.append(dict(no=k, rank=L.rank, wyckoff=L.wyckoff(), bond_orbits=[len(o) for o in L.family.orbits],
                             free=L.family.names, open=L.meta.get("open", False), routes_agree=L.meta.get("routes_agree"),
                             vertex_types=L.vertex_types(), symmetry_index=L.symmetry_index,
                             other_drawings=len(self.other[k - 1])))
        return rows

    def show(self, plot=True, ncols=4, n=2):
        print(self.coverage)
        for t in self.tiers:
            if t["status"] != "empty (proved)" and self.rule != "points":
                print(f"  tier ({t['k']}, {t['n']}): {t['status']}, {t['entries']} new entr(y/ies)")
        what = "lattices (points-only rule)" if self.rule == "points" else "networks" if self.same == "network" else "drawings"
        print(f"{self.group}: first {len(self.lattices)} of {len(self.all)} {what} found ({len(self.drawings)} drawings)  "
              "rank = (site orbits, bond orbits, sites/cell, bonds/cell)")
        for r, L in zip(self.table(), self.lattices):
            flag = "  [OPEN tier: partial]" if r["open"] else ""
            flag += "" if r["routes_agree"] in (None, True) else "  [group routes DISAGREE]"
            print(f"#{r['no']} {r['rank']}  sites {'; '.join(r['wyckoff'])}  bond orbits {r['bond_orbits']}  "
                  f"free {r['free'] or '-'}{flag}")
            own = ("its own symmetry is this group" if r["symmetry_index"] == 1 else
                   f"its own symmetry is {r['symmetry_index']}x larger (this is a less symmetric drawing of it)")
            if self.rule == "points":
                more = (f"; {L.meta.get('same_network', 0)} other set(s) of points of this group give the same network "
                        f"(res.other[{r['no'] - 1}])" if L.meta.get("same_network") else "")
                if L.meta.get("phases", 1) > 1:
                    more += (f"; the Voronoi network of these points changes with the parameters "
                             f"(this one at {100 * L.meta.get('phase_share', 0):.0f} % of the sampled values)")
            else:
                more = (f"; {r['other_drawings']} more drawing(s) of it in this group: res.other[{r['no'] - 1}]"
                        if r["other_drawings"] else "")
            print(f"    network: faces round a site {' | '.join(r['vertex_types'])}; {own}{more}")
            print("    " + L.code())
        if self.ties:
            print(f"({len(self.ties)} further entr(y/ies) tie with #{len(self.lattices)} in rank: see .all)")
        for m in self.notes[:10]:
            print("note:", m)
        if plot and self.lattices:
            import matplotlib.pyplot as plt
            k = len(self.lattices); cols = min(ncols, k); rows = int(math.ceil(k / cols))
            fig, axs = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.6 * rows), squeeze=False)
            for i, (ax, L) in enumerate(zip(axs.flat, self.lattices), 1):
                L.plot(n=n, ax=ax, title=False)
                ax.set_title(f"#{i} {self.group} {L.rank}\n{'; '.join(L.wyckoff())}", fontsize=8)
            for ax in list(axs.flat)[k:]:
                ax.axis("off")
            fig.tight_layout()
            return fig
        return None

    def to_dict(self):
        return dict(group=self.group, n=self.n, coverage=self.coverage, tiers=self.tiers, notes=self.notes,
                    seconds=self.seconds, bounds=self.bounds, counts=self.counts, gaps=self.gaps,
                    entries=[L.to_dict() for L in (self.all if self.rule == "points" else self.drawings)],
                    other=[[L.to_dict() for L in o] for o in self.other] if self.rule == "points" else None)

    @classmethod
    def from_dict(cls, d, same=None):
        return cls(d["group"], [Lattice.from_dict(x) for x in d["entries"]], d["tiers"], d["coverage"], d.get("notes", ()),
                   d.get("n", 7), d.get("seconds"), d.get("bounds"), d.get("counts"), d.get("gaps", ()), same=same,
                   other=None if d.get("other") is None else [[Lattice.from_dict(x) for x in o] for o in d["other"]])


def _vertex_types(cell):
    """Sorted face sizes round each site of the cell, as strings, one per different kind."""
    from .dsymbol import _darts, _rotation, _faces
    F, darts, index, rev = _darts(cell)
    nxt, prv, deg = _rotation(cell, F, darts)
    faces, face_of = _faces(darts, rev, prv)
    out = {}
    for k, d in enumerate(darts):                          # the face on the left of dart k is a corner at its tail
        out.setdefault(d[0], []).append(len(faces[face_of[k]]))
    return sorted({".".join(map(str, sorted(v))) for v in out.values()})


def _fmt(x):
    from fractions import Fraction
    f = Fraction(x).limit_denominator(12)
    if abs(float(f) - x) < 1e-9:
        return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
    return repr(round(float(x), 6))


def coverage_text(group, b, tiers, gaps=()):
    done = [f"({t['k']},{t['n']})" for t in tiers if t["status"] == "complete"]
    opn = [f"({t['k']},{t['n']})" for t in tiers if t["status"].startswith("OPEN")]
    exh = any(t["status"].startswith("exhausted") or t["status"] == "empty (bound)" for t in tiers)
    s = (f"{group}: tiers (site orbits k, bond orbits n) searched in order; complete: {', '.join(done) or '-'}"
         + (f"; OPEN (time limit, partial): {', '.join(opn)}" if opn else "")
         + (f"; every tier inside the bounds was searched" if exh else "") + ".\n"
         f"Bounds: bond orbits with a bond at most {b['reach']} primitive cell(s) long in each direction for some "
         f"position of the generating points; at most {b['max_bond_orbits']} bond orbits (lattices with more bond "
         f"orbits for the same k are NOT covered); every site >= {b['min_degree']} bonds; planarity tested at "
         f"{b['samples']} random site-parameter samples per configuration (seed {b['seed']}): a planar window "
         "missed by every sample is not found.")
    if b.get("rule") == "sheet":
        s += ("\nSheet rule: only choices of sites whose points alone have exactly the symmetry of the group were used "
              "(tested at 3 random parameter values each).")
    if gaps:
        s += ("\nRANK GAP: tiers with more than max_bond_orbits bond orbits are possible (planarity bound) and were "
              "NOT searched for k = " + ", ".join(sorted({str(g["k"]) for g in gaps}))
              + "; lattices there would rank before the stopping tier. Configurations: "
              + "; ".join(f"k={g['k']} {g['wyckoff']} up to n={g['n_max']}" for g in gaps[:8])
              + (" ..." if len(gaps) > 8 else ""))
    elif any(t["status"] == "complete" for t in tiers):
        s += "\nNo rank gap: below the stopping tier no configuration allows more than max_bond_orbits bond orbits (planarity bound)."
    return s


def minimal_lattices(group, n=7, min_degree=3, reach=1, max_bond_orbits=6, samples=256, time_limit=600, seed=0,
                     use_numba=True, verbose=False, same="network", rule="points"):
    """The first n most Bravais-like lattices of `group`.
    rule="points" (default; author, 2026-11-14): points.points_lattices -- sets of points with exactly the symmetry
        of the group, fewest kinds of site and fewest sites first, bonds = Voronoi neighbours (only n, seed and
        time_limit are used).
    rule="sheet" (the rule of the author's hand-made sheet, 2026-11-14): sites as in the points rule (the points
        alone have exactly the symmetry of the group), then the fewest kinds of bond: rank (kinds of site, kinds of
        bond, sites, bonds), bonds chosen among short ones, every site >= min_degree bonds, one entry per network.
    rule="bonds": the same search by rank without the condition on the points (bonds may lower the symmetry);
        same="network": n different networks, each by its simplest drawing; same="action": rule B."""
    if rule == "points":
        from .points import points_lattices
        return points_lattices(group, n=n, seed=seed, time_limit=time_limit)
    from .enumerate import search
    t0 = time.time()
    r = search(group, n=n, min_degree=min_degree, reach=reach, max_bond_orbits=max_bond_orbits, samples=samples,
               time_limit=time_limit, seed=seed, use_numba=use_numba, verbose=verbose, same=same,
               points_exact=(rule == "sheet"))
    rng = np.random.default_rng(seed + 1)
    lats = [lattice_from_entry(e, rng) for e in r["entries"]]
    notes = list(r["notes"])
    for L in lats:
        if not L.meta["routes_agree"]:
            notes.append(f"group routes disagree for {L.rank} {L.wyckoff()}: {L.meta['routes']}, status {L.status}")
    return MinimalResult(r["group"], lats, r["tiers"], coverage_text(r["group"], r["bounds"], r["tiers"], r["gaps"]), notes, n,
                         time.time() - t0, r["bounds"], r["counts"], r["gaps"], same=same)


def load_catalogue(path=None, group=None, rule="points"):
    """{group: MinimalResult} of the stored lists: rule="points" (default) from data/points.json, rule="sheet" from
    data/sheet.json (both built by tools/build_points.py), rule="bonds" from data/minimal_b.json (tools/build_catalogue.py)."""
    with open(path or dict(points=DATA_POINTS, sheet=DATA_SHEET, bonds=DATA)[rule], encoding="utf8") as f:
        data = json.load(f)
    groups = data["groups"]
    if group is not None:
        return MinimalResult.from_dict(groups[group])
    return {g: MinimalResult.from_dict(d) for g, d in groups.items()}
