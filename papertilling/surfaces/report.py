"""Which surfaces carry a lattice: one table, OK / IMPOSSIBLE / NOT FOUND, with the reason in words.

    from papertilling import named
    from papertilling.surfaces import surface_table
    T = surface_table(named("chiral_square"))          # prints the table; T is a list of dicts (T.text() for the text)

For each surface (torus, cylinder, Moebius band, Klein bottle, sphere, projective plane, genus 2, genus 3) the constructor
is called at its default size; if it is refused for a reason that another size, centre or glide class could change
(SurfaceNotFound), a few other settings are tried, within `budget` seconds per surface. A surface counts as OK only when
it is built AND its checks() pass.

    OK          built, checks() passed (sites, cones and the settings used are listed)
    IMPOSSIBLE  proved from the lattice's symmetry (SurfaceImpossible; refusal.py): no mirror or glide line -> no Moebius
                band, Klein bottle or projective plane; no rotation -> no sphere, no projective plane
    NOT FOUND   every setting tried was refused (SurfaceNotFound) or built with failing checks: not a proof that the
                surface does not exist
    ERROR       an unexpected exception (a bug in the package, not a statement about the lattice): please report it
"""
from __future__ import annotations

import re
import time

SURFACES = ("torus", "cylinder", "Moebius band", "Klein bottle", "sphere", "projective plane", "genus 2", "genus 3")
_KEYS = ("T1", "T2", "wrap", "length", "along", "across", "width", "glide")


class SurfaceTable(list):
    """The rows of surface_table (dicts); .text() the printed table; .lattice; .seconds; .figure (plot=True)."""

    def text(self):
        L = self.lattice
        head = f"Surfaces for {_name(L)} ({L.group}), {self.seconds:.1f} s:"
        lines = [head, f"  {'surface':<17}{'sites':>6}  {'cones (Frank angles)':<22}{'time':>7}   result"]
        for r in self:
            lines.append(f"  {r['surface']:<17}{(r['sites'] if r['sites'] is not None else '-'):>6}  {r['cones']:<22}"
                         f"{r['seconds']:6.1f}s   {r['status']}: {r['reason']}")
        return "\n".join(lines)

    def statuses(self):
        return {r["surface"]: r["status"] for r in self}


def _name(L):
    meta = getattr(L, "meta", None) or {}
    return meta.get("name") or meta.get("key") or "the lattice"


def _constructor(name):
    from . import flat, genus, polyhedra
    return {"torus": flat.torus, "cylinder": flat.cylinder, "Moebius band": flat.mobius_band,
            "Klein bottle": flat.klein_bottle, "sphere": polyhedra.sphere, "projective plane": polyhedra.projective_plane,
            "genus 2": lambda L, **kw: genus.genus_surface(L, g=2, **kw),
            "genus 3": lambda L, **kw: genus.genus_surface(L, g=3, **kw)}[name]


def _attempts(name, L, facts):
    """The settings tried, in order (the first is the constructor's default)."""
    n = max(facts["rotation_orders"], default=1)
    if name in ("torus", "cylinder"):
        return [{}, {"sites": 400}]
    if name in ("Moebius band", "Klein bottle"):
        from .flat import glides
        k = max(1, len(glides(L)))
        return [{}] + [{"which": w} for w in range(1, k)] + [{"sites": 400}]
    pillow = sorted(((h, k) for h in range(1, 9) for k in range(1, 9)), key=lambda z: (z[0] * z[1], z))   # cheap nets
    if name == "sphere":
        zs = pillow if n == 2 else [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1)]
        return [{"size": z} for z in zs]
    if name == "projective plane":
        zs = pillow if n == 2 else [(1, 1), (1, 0), (2, 0), (2, 1), (2, 2), (3, 0)]
        return [{"size": z} for z in zs]
    return [{"torus_size": z} for z in ((6, 6), (8, 8), (10, 10))]


def _kw(kw):
    return ", ".join(f"{k}={v}" for k, v in kw.items()) or "default"


def _cones(S):
    angles = {}
    for c in getattr(S, "cones", []) or []:
        a = round(float(c["angle"]), 3)
        angles[a] = angles.get(a, 0) + 1
    return ", ".join(f"{k} x {a:+g}" for a, k in sorted(angles.items(), reverse=True)) or "none"


def _built(S):
    P = S.params
    if "branch" in P:
        out = f"double cover of the torus {P['T1']} x {P['T2']}, {2 * len(P['branch'])} branch points"
    elif "solid" in P:
        out = f"size {P['size']}, corners at centre class {P['centre']} ({P['centre_location']})"
    else:
        out = ", ".join(f"{k} {P[k]}" for k in _KEYS if k in P and P[k] is not None)
    return f"{S.kind}" + (f": {out}" if out else "")


def _why(msg):
    """Plain words for one refusal message."""
    m = re.search(r"\(([a-z_0-9, ]+) failed", msg)
    if "fold the lattice" in msg:
        return "the cones fold the lattice" + (f" ({m.group(1)} failed)" if m else "")
    if "midpoint of a bond" in msg:
        return "a cone falls on a bond midpoint"
    if "not symmetries" in msg:
        return "a gluing of the net is not a symmetry of the lattice"
    if "cut by a cone" in msg:
        return "a bond or a face is cut by a cone"
    if "fixes a site" in msg:
        return "the glide fixes a site"
    if "checks failed" in msg:
        return msg
    return msg.split(":")[0][:90]


def _one(name, L, facts, attempts, budget):
    t0 = time.perf_counter()
    ctor = _constructor(name)
    row = dict(surface=name, status="NOT FOUND", reason="", S=None, sites=None, cones="-", seconds=0.0, tried=[], hint=None,
               scope=None)
    whys, centres, last, folds = [], [], None, []
    from .refusal import SurfaceImpossible, SurfaceNotFound
    for k, kw in enumerate(attempts):
        if k and time.perf_counter() - t0 > budget:
            whys.append(f"time budget of {budget:g} s reached")
            break
        try:
            S = ctor(L, **kw)
            ch = S.checks(show=False)
        except SurfaceImpossible as e:
            row.update(status="IMPOSSIBLE", reason=e.reason, tried=row["tried"] + [_kw(kw)],
                       scope=getattr(e, "scope", "group"))
            break
        except SurfaceNotFound as e:
            last = e
            row["tried"].append(_kw(kw))
            for msg in (getattr(e, "causes", None) or [str(e)]):
                m = re.search(r"fold the lattice \(([a-z_0-9, ]+) failed", msg)
                if m:
                    folds += [c for c in m.group(1).split(", ") if c not in folds]
                    continue
                w = _why(msg)
                if w not in whys:
                    whys.append(w)
            centres += [c for c in getattr(e, "centres", []) if c not in centres]
            if not e.retry:
                break
            continue
        except Exception as e:                                       # a bug, not a statement about the lattice
            row.update(status="ERROR", reason=f"{type(e).__name__}: {e} (unexpected: please report)",
                       tried=row["tried"] + [_kw(kw)])
            break
        if ch.get("ok"):
            before = f"; first refused at {', '.join(row['tried'])}" if row["tried"] else ""
            row.update(status="OK", reason=f"{_built(S)}{before}", S=S, sites=len(S.sites), cones=_cones(S))
            row["tried"].append(_kw(kw))
            break
        bad = [c for c in ("no_loops", "no_double_bonds", "plane_coordination", "euler", "faces_at_least_3") if ch.get(c) is False]
        row["tried"].append(_kw(kw))
        w = f"built at {_kw(kw)} but its checks failed ({', '.join(bad) or 'see checks()'})"
        whys.append(w)
    if folds:
        whys.insert(0, f"the cones fold the lattice ({', '.join(folds)} failed)")
    if row["status"] == "NOT FOUND":
        if last is not None and not last.retry:
            row["reason"] = last.reason + "; " + "; ".join(last.tried)
        else:
            tried = row["tried"] if len(row["tried"]) <= 8 else row["tried"][:3] + ["..."] + row["tried"][-2:]
            row["reason"] = ("refused at every setting tried: " + "; ".join(whys) + f". Tried {len(row['tried'])} "
                             f"setting(s): {' | '.join(tried)}" + (f", at centres {', '.join(centres)}" if centres else ""))
        row["hint"] = last.hint if (last is not None and len(row["tried"]) == 1) else None   # else: already tried
    row["seconds"] = time.perf_counter() - t0
    return row


def surface_table(lattice, sizes="auto", show=True, plot=False, budget=20.0, only=None):
    """Try every surface (SURFACES) for `lattice` (a Lattice or a named lattice's name) and return a SurfaceTable: a list
    of dicts surface, status ('OK', 'IMPOSSIBLE', 'NOT FOUND', or 'ERROR' for a bug), reason (words), S (the Surface or
    None), sites, cones (summary of the Frank angles), seconds, tried (the settings tried), hint.
    sizes: 'auto' (the default, then a few other sizes / centres / glide classes / torus sizes while refused, within
    `budget` seconds per surface), 'default' (the default only), or a dict {surface: [kwargs, ...]} for the settings of
    some surfaces (the others 'auto'). only: a list of surface names to try. show: print the table. plot: thumbnails of
    the OK surfaces (3D where there is a 3D shape, else the flat net) in one figure (table.figure)."""
    from .flat import _cell_and_ops
    from .refusal import symmetry_facts
    t0 = time.perf_counter()
    L = _cell_and_ops(lattice)[0]
    facts = symmetry_facts(L)
    table = SurfaceTable()
    table.lattice, table.facts, table.figure = L, facts, None
    for name in (only or SURFACES):
        if name not in SURFACES:
            raise ValueError(f"unknown surface {name!r}: one of {SURFACES}")
        att = _attempts(name, L, facts)
        if isinstance(sizes, dict) and name in sizes:
            att = [dict(kw) for kw in sizes[name]]
        elif sizes in ("default", None):
            att = att[:1]
        elif sizes != "auto" and not isinstance(sizes, dict):
            raise ValueError("sizes: 'auto', 'default' or a dict {surface: [kwargs, ...]}")
        table.append(_one(name, L, facts, att, budget))
    table.seconds = time.perf_counter() - t0
    if show:
        print(table.text())
    if plot:
        table.figure = _plot(table)
    return table


surfaces = surface_table


def _plot(table, size=3.2):
    import matplotlib.pyplot as plt
    ok = [r for r in table if r["status"] == "OK"]
    if not ok:
        return None
    nc = min(4, len(ok))
    nr = -(-len(ok) // nc)
    fig = plt.figure(figsize=(size * nc, size * nr))
    for k, r in enumerate(ok):
        S = r["S"]
        title = f"{r['surface']}: {r['sites']} sites"
        if S.xyz is not None:
            ax = fig.add_subplot(nr, nc, k + 1, projection="3d")
            S.plot(ax=ax, title=title)
        else:
            ax = fig.add_subplot(nr, nc, k + 1)
            S.plot_net(ax=ax, title=title, legend=False)
        ax.title.set_fontsize(9)
    fig.suptitle(f"{_name(table.lattice)} ({table.lattice.group}): the surfaces found", fontsize=10)
    fig.tight_layout()
    return fig
