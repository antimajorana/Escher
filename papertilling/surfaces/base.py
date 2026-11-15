"""The Surface object shared by every surface construction (flat.py, polyhedra.py, genus.py).

A Surface holds the lattice on the surface as a graph (sites, bonds, faces), where each site came from in the plane, the
cone points (disclinations) the surface needs, a flat picture of the pieces it is glued from, and an optional 3D shape.

PICTURES. plot_net() draws the flat piece (exact); plot() the 3D shape (faces as light shaded translucent polygons under
the bonds, far bonds paler); show() both side by side with one line saying what is exact and what is distorted. When a
construction knows the 3D shape as a map of the plane (xyz_map, plane points -> 3D, invariant under the gluings), bonds and
faces are drawn as curves on the surface; otherwise as straight segments between the sites' xyz.
"""
from __future__ import annotations

import numpy as np

BOND, BOND_FAR, FACE, SITE, CONE = "#1f5fb0", "#b4cbe8", "#f3e2a6", "#222222", "#d62728"


def _mix(c, w, t):
    """Colour c blended with colour w by t (0 -> c)."""
    from matplotlib.colors import to_rgb
    c, w = np.array(to_rgb(c)), np.array(to_rgb(w))
    return (1 - t)[..., None] * c + t[..., None] * w if np.ndim(t) else (1 - t) * c + t * w


def _view(elev, azim):
    """Unit vector pointing from the scene towards the camera (matplotlib's elev/azim, degrees)."""
    e, a = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


class Surface:
    """A lattice on a surface. sites: list of plane labels (i, n1, n2) (one representative per site); uv: their plane
    positions (Cartesian, inside the fundamental domain drawn by plot_net); bonds: list of (a, b, v) with v the plane bond
    vector from a's representative; faces: list of site cycles; cones: list of dicts (a cone at a site: 'site' or 'sites'
    as site indices, or at a face: 'face' as a face index or a site cycle; plus its angle); chi, orientable, closed; domain:
    the fundamental domain polygon and its gluings (for plot_net); xyz: a 3D position for every site (for plot); xyz_note:
    how faithful the 3D shape is; xyz_map: optional map of plane points (n, 2) -> (n, 3) giving the 3D shape (it must take
    the same value on points the gluings identify); view: optional default (elev, azim) for plot."""

    def __init__(self, kind, lattice, sites, uv, bonds, faces, chi, orientable, closed, domain, params, plane_degree, edge_sites=()):
        self.kind, self.lattice, self.sites, self.uv = kind, lattice, sites, np.asarray(uv, float)
        self.bonds, self.faces, self.chi, self.orientable, self.closed = bonds, faces, chi, orientable, closed
        self.domain, self.params, self.plane_degree, self.edge_sites = domain, params, plane_degree, set(edge_sites)
        self.cones = []
        self.xyz, self.xyz_note, self.xyz_map, self.view = None, "", None, None
        self.status = "OK"          # built (a refusal raises SurfaceImpossible or SurfaceNotFound instead; see refusal.py)

    def __repr__(self):
        return (f"Surface({self.kind}, {self.lattice.group}: {len(self.sites)} sites, {len(self.bonds)} bonds, {len(self.faces)} faces, "
                f"chi {self.chi}, {'orientable' if self.orientable else 'non-orientable'}, {'closed' if self.closed else 'with an open edge'})")

    def checks(self, show=True):
        """Every check as a dict; True means passed."""
        pairs = [tuple(sorted((a, b))) for a, b, _ in self.bonds]
        deg = np.zeros(len(self.sites), int)
        for a, b, _ in self.bonds:
            deg[a] += 1
            deg[b] += 1
        inner = [k for k in range(len(self.sites)) if k not in self.edge_sites]
        V, E, Fc = len(self.sites), len(self.bonds), len(self.faces)
        out = dict(no_loops=all(a != b for a, b in pairs), no_double_bonds=len(set(pairs)) == len(pairs),
                   plane_coordination=all(deg[k] == self.plane_degree[self.sites[k][0]] for k in inner),
                   euler=(V - E + Fc) == self.chi, V=V, E=E, F=Fc, chi=self.chi, V_minus_E_plus_F=V - E + Fc,
                   faces_at_least_3=all(len(f) >= 3 for f in self.faces))
        out["ok"] = all(out[k] for k in ("no_loops", "no_double_bonds", "plane_coordination", "euler", "faces_at_least_3"))
        if show:
            print(f"{self.kind}: V {V}, E {E}, F {Fc}, V - E + F = {V - E + Fc} (chi {self.chi}); no loops {out['no_loops']}, "
                  f"no double bonds {out['no_double_bonds']}, plane coordination kept {out['plane_coordination']}, "
                  f"faces >= 3 sides {out['faces_at_least_3']}  ->  {'OK' if out['ok'] else 'FAILED'}")
        return out

    # ------------------------------------------------------------------------------------------------------------
    # helpers for the pictures
    # ------------------------------------------------------------------------------------------------------------
    @property
    def name(self):
        meta = getattr(self.lattice, "meta", None) or {}
        return meta.get("name") or meta.get("key") or "lattice"

    def _cone_marks(self):
        """(site indices, face indices) the cones sit on, read from self.cones ('site', 'sites', 'face', 'faces')."""
        sites, faces = set(), set()
        index = {tuple(s): k for k, s in enumerate(self.sites)}
        canon = {frozenset(f): k for k, f in enumerate(self.faces)}

        def as_site(s):
            if isinstance(s, (int, np.integer)):
                return int(s)
            return index.get(tuple(s))

        def as_face(f):
            if isinstance(f, (int, np.integer)):
                return int(f)
            return canon.get(frozenset(int(x) for x in f))

        for c in self.cones:
            if not isinstance(c, dict):
                continue
            for key in ("site", "sites"):
                if c.get(key) is not None:
                    v = c[key]
                    for s in ([v] if key == "site" else v):
                        k = as_site(s)
                        if k is not None:
                            sites.add(k)
            for key in ("face", "faces"):
                if c.get(key) is not None:
                    v = c[key]
                    for f in ([v] if key == "face" else v):
                        k = as_face(f)
                        if k is not None and 0 <= k < len(self.faces):
                            faces.add(k)
        return sorted(sites), sorted(faces)

    def _curves(self, k):
        """Every bond as a 3D polyline of k + 1 points (on the surface when xyz_map is known): array (E, k + 1, 3)."""
        a = np.array([b[0] for b in self.bonds], int)
        b = np.array([b[1] for b in self.bonds], int)
        if self.xyz_map is None or k <= 1 or not len(self.bonds):
            return np.stack([self.xyz[a], self.xyz[b]], axis=1)
        v = np.array([np.asarray(x[2], float) for x in self.bonds])
        s = np.linspace(0.0, 1.0, k + 1)
        pts = self.uv[a][:, None, :] + s[None, :, None] * v[:, None, :]
        C = np.asarray(self.xyz_map(pts.reshape(-1, 2)), float).reshape(len(a), k + 1, 3)
        C[:, 0], C[:, -1] = self.xyz[a], self.xyz[b]                 # the ends sit exactly on the sites
        return C

    def _face_polygons(self, C):
        """Each face as a closed 3D polyline following its bonds' curves C (None for a face with a missing bond)."""
        ends = {}
        for e, (a, b, _) in enumerate(self.bonds):
            ends.setdefault((a, b), e)
            ends.setdefault((b, a), ~e)
        polys = []
        for f in self.faces:
            pieces = []
            for j in range(len(f)):
                e = ends.get((f[j], f[(j + 1) % len(f)]))
                if e is None:
                    pieces = None
                    break
                pieces.append(C[e][:-1] if e >= 0 else C[~e][::-1][:-1])
            polys.append(None if pieces is None else np.concatenate(pieces))
        return polys

    def stretch(self):
        """(smallest, largest) ratio of a bond's 3D length (chord) to its plane length: 1, 1 for an isometric shape."""
        if self.xyz is None or not self.bonds:
            return None
        r = [np.linalg.norm(self.xyz[b] - self.xyz[a]) / np.linalg.norm(v) for a, b, v in self.bonds if np.linalg.norm(v) > 0]
        return float(min(r)), float(max(r))

    # ------------------------------------------------------------------------------------------------------------
    # the flat piece
    # ------------------------------------------------------------------------------------------------------------
    def plot_net(self, ax=None, title=None, size=6, legend=True, rotate=False):
        """The fundamental domain in the plane with the lattice in it (exact lengths and angles). Edges with the same colour
        and the same arrows are glued, arrow onto arrow; bonds drawn faintly leave the piece and come back in through the
        glued edge; a grey edge is an open edge of the surface. rotate: turn a long piece so that it lies horizontally
        (a rigid motion of the picture only)."""
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.path import Path
        if ax is None:
            ax = plt.subplots(figsize=(size, size))[1]
        poly = np.asarray(self.domain["polygon"], float)
        Rm = np.eye(2)
        if rotate:
            c = poly - poly.mean(axis=0)
            w, Vec = np.linalg.eigh(c.T @ c)
            if w[1] > 2.25 * w[0]:                                     # long: its longest side horizontal
                sides = np.roll(poly, -1, axis=0) - poly
                d = sides[np.argmax(np.linalg.norm(sides, axis=1))]
                th = -np.arctan2(d[1], d[0])
                Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        P, poly = self.uv @ Rm.T, poly @ Rm.T
        scale = float(np.ptp(poly, axis=0).max()) or 1.0
        path = Path(np.vstack([poly, poly[:1]]), closed=True)
        eps = 1e-6 * scale + 1e-9

        def inside(Q):
            return path.contains_points(Q, radius=eps) | path.contains_points(Q, radius=-eps)

        ax.fill(poly[:, 0], poly[:, 1], color="#f6edc8", alpha=0.7, zorder=0, lw=0)
        if self.bonds:
            A = np.array([P[a] for a, _, _ in self.bonds])
            B = A + np.array([np.asarray(v, float) for _, _, v in self.bonds]) @ Rm.T
            ins = inside(A) & inside(B)
            segs = np.stack([A, B], axis=1)
            ax.add_collection(LineCollection(segs[ins], colors=BOND, linewidths=1.1, zorder=2))
            ax.add_collection(LineCollection(segs[~ins], colors=BOND, linewidths=0.9, alpha=0.3, linestyles=(0, (3, 2)), zorder=1))
            n_out = int((~ins).sum())
        else:
            n_out = 0
        ss = float(np.clip(1500.0 / max(len(P), 1), 3, 18))
        ax.scatter(P[:, 0], P[:, 1], s=ss, color=SITE, zorder=3, lw=0)
        cs, cf = self._cone_marks()
        if cs:
            ax.scatter(P[cs, 0], P[cs, 1], s=4 * ss + 20, color=CONE, zorder=4, lw=0)
        # the edges: glued ones in colour with matching arrows, open ones grey
        glued = [(Rm @ np.asarray(p, float), Rm @ np.asarray(q, float), lab, col) for p, q, lab, col in self.domain.get("edges", [])]
        kinds = []
        for (p, q, label, colour) in glued:
            if label not in kinds:
                kinds.append(label)
        cen = poly.mean(axis=0)
        for (p, q, label, colour) in glued:
            p, q = np.asarray(p, float), np.asarray(q, float)
            ax.plot([p[0], q[0]], [p[1], q[1]], "-", color=colour, lw=2.4, zorder=4, solid_capstyle="butt")
            d = q - p
            L = np.linalg.norm(d)
            if L == 0:
                continue
            d = d / L
            m = (p + q) / 2
            nk = kinds.index(label) + 1
            h = 0.035 * scale
            for j in range(nk):
                c = m + d * (j - (nk - 1) / 2) * 1.3 * h
                ax.annotate("", xy=c + d * h / 2, xytext=c - d * h / 2, zorder=5,
                            arrowprops=dict(arrowstyle="-|>", color=colour, lw=0, mutation_scale=16, shrinkA=0, shrinkB=0))
            nrm = np.array([-d[1], d[0]])
            if np.dot(nrm, m - cen) < 0:
                nrm = -nrm
            ax.text(*(m + nrm * 0.045 * scale), label, color=colour, fontsize=10, ha="center", va="center", zorder=6, weight="bold")
        n_open = 0
        for j in range(len(poly)):
            p, q = poly[j], poly[(j + 1) % len(poly)]
            hit = any((np.allclose(p, g[0], atol=eps) and np.allclose(q, g[1], atol=eps)) or
                      (np.allclose(p, g[1], atol=eps) and np.allclose(q, g[0], atol=eps)) for g in glued)
            if not hit:
                ax.plot([p[0], q[0]], [p[1], q[1]], "-", color="#9a9a9a", lw=2.0, zorder=4)
                n_open += 1
        ax.set_aspect("equal")
        pad = 0.08 * scale
        lo, hi = np.minimum(poly.min(axis=0), P.min(axis=0)) - pad, np.maximum(poly.max(axis=0), P.max(axis=0)) + pad
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.axis("off")
        ax.set_title(title or f"{self.name} ({self.lattice.group}) on a {self.kind}: the flat piece (exact)", fontsize=9)
        if legend:
            parts = ["same colour and arrows: glued edges, arrow onto arrow"]
            if n_out:
                parts.append("faint dashed: bonds through a glued edge")
            if n_open:
                parts.append("grey: open edge")
            if cs:
                parts.append("red: cone sites")
            ax.text(0.0, 0.0, chr(10).join(parts), transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color="#444444")
        return ax

    # ------------------------------------------------------------------------------------------------------------
    # the 3D shape
    # ------------------------------------------------------------------------------------------------------------
    def plot(self, ax=None, title=None, size=6, elev=None, azim=None, faces=True, sites=None, curve=None, alpha=0.6):
        """The surface in 3D (see xyz_note for how faithful the shape is): faces as light shaded translucent polygons under
        the bonds, sorted by depth together with the bonds (orthographic view), so near faces veil far bonds; bonds paler
        the further they are from the eye; sites small; cones (self.cones) in red. curve: pieces per bond when the shape
        is known as a map of the plane (default 6, 3 for big surfaces); alpha: opacity of the faces."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        if self.xyz is None:
            raise ValueError("no 3D shape for this surface; use plot_net()")
        e0, a0 = self.view if self.view is not None else (25, -60)
        elev, azim = (e0 if elev is None else elev), (a0 if azim is None else azim)
        if ax is None:
            ax = plt.figure(figsize=(size, size)).add_subplot(projection="3d")
        X = self.xyz
        E = len(self.bonds)
        k = curve if curve is not None else (6 if E <= 1500 else 3 if E <= 6000 else 1)
        C = self._curves(k)
        eye = _view(elev, azim)
        ax.computed_zorder = False
        ax.set_proj_type("ortho")
        cs, cf = self._cone_marks()
        polys, fc, ec, lw = [], [], [], []
        if faces and self.faces:
            fp = self._face_polygons(C)
            keep = [j for j, p in enumerate(fp) if p is not None and len(p) >= 3]
            if keep:
                fp = [fp[j] for j in keep]
                nrm = np.array([np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0) for p in fp])   # Newell normals
                nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-300
                lam = np.abs(nrm @ _view(elev + 30, azim - 40))
                col = _mix(FACE, "#8a7a52", 0.6 * (1 - lam))
                if cf:
                    hl = np.isin(np.array(keep), cf)
                    col[hl] = _mix(CONE, "#ffffff", np.full(int(hl.sum()), 0.3))
                polys += fp
                fc.append(np.column_stack([col, np.full(len(fp), alpha)]))
                ec.append(np.zeros((len(fp), 4)))
                lw.append(np.zeros(len(fp)))
        if E:
            # every piece of every bond is a 2-point polygon sorted together with the faces (painter's order), pulled a
            # little towards the eye so that it lies on top of the faces it borders (orthographic: the pull is invisible)
            blen = float(np.median(np.linalg.norm(C[:, -1] - C[:, 0], axis=1))) or 1.0
            depth = C.mean(axis=1) @ eye
            t = (depth - depth.min()) / (np.ptp(depth) or 1.0)
            reps = C.shape[1] - 1
            segs = np.concatenate([np.stack([C[:, j], C[:, j + 1]], axis=1) for j in range(reps)]) + 0.6 * blen * eye
            polys += list(segs)
            fc.append(np.zeros((len(segs), 4)))
            ec.append(np.tile(np.column_stack([_mix(BOND_FAR, BOND, t), np.ones(E)]), (reps, 1)))
            lw.append(np.tile(0.5 + 0.8 * t, reps))
        if polys:
            pc = Poly3DCollection(polys, facecolors=np.concatenate(fc), edgecolors=np.concatenate(ec),
                                  linewidths=np.concatenate(lw), zsort="average")
            pc.set_zorder(1)
            ax.add_collection3d(pc)
        if sites is None:
            sites = len(X) <= 1500
        if sites:
            d = X @ eye
            t = (d - d.min()) / (np.ptp(d) or 1.0)
            rgba = np.column_stack([_mix("#9fb4cc", SITE, t), 0.15 + 0.85 * t ** 2])
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=float(np.clip(1200.0 / max(len(X), 1), 1.5, 9)),
                       c=rgba, depthshade=False, lw=0, zorder=3)
        if cs:
            ax.scatter(X[cs, 0], X[cs, 1], X[cs, 2], s=40, color=CONE, depthshade=False, lw=0, zorder=4)
        lo, hi = X.min(axis=0), X.max(axis=0)
        span = np.maximum(hi - lo, 1e-3 * max(float(np.ptp(X)), 1.0))
        ax.set_xlim(lo[0], lo[0] + span[0]); ax.set_ylim(lo[1], lo[1] + span[1]); ax.set_zlim(lo[2], lo[2] + span[2])
        ax.set_box_aspect(tuple(span), zoom=1.25)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(title or f"{self.name} ({self.lattice.group}) on a {self.kind} in 3D\n{self.xyz_note}", fontsize=9)
        return ax

    def show(self, size=5.0, title=None, **kw):
        """The flat piece (exact) and the 3D shape side by side, with one line saying what is exact and what is distorted.
        Extra keywords go to plot(). Returns the figure."""
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(2 * size + 0.6, size + 0.3))
        ax1 = fig.add_subplot(1, 2, 1)
        self.plot_net(ax=ax1, title="flat piece", rotate=True)
        if self.xyz is not None:
            ax2 = fig.add_subplot(1, 2, 2, projection="3d")
            self.plot(ax=ax2, title="3D shape", **kw)
        V = len(self.sites)
        line = title or (f"{self.name} ({self.lattice.group}) on a {self.kind}: {V} sites, chi = {self.chi}.  "
                         f"Flat piece: exact.  3D: {self.xyz_note or 'none'}")
        fig.suptitle(line, fontsize=min(10.0, 10.0 * 125 / max(len(line), 1) * size / 5.0))
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.13, top=0.9, wspace=0.02)
        return fig


def show(S, **kw):
    """The flat piece and the 3D shape of a Surface side by side (S.show())."""
    return S.show(**kw)
