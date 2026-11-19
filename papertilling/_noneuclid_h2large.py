"""O(N) H^2 patch with spatial-hash dedup, Rust-backed or pure Python."""
from __future__ import annotations

import json
import math

from . import _hyperbolic_vc as vc
from ._noneuclid_common import _check_p, _rs_fn
from ._noneuclid_models import geodesic_arc_poincare, hyp_distance_poincare

def _su_trans(z):
    inv = 1.0 / math.sqrt(1.0 - _check_p(z))
    return (complex(inv), complex(z[0] * inv, z[1] * inv))


class _PolarHashPy:
    def __init__(self, min_sep: float):
        self.tol = 0.5 * max(1e-4, min_sep); self.items: list[tuple[complex, complex]] = []; self.z_pts: list[list[float]] = []; self.buckets: dict[tuple[int, int], list[int]] = {}
    @staticmethod
    def _mul(m1, m2):
        a, b = m1[0] * m2[0] + m1[1] * m2[1].conjugate(), m1[0] * m2[1] + m1[1] * m2[0].conjugate()
        inv = 1.0 / math.sqrt(max(1e-18, abs(a) ** 2 - abs(b) ** 2))
        return (a * inv, b * inv)
    @staticmethod
    def _origin(m): z = m[1] / m[0].conjugate(); return [z.real, z.imag]
    @staticmethod
    def _dist(m1, m2): return 2.0 * math.asinh(abs(m1[0].conjugate() * m2[1] - m1[1] * m2[0].conjugate()))
    def _nsec(self, br: int) -> int: return 1 if br <= 0 else int(min(1_000_000_000, max(1, math.floor(math.tau * math.sinh((br + 0.5) * self.tol) / self.tol))))
    def _iter_candidates(self, rho, th, rad, r_sh):
        b0, b1 = max(0, int(math.floor((rho - rad - self.tol) / self.tol))), max(0, int(math.ceil((rho + rad + self.tol) / self.tol)))
        for b in range(b0, b1 + 1):
            nsec = self._nsec(b); btc = int(math.floor((th / math.tau) * nsec)) % nsec
            dsec = nsec if (rho <= rad + 2.2 * self.tol or b <= 2) else int(math.ceil(2.0 * math.asin(min(1.0, r_sh / max(1e-6, math.sqrt(math.sinh(rho) * math.sinh(b * self.tol))))) / math.tau * nsec)) + 2
            span = min(nsec, 2 * dsec + 1)
            for off in range(span): yield from self.buckets.get((b, (btc - span // 2 + off) % nsec), ())
    def insert(self, m) -> tuple[int, bool]:
        rho, z = 2.0 * math.asinh(abs(m[1])), self._origin(m); th, br = math.atan2(z[1], z[0]) % math.tau, max(0, int(math.floor(2.0 * math.asinh(abs(m[1])) / self.tol)))
        best = min(((self._dist(self.items[i], m), i) for i in self._iter_candidates(rho, th, self.tol, math.sinh(0.5 * self.tol))), default=(math.inf, -1))
        if best[0] < self.tol: return best[1], False
        idx = len(self.items); nsec = self._nsec(br); self.items.append(m); self.z_pts.append(z); self.buckets.setdefault((br, int(math.floor((th / math.tau) * nsec)) % nsec), []).append(idx); return idx, True
    def ball_query(self, c_su, radius: float) -> list[int]:
        rho0 = 2.0 * math.asinh(abs(c_su[1])); z0 = self._origin(c_su)
        return [i for i in self._iter_candidates(rho0, math.atan2(z0[1], z0[0]) % math.tau, radius, math.sinh(0.5 * (radius + self.tol))) if self._dist(c_su, self.items[i]) <= radius]


_KERNELS = {"graphcombinatorial": "GraphCombinatorial", "graph": "GraphCombinatorial", "combinatorial": "GraphCombinatorial",
            "dunhamhyperboloid": "DunhamHyperboloid", "dunham": "DunhamHyperboloid", "hyperboloid": "DunhamHyperboloid",
            "spatialhashpoincare": "SpatialHashPoincare", "spatialhash": "SpatialHashPoincare", "poincare": "SpatialHashPoincare"}


class H2LargePatch:
    """Fast O(N) hyperbolic tiling wrapper backed by Rust or pure-Python spatial-hash fallback.

    The pure-Python fallback always grows the patch with the spatial-hash rule (all three kernels build
    the same combinatorial vertex-completion patch) and additionally exposes `faces`, `poincare_pts`,
    `face_neighbors`, `face_shell` and `face_centres_poincare` as attributes; use the methods for code
    that must run in both modes.
    """
    def __init__(self, p: int = 7, q: int = 3, target_faces: int = 100, kernel: str = "SpatialHashPoincare", _rs_obj=None):
        self._rs = _rs_obj
        if self._rs is not None: return
        if not vc.is_hyperbolic(p, q): raise ValueError(f"{{{p}, {q}}} is not hyperbolic (requires (p-2)(q-2) > 4)")
        if (kname := _KERNELS.get(str(kernel).strip().lower())) is None: raise ValueError(f"unknown H2 kernel {kernel!r}")
        self._p, self._q, self._kernel = p, q, kname
        base = vc.BaseFace(p, q); fhash, vhash = _PolarHashPy(2.0 * base.inradius), _PolarHashPy(base.edge_length()); r_v = math.tanh(0.5 * base.circumradius)
        v_su0 = [(complex(1.0 / math.sqrt(1 - r_v * r_v)), complex(r_v * math.cos(2 * math.pi * k / p), r_v * math.sin(2 * math.pi * k / p)) / math.sqrt(1 - r_v * r_v)) for k in range(p)]
        ch, sh, rp1 = math.cosh(base.inradius), math.sinh(base.inradius), (complex(math.cos(math.pi / p), math.sin(math.pi / p)), 0j)
        star = []
        for j in range(p):
            sj = fhash._mul((1j * ch, -1j * sh * complex(math.cos(2 * math.pi * (j + 0.5) / p), math.sin(2 * math.pi * (j + 0.5) / p))), rp1)
            cur, ring = (1 + 0j, 0j), []
            for _ in range(1, q): cur = fhash._mul(cur, sj); ring.append(cur)
            star.append(ring)
        fhash.insert((1 + 0j, 0j)); self.face_shell, self.faces, self.vertex_faces, comp = [0], [], [], []
        def add_face(h_su, sh_i):
            nf = len(self.faces); cyc = []
            for vs in v_su0:
                vi, nw = vhash.insert(fhash._mul(h_su, vs))
                if nw: comp.append(False); self.vertex_faces.append([])
                self.vertex_faces[vi].append(nf); cyc.append(vi)
            self.face_shell.append(sh_i) if nf > 0 else None; self.faces.append(cyc)
        add_face((1 + 0j, 0j), 0); cur_i, sh_i, sh_end = 0, 1, 1
        while len(self.faces) < target_faces and cur_i < len(fhash.items):
            if cur_i >= sh_end: sh_i += 1; sh_end = len(fhash.items)
            fi = cur_i; cur_i += 1; g = fhash.items[fi]
            for j in range(p):
                if comp[self.faces[fi][j]]: continue
                comp[self.faces[fi][j]] = True
                for sm in star[j]:
                    if len(self.faces) >= target_faces: break
                    h = fhash._mul(g, sm); _, ins = fhash.insert(h)
                    if ins: add_face(h, sh_i)
        self._fhash, self.face_centres_poincare, self.poincare_pts = fhash, list(fhash.z_pts), list(vhash.z_pts)
        self.face_neighbors = [[None] * p for _ in self.faces]; emap, ne = {}, 0
        for fi, f in enumerate(self.faces):
            for s in range(p):
                k = (min(f[s], f[(s + 1) % p]), max(f[s], f[(s + 1) % p]))
                if k in emap: of, os = emap[k]; self.face_neighbors[fi][s] = of; self.face_neighbors[of][os] = fi
                else: emap[k] = (fi, s); ne += 1
        self.active = [True] * len(self.faces); self._na, self._nv, self._ne = len(self.faces), len(self.poincare_pts), ne

    p = property(lambda self: self._rs.p if self._rs else self._p)
    q = property(lambda self: self._rs.q if self._rs else self._q)
    kernel = property(lambda self: self._rs.kernel if self._rs else self._kernel)
    total_faces = property(lambda self: self._rs.total_faces if self._rs else len(self.faces))
    n_active_faces = property(lambda self: self._rs.n_active_faces if self._rs else self._na)
    n_vertices = property(lambda self: self._rs.n_vertices if self._rs else self._nv)
    n_edges = property(lambda self: self._rs.n_edges if self._rs else self._ne)
    def summary(self): return json.loads(self._rs.summary_json()) if self._rs else {"p": self._p, "q": self._q, "kernel": self._kernel, "total_faces": len(self.faces), "n_active_faces": self._na, "n_vertices": self._nv, "n_edges": self._ne, "max_shell": max(self.face_shell or [0])}
    def face_vertices(self, fid: int): return list(self._rs.face_vertices(int(fid))) if self._rs else list(self.faces[fid])
    def face_centre_poincare(self, fid: int): return list(self._rs.face_centre_poincare(int(fid))) if self._rs else list(self.face_centres_poincare[fid])
    def hyperbolic_ball_query(self, center, radius: float): return list(self._rs.hyperbolic_ball_query(list(center), float(radius))) if self._rs else sorted(fid for fid in self._fhash.ball_query(_su_trans(center), max(0.0, radius)) if self.active[fid])
    def edge_neighbors(self, fid: int): return list(self._rs.edge_neighbors(int(fid))) if self._rs else ([n for n in self.face_neighbors[fid] if n is not None and self.active[n]] if 0 <= fid < len(self.faces) and self.active[fid] else [])
    def vertex_neighbors(self, fid: int): return list(self._rs.vertex_neighbors(int(fid))) if self._rs else (sorted({f for vid in self.faces[fid] for f in self.vertex_faces[vid] if f != fid and self.active[f]}) if 0 <= fid < len(self.faces) and self.active[fid] else [])
    def remove_face(self, fid: int) -> bool:
        if self._rs: return bool(self._rs.remove_face(int(fid)))
        if 0 <= fid < len(self.active) and self.active[fid]: self.active[fid] = False; self._na -= 1; return True
        return False
    def restore_face(self, fid: int) -> bool:
        if self._rs: return bool(self._rs.restore_face(int(fid)))
        if 0 <= fid < len(self.active) and not self.active[fid]: self.active[fid] = True; self._na += 1; return True
        return False
    def add_face_across_edge(self, fid: int, side: int) -> int:
        if self._rs: return int(self._rs.add_face_across_edge(int(fid), int(side)))
        if (nb := self.face_neighbors[fid][side]) is not None:
            if not self.active[nb]: self.active[nb] = True; self._na += 1; return nb
            raise ValueError("edge already occupied")
        u, v = self.faces[fid][side], self.faces[fid][(side + 1) % self._p]; tm = _su_trans(geodesic_arc_poincare(self.poincare_pts[u], self.poincare_pts[v], 2)[1])
        ht = self._fhash._mul(self._fhash._mul(tm, (1j, 0j)), (tm[0].conjugate(), -tm[1])); nc_su = self._fhash._mul(ht, self._fhash.items[fid]); cyc = [v, u]
        # The half-turn swaps u, v and keeps orientation: new face reads v, u, H(f[side+2]), ..., H(f[side-1]).
        for st in range(1, self._p - 1):
            nvid = len(self.poincare_pts); self.poincare_pts.append(self._fhash._origin(self._fhash._mul(ht, _su_trans(self.poincare_pts[self.faces[fid][(side + 1 + st) % self._p]])))); self.vertex_faces.append([]); cyc.append(nvid)
        nfid = len(self.faces); [self.vertex_faces[x].append(nfid) for x in cyc]
        self.face_neighbors[fid][side] = nfid; self.faces.append(cyc); self.face_shell.append(self.face_shell[fid] + 1)
        self.face_centres_poincare.append(self._fhash._origin(nc_su)); self.face_neighbors.append([fid] + [None] * (self._p - 1))
        self.active.append(True); self._na += 1; self._nv = len(self.poincare_pts); self._ne += self._p - 1; self._fhash.insert(nc_su); return nfid
    def filter_by_region(self, kind: str, p1: float = 0.0, p2: float = 0.0, p3: float = 0.0) -> int:
        if self._rs: return int(self._rs.filter_by_region(kind, float(p1), float(p2), float(p3)))
        preds = {"poincareradiusmax": lambda fid, z: z[0] ** 2 + z[1] ** 2 <= p1 * p1, "hyperbolicradiusmax": lambda fid, z: 2.0 * math.atanh(min(0.999999999, math.hypot(*z))) <= p1,
                 "shellrange": lambda fid, z: int(p1) <= self.face_shell[fid] <= int(p2), "diskball": lambda fid, z: hyp_distance_poincare(z, [p1, p2]) <= p3}
        pred = preds.get(kind.lower(), lambda fid, z: p1 * z[0] + p2 * z[1] >= p3)
        self.active = [pred(fid, z) for fid, z in enumerate(self.face_centres_poincare)]; self._na = sum(self.active); return self._na


def build_hyperbolic_large(p: int = 7, q: int = 3, target_faces: int = 100, kernel: str = "SpatialHashPoincare") -> H2LargePatch:
    if (cls := _rs_fn("H2LargePatch")) is not None: return H2LargePatch(_rs_obj=cls(int(p), int(q), int(target_faces), kernel))
    return H2LargePatch(int(p), int(q), int(target_faces), kernel)
