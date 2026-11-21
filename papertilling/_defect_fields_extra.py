"""Elastic fields and motion export of `papertilling.defect_fields`: homogenised moduli with the Barnett-Lothe / Stroh
dislocation field, the anti-plane screw field, Voronoi coordination charge, and motion GIFs (re-exported there)."""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np

from ._defect_fields_util import TWO_PI, _cell_dict, _edge_list, _pos, _rs


_CLASS_STATUS = ("[COMPUTATIONAL EVIDENCE] order-2 / order-4 harmonics of C_iajb e_i e_a e_j e_b compared with a "
                 "1e-8 relative tolerance")


def _classify(C):
    ph = np.arange(16) * TWO_PI / 16
    e = np.stack([np.cos(ph), np.sin(ph)], 1)
    val = np.einsum("iajb,ki,ka,kj,kb->k", C, e, e, e, e)
    a2, b2, a4, b4 = (val @ f(k * ph) for k in (2, 4) for f in (np.cos, np.sin))
    tol = 1e-8 * max(float(np.max(np.abs(val))), 1e-300) * 16
    z2, z4 = math.hypot(a2, b2) <= tol, math.hypot(a4, b4) <= tol
    if z2 or z4:
        return {(True, True): "Isotropic", (True, False): "SquareCubic", (False, True): "Orthotropic"}[(z2, z4)]
    return "Orthotropic" if abs(math.sin(math.atan2(b4, a4) - 2 * math.atan2(b2, a2))) <= 1e-7 else "Monoclinic"


def _py_stroh(C, b, X):
    """Angular-integral dislocation field, same equations as the Rust `stroh_dislocation_field`."""
    ct = lambda x, y: np.einsum("iajb,ka,kb->kij", C, x, y)  # noqa: E731
    mn_of = lambda t: (np.stack([np.cos(t), np.sin(t)], -1), np.stack([-np.sin(t), np.cos(t)], -1))  # noqa: E731
    m, n = mn_of(np.arange(512) * TWO_PI / 512)
    nni, nm, mn = np.linalg.inv(ct(n, n)), ct(n, m), ct(m, n)
    lhs = np.block([[nni.mean(0), -(nni @ nm).mean(0)], [(mn @ nni).mean(0), (ct(m, m) - mn @ nni @ nm).mean(0)]])
    sol = np.linalg.solve(lhs, np.r_[np.asarray(b, float) / TWO_PI, 0.0, 0.0])
    T, c = sol[:2], sol[2:]

    def vprime(t):
        mm_, nn_ = mn_of(t)
        return np.einsum("kij,kj->ki", np.linalg.inv(ct(nn_, nn_)), T - ct(nn_, mm_) @ c)
    r, th = np.hypot(X[:, 0], X[:, 1]), np.arctan2(X[:, 1], X[:, 0])
    xg, wg = np.polynomial.legendre.leggauss(8)
    frac = ((np.arange(64)[:, None] + (xg[None, :] + 1) / 2) / 64).ravel()
    span = (th + math.pi)[:, None]
    vals = vprime((-math.pi + span * frac[None, :]).ravel()).reshape(len(X), -1, 2)
    v = np.einsum("kq,kqj->kj", span * np.tile(wg / 2, 64)[None, :] / 64, vals)
    m0, n0 = mn_of(th)
    grad = (c[None, :, None] * m0[:, None, :] + vprime(th)[:, :, None] * n0[:, None, :]) / r[:, None, None]
    return dict(u=c * np.log(r)[:, None] + v, grad=grad, stress=np.einsum("iajb,kjb->kia", C, grad),
                log_coeff=c.tolist(), traction=T.tolist(), method="Barnett-Lothe angular integral")


def barnett_lothe_stroh(lattice_or_C, b, points=None, *, n_quad: int = 128) -> dict:
    """Homogenised stiffness `C` (Cauchy-Born with Born-Huang sublattice relaxation, as
    `defects.gsv_elastic.cauchy_born_tensor`), its symmetry class, Barnett-Lothe tensor `B`, dipole prelogarithmic
    slope `K = b.B.b / (2 pi)`, and, at `points`, the displacement / gradient / stress of a straight dislocation at the
    origin (branch cut on the negative x axis, jump `+b` counter-clockwise)."""
    b = [float(x) for x in b]
    P = None if points is None else _pos(points)
    is_C = isinstance(lattice_or_C, np.ndarray) and lattice_or_C.shape == (2, 2, 2, 2)
    rs = _rs("barnett_lothe_stroh_json")
    if rs is not None:
        cell = None if is_C else json.dumps(_cell_dict(lattice_or_C)[0])
        cflat = np.asarray(lattice_or_C, float).ravel().tolist() if is_C else None
        out = json.loads(rs.barnett_lothe_stroh_json(cell, cflat, b, None if P is None else P.tolist(), int(n_quad)))
        out["C"], out["B"] = np.array(out["C"]).reshape(2, 2, 2, 2), np.array(out["B"])
        if out.get("field"):
            f = out.pop("field")
            out.update(u=np.array(f["u"]).reshape(-1, 2), grad=np.array(f["grad"]).reshape(-1, 2, 2),
                       stress=np.array(f["stress"]).reshape(-1, 2, 2), log_coeff=f["log_coeff"], traction=f["traction"], method=f["method"])
        return out
    from .defects.gsv_elastic import barnett_lothe_tensor, cauchy_born_tensor
    C = np.asarray(lattice_or_C, float) if is_C else cauchy_born_tensor(_lattice_for_gsv(lattice_or_C))
    B = barnett_lothe_tensor(C, n_quad=n_quad)
    out = dict(C=C, elastic_class=_classify(C), elastic_class_status=_CLASS_STATUS, B=B,
               K=float(np.asarray(b) @ B @ np.asarray(b)) / TWO_PI)
    if P is not None:
        out.update(_py_stroh(C, b, P))
    return out


def _lattice_for_gsv(lattice):
    if isinstance(lattice, dict) and "a1" in lattice:
        from .legacy.wallpaper_lattices import UnitCell
        c, _ = _cell_dict(lattice)
        return UnitCell("custom", "custom", lattice.get("group", "p1"), np.array(c["a1"]), np.array(c["a2"]),
                        np.array(c["basis"]), [(b["i"], b["j"], b["n1"], b["n2"], b["kind"]) for b in c["bonds"]])
    return lattice


def interpolate_crystal_frames(start, end, n_frames: int) -> list:
    """`n_frames` position arrays linearly interpolated from `start` to `end` (both included)."""
    S, E = _pos(start), _pos(end)
    if S.shape != E.shape or n_frames < 2:
        raise ValueError("start and end must match and n_frames >= 2")
    return [S + (E - S) * (k / (n_frames - 1)) for k in range(n_frames)]


def export_motion_gif(frames, edges, path=None, *, width: int = 320, height: int = 240, delay_cs: int = 8) -> bytes:
    """Looping GIF89a of a sequence of position arrays (a relaxation trajectory, `interpolate_crystal_frames`, ...):
    bonds dark grey, sites red, one common scale. Returns the bytes and writes them to `path` if given."""
    F = [_pos(f) for f in frames]
    es = _edge_list(edges) or []
    rs = _rs("export_motion_gif_bytes")
    if rs is not None:
        data = bytes(rs.export_motion_gif_bytes([f.tolist() for f in F], es, int(width), int(height), int(delay_cs)))
    else:
        from PIL import Image, ImageDraw
        allp = np.concatenate(F)
        lo, hi, m = allp.min(0), allp.max(0), 3.0
        s = min((width - 2 * m) / max(hi[0] - lo[0], 1e-9), (height - 2 * m) / max(hi[1] - lo[1], 1e-9))
        imgs = []
        for f in F:
            q = np.stack([m + (f[:, 0] - lo[0]) * s, height - 1 - m - (f[:, 1] - lo[1]) * s], 1)
            im = Image.new("P", (width, height), 0)
            im.putpalette([255, 255, 255, 40, 40, 40, 200, 30, 30])
            dr = ImageDraw.Draw(im)
            for i, j in es:
                dr.line([tuple(q[i]), tuple(q[j])], fill=1)
            for x, y in q:
                dr.rectangle([x - 1, y - 1, x + 1, y + 1], fill=2)
            imgs.append(im)
        buf = io.BytesIO()
        imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:], loop=0, duration=10 * int(delay_cs))
        data = buf.getvalue()
    if path is not None:
        Path(path).write_bytes(data)
    return data


def screw_dislocation_field(K, bz, points) -> dict:
    """Anti-plane (screw) field of a dislocation along z with Burgers vector `bz`, for anti-plane moduli
    `K_ab = C_{a3b3}` (2x2 symmetric positive definite): `u_z = (bz / 2 pi) arg(x + p y)`,
    `p = (-K_12 + i sqrt(det K)) / K_22` (Eshelby, Read & Shockley 1953; Hirth & Lothe 1982 ch. 13). Cut on the negative
    x axis, jump `+bz` counter-clockwise; `K = mu I` gives the Volterra screw `u_z = bz theta / 2 pi`. Returns `uz`,
    `grad` (d u_z / d x_a) and `stress` (sigma_za = K_ab d u_z / d x_b). Refuses a K that is not SPD."""
    K, P = np.asarray(K, float).reshape(2, 2), _pos(points)
    rs = _rs("screw_dislocation_json")
    if rs is not None:
        out = json.loads(rs.screw_dislocation_json(K.tolist(), float(bz), P.tolist()))
        return dict(uz=np.array(out["uz"], float), grad=np.array(out["grad"], float).reshape(-1, 2),
                    stress=np.array(out["stress"], float).reshape(-1, 2))
    det = float(np.linalg.det(K))
    if abs(K[0, 1] - K[1, 0]) > 1e-12 * (abs(K[0, 0]) + abs(K[1, 1])) or K[0, 0] <= 0 or det <= 0:
        raise ValueError("REFUSED: anti-plane moduli K must be symmetric positive definite")
    p = complex(-K[0, 1] / K[1, 1], math.sqrt(det) / K[1, 1])
    z = P[:, 0] + p * P[:, 1]
    if np.any(z == 0):
        raise ValueError("field is singular at the dislocation line")
    s = float(bz) / TWO_PI
    grad = np.stack([s * (1 / z).imag, s * (p / z).imag], 1)
    return dict(uz=s * np.angle(z), grad=grad, stress=grad @ K.T)


def _clip(poly, labels, pi, pj, j):
    d, m = pj - pi, 0.5 * (pi + pj)
    f = (poly - m) @ d
    inside = f <= 1e-14 * float(d @ d)
    out, lab = [], []
    for k in range(len(poly)):
        k1 = (k + 1) % len(poly)
        if inside[k]:
            out.append(poly[k]), lab.append(labels[k])
        if inside[k] != inside[k1]:
            t = f[k] / (f[k] - f[k1])
            out.append(poly[k] + t * (poly[k1] - poly[k])), lab.append(j if inside[k] else labels[k])
    return np.array(out).reshape(-1, 2), lab


def voronoi_coordination(pos, *, z_ideal: int = 6, tol: float = 0.05) -> dict:
    """Voronoi coordination `z` of a bare point pattern and the coordination charge `z_ideal - z` (`6 - z` for a
    triangular crystal: a 5-fold site +1, a 7-fold site -1, a 5-7 dislocation core neutral; Nelson 2002, Bowick &
    Giomi 2009). Voronoi edges shorter than `tol` x the site's nearest-neighbour distance are ignored (cocircular
    quadruples); cells reaching the frame (convex-hull sites) are `rim` and carry no charge. Same algorithm as the
    Rust `voronoi_coordination` (frame clipped by bisectors, nearest first, stopping at twice the cell radius)."""
    P = _pos(pos)
    rs = _rs("voronoi_coordination_json")
    if rs is not None:
        return json.loads(rs.voronoi_coordination_json(P.tolist(), int(z_ideal), float(tol)))
    if len(P) < 3 or not np.all(np.isfinite(P)):
        raise ValueError("REFUSED: need at least 3 sites with finite coordinates")
    lo, hi = P.min(0), P.max(0)
    ext = float(np.max(hi - lo))
    if ext <= 0:
        raise ValueError("REFUSED: all sites coincide")
    flo, fhi = lo - 1e3 * ext, hi + 1e3 * ext
    frame = np.array([[flo[0], flo[1]], [fhi[0], flo[1]], [fhi[0], fhi[1]], [flo[0], fhi[1]]])
    neighbours, rim = [], []
    for i, pi in enumerate(P):
        dist = np.linalg.norm(P - pi, axis=1)
        order = [j for j in np.lexsort((np.arange(len(P)), dist)) if j != i and dist[j] > 0]
        poly, lab = frame.copy(), [-1] * 4
        for j in order:
            if dist[j] > 2 * np.max(np.linalg.norm(poly - pi, axis=1)):
                break
            poly, lab = _clip(poly, lab, pi, P[j], int(j))
        dnn = float(dist[order[0]]) if order else 0.0
        edge_len = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
        neighbours.append(sorted({lb for lb, el in zip(lab, edge_len) if lb >= 0 and el > tol * dnn}))
        rim.append(-1 in lab)
    z = [len(nb) for nb in neighbours]
    charge = [None if r else int(z_ideal) - zi for zi, r in zip(z, rim)]
    return dict(neighbours=neighbours, z=z, rim=rim, charge=charge, total_charge=sum(q for q in charge if q is not None))
