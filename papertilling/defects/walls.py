"""Walls and the other mirror/glide-breaking defects (the author's four kinds, 2026-11-13: "we will see
which is correct"). Used by DefectCatalogue; every function takes the catalogue `cat`.

(a) domain_walls   L | sigma(L), sigma in the Euclidean normaliser N(G) but not in G. Then sigma(L) has
                   the same group (sigma G sigma^-1 = G) and is a different domain. Classes: N(G)/G.
(b) twin_walls     reflection walls (legacy `wall`, pure reflection) on lattice lines, all offsets.
(c) mobius_seams   one per mirror/glide class (legacy `mobius_strip`); the sheet is non-orientable.
(d) glide_breaking pure reflection on a glide line (and reflection + half-period glide on a mirror line).

NORMALISER, computed directly (independent of legacy catalogue().domains, which tries only holohedry
matrices placed at anchor points). Derivation (proved result, formal verification not performed):
  sigma = (M, s) normalises G  =>  sigma T sigma^-1 = T, so M is in the holohedry of the lattice (integer
  in the primitive basis, metric-preserving) and M P M^-1 = P. For each op g = (R, t) of G,
  sigma g sigma^-1 = (R', M t + (I - R') s) with R' = M R M^-1, which is in G iff
      (I - R') s = t_G(R') - M t   (mod lattice),                                    (*)
  t_G(R') being G's translation for the linear part R'. If P has a rotation, I - R' is invertible for
  it and (*) has finitely many solutions s mod the lattice (enumerated, then checked against all ops).
  If P = {1, m} (pm, pg, cm) the solutions are lines (free along the mirror): CONTINUOUS. If P = {1}
  (p1) every s works: CONTINUOUS. Then only sampled representatives are listed (complete=False).
  Cosets: sigma1 ~ sigma2 iff sigma1^-1 sigma2 in G.
  Check (two routes): (i) group-level sigma g sigma^-1 in G for all g; (ii) geometric: the conjugate maps
  the lattice onto itself and sigma does not (legacy Perfect.is_symmetry).
ITA comparison: ITA_GENERIC_INDEX below = [N_E(G) : G] for a generic metric, from the Euclidean
normalisers listed in ITA Vol. A, Part 15, Table 15.2.1.3 (locator as given in the brief; the table was
NOT opened in this session -- the values are recalled and re-derived by hand: index = |N/T_N| x
[T_N : T] / |P|, e.g. p3: N = p6m with a' = (2a+b)/3, so 12 x 3 / 3 = 12). A larger computed index
means a specialised metric (e.g. the cmm brick with a square conventional cell).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import networkx as nx

from ..legacy.lab_defects import (wall as _ld_wall, mobius_strip as _ld_mobius, interface as _ld_interface,
                                  reflection_across, describe, perfect_patch, is_orientable)
from .catalogue import DefectEntry, _orbits, _rot90, _canon_dir

ITA_GENERIC_INDEX = {"p1": None, "p2": 4, "pm": None, "pg": None, "cm": None, "pmm": 4, "pmg": 4, "pgg": 4,
                     "cmm": 2, "p4": 4, "p4m": 2, "p4g": 2, "p3": 12, "p3m1": 6, "p31m": 2, "p6": 2, "p6m": 1}
_I = np.eye(2, dtype=int)


def _key(M):
    return tuple(int(x) for x in np.asarray(M).ravel())


def _is_int(v, tol=1e-5):
    v = np.asarray(v, float)
    return bool(np.all(np.abs(v - np.round(v)) < tol))


def _inv(g):
    return g[0].T, -g[0].T @ g[1]


def _mul(a, b):
    return a[0] @ b[0], a[0] @ b[1] + a[1]


# ----------------------------------------------------------------------------------------------
# (a) Euclidean normaliser and N(G)/G
# ----------------------------------------------------------------------------------------------
def _holohedry(grp):
    Gm = grp.A.T @ grp.A
    out = []
    for e in itertools.product(range(-2, 3), repeat=4):
        M = np.array(e, int).reshape(2, 2)
        if abs(round(np.linalg.det(M))) == 1 and np.allclose(M.T @ Gm @ M, Gm, atol=1e-7 * np.abs(Gm).max()):
            out.append(M)
    return out


def _kernel_and_complement(D):
    r = D[0] if np.any(D[0]) else D[1]
    k = np.array([r[1], -r[0]], int)
    k //= math.gcd(int(abs(k[0])), int(abs(k[1])))
    for w in itertools.product(range(-3, 4), repeat=2):          # k x w = 1
        if k[0] * w[1] - k[1] * w[0] == 1:
            return k.astype(float), np.array(w, float)
    raise RuntimeError("no complement")


def normaliser_cosets(cat):
    g = cat.grp
    Pf = {_key(M): t for M, t in g.opsf}
    norm_M = []
    for M in _holohedry(g):
        Mi = np.round(np.linalg.inv(M)).astype(int)
        if all(_key(M @ R @ Mi) in Pf for R, _ in g.opsf):
            norm_M.append((M, Mi))
    rot = [(R, t) for R, t in g.opsf if round(np.linalg.det(R)) == 1 and not np.array_equal(R, _I)]
    ref = [(R, t) for R, t in g.opsf if round(np.linalg.det(R)) == -1]
    cont = 0 if rot else (1 if ref else 2)
    cands = []
    for M, Mi in norm_M:
        eqs = [(M @ R @ Mi, Pf[_key(M @ R @ Mi)] - M @ t) for R, t in g.opsf]
        sols = []
        if rot:
            R, t = rot[0]; Rp = M @ R @ Mi; c = Pf[_key(Rp)] - M @ t
            sols = [np.linalg.solve(_I - Rp, c + np.array(n, float)) for n in itertools.product(range(4), repeat=2)]
        elif ref:
            R, t = ref[0]; Rp = M @ R @ Mi; c = Pf[_key(Rp)] - M @ t
            k, w = _kernel_and_complement(_I - Rp)
            d = (_I - Rp) @ w
            lams = []
            for n in itertools.product(range(-6, 7), repeat=2):
                v = c + np.array(n, float)
                if abs(v[0] * d[1] - v[1] * d[0]) < 1e-7:
                    lam = float(v @ d / (d @ d)) % 1.0
                    if not any(abs(lam - l2) < 1e-7 or abs(abs(lam - l2) - 1) < 1e-7 for l2 in lams):
                        lams.append(lam)
            sols = [lam * w + mu * k for lam in lams for mu in (0.0, 0.5)]        # mu: free, sampled
        else:
            sols = [np.array(x, float) for x in ((0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5))]   # free, sampled
        for s in sols:
            s = s - np.floor(s + 1e-7)
            if all(_is_int((_I - Rp) @ s - c) for Rp, c in eqs) and \
                    not any(np.array_equal(M, M2) and _is_int(s - s2) for M2, s2 in cands):
                cands.append((M, s))

    def same_coset(a, b):
        Mai = np.round(np.linalg.inv(a[0])).astype(int)
        kk = _key(Mai @ b[0])
        return kk in Pf and _is_int(Mai @ (b[1] - a[1]) - Pf[kk])
    cosets = []
    for c in cands:
        for cl in cosets:
            if same_coset(cl[0], c):
                cl.append(c); break
        else:
            cosets.append([c])
    ident = (_I, np.zeros(2))
    A, Ai = g.A, g.Ainv
    recs = []
    for cl in cosets:
        if same_coset(ident, cl[0]):
            continue
        M, s = cl[0]
        best = None
        for R, t in g.opsf:                                        # every element (M R, M t + s + v)
            for v in itertools.product(range(-2, 3), repeat=2):
                Mf = M @ R; sf = M @ t + s + np.array(v, float)
                Mc, tc = A @ Mf @ Ai, A @ sf
                if np.array_equal(Mf, _I):
                    sc = (0, round(float(np.linalg.norm(tc)), 6), 0.0)
                elif round(np.linalg.det(Mf)) == 1:
                    ang = abs(math.atan2(Mc[1, 0], Mc[0, 0]))
                    cen = np.linalg.solve(np.eye(2) - Mc, tc)
                    sc = (1, round(ang, 6), round(float(np.linalg.norm(cen)), 6))
                else:
                    wv, V = np.linalg.eigh((Mc + Mc.T) / 2)
                    u = V[:, int(np.argmax(wv))]
                    sc = (2, round(abs(float(tc @ u)), 6), round(abs(float(tc @ _rot90(u))), 6))
                if best is None or sc < best[0]:
                    best = (sc, Mf, sf, (Mc, tc))
        sc, Mf, sf, sig = best
        kind = ("translation", "rotation", "mirror")[sc[0]]
        recs.append(dict(kind=kind, sigma=sig, sigma_frac=(Mf, sf), score=sc, what=describe(sig)))
    recs.sort(key=lambda r: r["score"])
    for k, r in enumerate(recs):
        r["id"] = f"N{k + 1}"
        sig = r["sigma"]
        conj = [_mul(_mul(sig, h), _inv(sig)) for h in g.ops]
        r["check_group"] = all(g.contains(h) for h in conj)
        r["check_geometry"] = all(g.L.is_symmetry(h) for h in conj) and not g.L.is_symmetry(sig)
        cm = []
        for c in cat.centre_classes:
            tgt = cat.centre_class_of(sig[0] @ c["point"] + sig[1])
            cm.append(f"{c['label']}->{tgt['label'] if tgt else '?'}")
        r["centre_map"] = ", ".join(cm)
        r["sites_to"] = cat.location(sig[0] @ cat.cell.basis[0] + sig[1])
    ita = ITA_GENERIC_INDEX.get(cat.group)
    index = len(cosets)
    if cont:
        note = (f"CONTINUOUS normaliser ({cont} free translation direction(s)): the representatives are samples; "
                "the list is NOT complete")
    elif ita is None or index == ita:
        note = "index equals ITA's generic value"
    elif index > ita:
        note = f"index {index} > ITA generic {ita}: specialised metric (larger normaliser)"
    else:
        note = f"DISAGREEMENT: index {index} < ITA generic {ita}"
    return dict(index=index, cosets=recs, continuous=cont, complete=(cont == 0), ita_generic_index=ita,
                note=note, holohedry_order=len(_holohedry(g)), point_group_order=len(g.opsf))


def coset_of(cat, sigma):
    """'G' if sigma in G; the id of its N(G)/G coset if listed; 'N(G), unlisted' or None (not in N(G))."""
    g = cat.grp
    if g.contains(sigma):
        return "G"
    for r in cat.normaliser["cosets"]:
        if g.contains(_mul(_inv(r["sigma"]), sigma)):
            return r["id"]
    if all(g.contains(_mul(_mul(sigma, h), _inv(sigma))) for h in g.ops):
        return "N(G), unlisted"
    return None


def _default_wall_line(cat):
    P = cat.cell
    return P.basis.mean(axis=0) + 0.137 * P.a2, P.a1 / np.linalg.norm(P.a1)


def domain_walls(cat):
    out = []
    pt, u = _default_wall_line(cat)
    for r in cat.normaliser["cosets"]:
        sig = r["sigma"]
        spec = dict(kind="wall", wall_type="a", sigma=(sig[0].tolist(), sig[1].tolist()), point=tuple(pt), direction=tuple(u))
        out.append(DefectEntry(lambda R=None, sig=sig, **kw: _ld_interface(cat.cell, sig, pt, u, R=R or cat.R0),
                               kind="domain_wall", id=f"Wa{len(out) + 1}", coset=r["id"], type=r["kind"],
                               label=f"Wa{len(out) + 1} ({r['kind']}): {r['what']}", sigma=sig, what=r["what"],
                               centre_map=r["centre_map"], sites_to=r["sites_to"],
                               complete=cat.normaliser["complete"], spec=spec))
    return out


# ----------------------------------------------------------------------------------------------
# (b) reflection (twin) walls on lattice lines
# ----------------------------------------------------------------------------------------------
def _rooted_hash(G, v, r):
    H = nx.ego_graph(G, v, radius=r)
    nx.set_node_attributes(H, {w: ("1" if w == v else "0") for w in H}, "root")
    return nx.weisfeiler_lehman_graph_hash(H, node_attr="root", iterations=r + 1)


def _perfect_hashes(cat, r):
    """Rooted r-ball hashes of the perfect lattice: one per site of cell 0 (every other site is a
    translate of one of these, so this is the full set)."""
    key = ("_perfect_hashes", r)
    if key not in cat.__dict__:
        g = cat.grp
        d = perfect_patch(cat.cell, (r + 2) * g.lmax + 2 * g.scale)
        G = d.to_networkx()
        cell0 = [i for i, k in enumerate(d.sites) if k[0] == 0 and k[1] == 0]
        cat.__dict__[key] = {_rooted_hash(G, i, r) for i in cell0}
    return cat.__dict__[key]


def network_irregular(cat, d, x0, u, period, r=4):
    """Sites near the wall whose bond count or rooted r-bond neighbourhood (WL hash) differs from every
    site of the perfect lattice. Empty list = the wall is invisible in the bond network (C02 definition
    A7). Where to look (proved, from the wall rule): only bonds 'across' the line and sites on the line
    join the two sides, so a site farther than r bonds from those has a ball lying on one side, i.e.
    a ball of the lattice or of its mirror image, both perfect as graphs. The wall is periodic along
    the line with the lattice period, so one period of seeds suffices.
    Hash equality is computational evidence; a hash or degree difference proves the ball differs."""
    g = cat.grp
    perfect = _perfect_hashes(cat, r)
    G = d.to_networkx(); P = d.positions
    uh = np.asarray(u, float) / np.linalg.norm(u); n = _rot90(uh)
    seeds = {a for (a, b), e in d.edge.items() if e.get("across")}
    seeds |= {i for i in range(len(P)) if abs((P[i] - x0) @ n) < 1e-7}
    seeds = [i for i in seeds if abs((P[i] - x0) @ uh) <= 0.5 * period + 1e-6]
    band = set()
    for i in seeds:
        band |= set(nx.single_source_shortest_path_length(G, i, cutoff=r))
    Rd = max(np.linalg.norm(P - x0, axis=1))
    if any(np.linalg.norm(P[i] - x0) > Rd - (r + 1) * g.lmax for i in band):
        raise RuntimeError("wall patch too small for the neighbourhood test; increase R")
    bad = []
    for i in sorted(band):
        key = d.sites[i][0]
        if G.degree(i) != len(g.L.rule[key[2]]) or _rooted_hash(G, i, r) not in perfect:
            bad.append(i)
    return bad


def _wall_dirs(cat):
    g = cat.grp
    vecs = sorted(cat._primitive_vectors(cat.wall_bmax), key=lambda m: np.linalg.norm(g.A @ np.array(m, float)))
    can = lambda m: tuple(int(x) for x in np.round(g.Ainv @ _canon_dir(g.A @ np.array(m, float)) * np.linalg.norm(g.A @ np.array(m, float))))
    vecs = list(dict.fromkeys(can(m) for m in vecs))
    orbs = _orbits(vecs, [Mf for Mf, _ in g.opsf], lambda m, Mf: can(Mf @ np.array(m)), lambda a, b: a == b)
    ang = lambda m: math.atan2(*(_canon_dir(g.A @ np.array(m, float)))[::-1])
    return [min((vecs[i] for i in o), key=lambda m: (round(float(np.linalg.norm(g.A @ np.array(m, float))), 6), round(ang(m), 6)))
            for o in orbs]


def _twin_setup(cat):
    """Per direction: the intervals of offsets (through a site / gap) and the classes of oriented walls."""
    if "_twin" in cat.__dict__:
        return cat.__dict__["_twin"]
    g = cat.grp
    dirs = []
    for m in _wall_dirs(cat):
        u = _canon_dir(g.A @ np.array(m, float)); n = _rot90(u)
        per, dl = cat._line_data(u)
        hs = []
        for x in sorted(0.0 if dl - z < 1e-9 else z for z in (float(x) % dl for x in cat.cell.basis @ n)):
            if not any(min(abs(x - y), dl - abs(x - y)) < 1e-7 for y in hs):
                hs.append(x)
        ivs = [("site", h, h) for h in hs] + [("gap", hs[i], hs[i + 1] if i + 1 < len(hs) else hs[0] + dl) for i in range(len(hs))]
        mir = [ln["offset"] for ln in cat.all_lines if ln["glide"] == 0.0 and np.allclose(ln["direction"], u, atol=1e-6)]
        dirs.append(dict(m=m, u=u, n=n, per=per, dl=dl, ivs=ivs, mirrors=mir))
    cands = [(k, o, i) for k, D in enumerate(dirs) for o in (1, -1) for i in range(len(D["ivs"]))]

    def where(k, c):
        D = dirs[k]; c = c % D["dl"]
        for i, (kind, lo, hi) in enumerate(D["ivs"]):
            if kind == "site" and min(abs(c - lo), D["dl"] - abs(c - lo)) < 1e-6:
                return i
        for i, (kind, lo, hi) in enumerate(D["ivs"]):
            if kind == "gap" and (lo < c < hi or lo < c + D["dl"] < hi):
                return i
        raise RuntimeError("offset not placed")

    def image(cand, h):
        k, o, i = cand
        D = dirs[k]
        kind, lo, hi = D["ivs"][i]
        c = lo if kind == "site" else 0.5 * (lo + hi)
        d = h[0] @ (o * D["u"])
        if np.linalg.det(h[0]) < 0:
            d = -d                                   # an improper map swaps left and right
        p = h[0] @ (c * D["n"]) + h[1]
        for k2, D2 in enumerate(dirs):
            for o2 in (1, -1):
                if np.allclose(d, o2 * D2["u"], atol=1e-6):
                    return (k2, o2, where(k2, float(p @ D2["n"])))
        return None
    orbs = _orbits(cands, g.ops, image, lambda a, b: a == b)
    cls = {}
    for o in orbs:
        mem = sorted((cands[i] for i in o), key=lambda x: (x[0], -x[1], x[2]))
        cls.setdefault(mem[0][0], []).append((mem[0], mem))
    cat.__dict__["_twin"] = (dirs, cls, where, image)
    return cat.__dict__["_twin"]


def twin_class_of(cat, point, direction):
    """Id of the twin wall (b) built on the line through `point` along `direction` (pure reflection,
    side A on the left), or None if that direction is not among the listed ones."""
    dirs, cls, where, image = _twin_setup(cat)
    u = np.asarray(direction, float) / np.linalg.norm(direction)
    for h in cat.grp.ops:
        d = h[0] @ u * (-1 if np.linalg.det(h[0]) < 0 else 1)
        p = h[0] @ np.asarray(point, float) + h[1]
        for k, D in enumerate(dirs):
            for o in (1, -1):
                if np.allclose(d, o * D["u"], atol=1e-6):
                    c = (k, o, where(k, float(p @ D["n"])))
                    for e in cat.twin_walls:
                        if c in e["members"]:
                            return e["id"]
    return None


def twin_walls(cat, r=4):
    dirs, cls, where, image = _twin_setup(cat)
    g = cat.grp
    out = []
    for k, D in enumerate(dirs):
        for (rep, mem) in sorted(cls.get(k, []), key=lambda x: (-x[0][1], D["ivs"][x[0][2]][0] != "site", D["ivs"][x[0][2]][1])):
            _, o, i = rep
            kind, lo, hi = D["ivs"][i]
            if kind == "site":
                c = lo
            else:
                inside = [x for x in D["mirrors"] if any(lo + 1e-6 < y < hi - 1e-6 for y in (x, x + D["dl"]))]
                c = inside[0] if inside else 0.5 * (lo + hi)
            c = c % D["dl"]
            p, u = c * D["n"], o * D["u"]
            sig = reflection_across(p, u, 0.0)
            Rw = 0.5 * D["per"] + (2 * r + 3) * g.lmax + g.scale
            d = _ld_wall(cat.cell, p, u, 0.0, R=Rw)
            in_g = bool(d.info["invisible"])
            bad = [] if in_g else network_irregular(cat, d, p, u, D["per"], r)
            eid = f"Wb{len(out) + 1}"
            where_s = f"through a site, offset {lo:.4f}" if kind == "site" else f"gap {lo:.4f} < offset < {hi:.4f}"
            vis = "invisible (mirror line)" if in_g else ("invisible in the bond network" if not bad else "visible")
            spec = dict(kind="wall", wall_type="b", sigma=(sig[0].tolist(), sig[1].tolist()), point=tuple(p), direction=tuple(u))
            out.append(DefectEntry(lambda R=None, p=p, u=u, **kw: _ld_wall(cat.cell, p, u, 0.0, R=R or cat.R0),
                                   kind="twin_wall", id=eid,
                                   label=f"{eid}: along {tuple(int(x) * o for x in D['m'])}, {where_s}: {vis}",
                                   direction=u, direction_frac=tuple(int(x) * o for x in D["m"]), point=p, offset=c,
                                   spacing=D["dl"], interval=(kind, lo, hi), through_site=(kind == "site"),
                                   members=mem, class_size=len(mem), sigma=sig, sigma_in_G=in_g,
                                   network_invisible=(not bad), n_irregular=len(bad), visible=(not in_g and bool(bad)),
                                   domain_coset=coset_of(cat, sig), spec=spec))
    return out


# ----------------------------------------------------------------------------------------------
# (c) Moebius seams, (d) glide-breaking walls
# ----------------------------------------------------------------------------------------------
def mobius_seams(cat, strip_length=4):
    out = []
    w = 1.6 * cat.grp.scale
    for ln in cat.mirrors:
        L = strip_length if ln["kind"] == "mirror" else max(1, strip_length - 1)
        sig = ln["sigma"]
        spec = dict(kind="wall", wall_type="c", sigma=(sig[0].tolist(), sig[1].tolist()), point=tuple(ln["point"]),
                    direction=tuple(ln["direction"]), length=L, width=w)
        out.append(DefectEntry(lambda R=None, sig=sig, L=L, **kw: _ld_mobius(cat.cell, sig, length=L, width=w),
                               kind="mobius_seam", id=f"Wc{len(out) + 1}", line=ln["id"], line_kind=ln["kind"],
                               label=f"Wc{len(out) + 1}: Moebius seam on {ln['id']} ({ln['kind']})",
                               sigma=sig, non_orientable=True, spec=spec))
    return out


def glide_breaking(cat):
    out = []
    for ln in cat.mirrors:
        p, u, per = ln["point"], ln["direction"], ln["period"]
        gl = 0.0 if ln["kind"] == "glide" else 0.5 * per
        sig = reflection_across(p, u, gl)
        shift = _mul(sig, _inv(ln["sigma"]))[1]                # sigma = (shift) o g, g the symmetry of the line
        f = float(shift @ u) % per
        mis = min(f, per - f)
        tw = twin_class_of(cat, p, u) if gl == 0.0 else None
        what = ("pure reflection on a glide line" if gl == 0.0 else "reflection + half-period glide on a mirror line")
        spec = dict(kind="wall", wall_type="d", sigma=(sig[0].tolist(), sig[1].tolist()), point=tuple(p), direction=tuple(u))
        out.append(DefectEntry(lambda R=None, p=p, u=u, gl=gl, **kw: _ld_wall(cat.cell, p, u, gl, R=R or cat.R0),
                               kind="glide_breaking", id=f"Wd{len(out) + 1}", line=ln["id"], line_kind=ln["kind"],
                               label=f"Wd{len(out) + 1}: {what} ({ln['id']}), slip {mis:.4g} along the wall",
                               sigma=sig, slip=mis, slip_over_period=mis / per, glide_of_line=ln["glide"],
                               same_domains_as=coset_of(cat, sig), same_as_twin=tw, spec=spec))
    return out


# ----------------------------------------------------------------------------------------------
# Legacy comparison and the printed summary
# ----------------------------------------------------------------------------------------------
def compare_legacy(cat, show=True):
    """Compare with legacy lab_defects.catalogue (slow: 1-7 s). Returns a dict of findings."""
    from ..legacy.lab_defects import catalogue as _legacy
    leg = _legacy(cat.cell, bmax=1.0)
    g = cat.grp
    mine_lines = [(ln["kind"], ln["size"]) for ln in cat.mirrors]
    leg_lines = [(ln["kind"], ln["size"]) for ln in leg.lines]
    gclasses = {c["gclass"] for c in cat.centre_classes}
    dom = [coset_of(cat, d["sigma"]) for d in leg.domains]
    res = dict(lines_mine=sorted(mine_lines), lines_legacy=sorted(leg_lines), lines_agree=sorted(mine_lines) == sorted(leg_lines),
               centre_G_classes_mine=len(gclasses), centre_G_classes_legacy=len(leg.centres),
               legacy_domains=len(leg.domains), legacy_domain_cosets=dom, normaliser_index=cat.normaliser["index"],
               legacy_domains_all_in_N=all(x not in (None, "G") for x in dom),
               legacy_domains_distinct=len(set(dom)) == len(dom),
               legacy_domains_complete=len(set(dom)) == cat.normaliser["index"] - 1)
    if show:
        for k, v in res.items():
            print(f"  {k}: {v}")
    return res


def _fmt(v, n=4):
    return "(" + ", ".join(f"{float(x):.{n}g}" for x in v) + ")"


def summary_text(cat):
    L = []
    g = cat.grp
    L.append(f"== {cat.label}: group {cat.group}, |P| = {len(g.opsf)}, {'chiral' if cat.chiral else 'achiral'}; "
             f"primitive cell a1 = {_fmt(cat.cell.a1)}, a2 = {_fmt(cat.cell.a2)}, {len(cat.cell.basis)} site(s)/cell")
    ds = cat.dislocations
    L.append(f"-- Dislocations: {len(ds)} P+-classes, {sum(d['size'] for d in ds)} primitive Burgers vectors "
             f"with |b| <= {cat.bmax:g} |b_min| (|b_min| = {cat.b_min:.4g})")
    L.append(f"   {'id':4s} {'b (cell)':10s} {'b (Cartesian)':18s} {'|b|':>7s} {'angle':>7s} {'size':>4s} {'-b':>4s} {'mirror':>6s} b~-b")
    for d in ds:
        L.append(f"   {d['id']:4s} {str(d['b_frac']):10s} {_fmt(d['b']):18s} {d['length']:7.4f} {d['angle']:7.2f} "
                 f"{d['size']:4d} {d['inverse_id']:>4s} {str(d['mirror_id']):>6s} {'yes' if d['self_inverse'] else 'no'}")
    cc = cat.centre_classes
    L.append(f"-- Rotation-centre G+-classes: {len(cc)}")
    for c in cc:
        L.append(f"   {c['id']:4s} {c['label']:18s} order {c['order']}  point {_fmt(c['point'])}  frac {_fmt(c['frac'])}  "
                 f"{c['location']:13s} per cell {c['size']}  mirror -> {c['mirror_of']}")
    dl = cat.disclinations
    L.append(f"-- Disclinations: {len(dl)} classes (angle in [-180, 180], +180 != -180)")
    for d in dl:
        H = d["H"]
        L.append(f"   {d['id']:26s} {d['location']:13s} H = (R({d['angle']:+.0f}), {_fmt(H[1])})  mirror -> {d['mirror_id']}")
    ms = cat.mirrors
    L.append(f"-- Mirror and glide lines: {sum(m['kind'] == 'mirror' for m in ms)} mirror + {sum(m['kind'] == 'glide' for m in ms)} glide classes")
    for m in ms:
        L.append(f"   {m['id']:3s} {m['kind']:6s} dir {_fmt(m['direction'], 3):16s} offset {m['offset']:.4f}  glide {m['glide']:.4f}  "
                 f"period {m['period']:.4f}  {'through a site' if m['through_site'] else 'no site'}  ({m['size']} per cell)")
    N = cat.normaliser
    L.append(f"-- (a) Domain walls: [N(G):G] = {N['index']} (ITA generic: {N['ita_generic_index']}); {N['note']}")
    for w in cat.domain_walls:
        L.append(f"   {w['id']:4s} {w['type']:11s} {w['what']}; sites -> {w['sites_to']}; centres {w['centre_map']}")
    tw = cat.twin_walls
    L.append(f"-- (b) Twin (reflection) walls: {len(tw)} classes, {sum(not w['visible'] for w in tw)} invisible")
    for w in tw:
        L.append(f"   {w['label']}; domain coset {w['domain_coset']}")
    L.append(f"-- (c) Moebius seams: {len(cat.mobius_seams)} (non-orientable sheet; cannot exist on an orientable sheet)")
    for w in cat.mobius_seams:
        L.append(f"   {w['label']}")
    gb = cat.glide_breaking
    L.append(f"-- (d) Glide-breaking walls: {len(gb)}")
    for w in gb:
        L.append(f"   {w['label']}; same domains as (a): {w['same_domains_as']}; same as (b): {w['same_as_twin']}")
    return "\n".join(L)
