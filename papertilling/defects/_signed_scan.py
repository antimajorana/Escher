"""Single-dislocation patches for the signed, finite core gallery."""
from __future__ import annotations

from functools import partial

import networkx as nx
import numpy as np

from ..legacy import lab_defects as ld
from . import fast as engines
from ._surgery import _insertion_rule


def _signed_patch(cell, b, core, cut, R, *, fast):
    """Build a CCW Burgers circuit of +b for either orientation of the cut ray."""
    b, core, cut = (np.asarray(v, float) for v in (b, core, cut))
    if ld.cross2(cut, b) < 0:
        remove = engines.dislocation_fast if fast else ld.dislocation
        return remove(cell, b, core=core, cut=cut, R=R)

    lattice = engines.perfect(cell) if fast else ld.Perfect(cell)
    frac_b = lattice.Ainv @ b
    if not np.allclose(frac_b, np.round(frac_b), rtol=0, atol=1e-6):
        raise ValueError("b must be a lattice translation")
    direction = cut / np.linalg.norm(cut)
    keys = lattice.sites_in_disk(core, R)
    reference = np.array([lattice.pos(key) for key in keys], float).reshape(-1, 2)
    # p = core + t b + s direction. A forward half-strip has an original and an inserted copy.
    ts = np.linalg.solve(np.column_stack((b, direction)), (reference - core).T).T
    t, s = ts.T
    sources = np.flatnonzero((s > 1e-9) & (t >= -1e-9) & (t < 1 - 1e-9))
    copies = {source: len(keys) + i for i, source in enumerate(sources)}
    rule = _insertion_rule(lattice, range(len(keys)), ts, copies, reference, b)
    edge = {pair: dict(vec=vec, type=kind) for pair, (vec, _, kind) in rule.items()}
    # Open the two banks in the developed plane, then close the opposite slit continuously.
    right = ((s <= 1e-9) & (t >= -1e-9)) | ((s > 1e-9) & (t >= 1 - 1e-9))
    opened = np.concatenate((reference + right[:, None] * b, reference[sources] + b))
    r = opened - (core + 0.5 * b)
    angle = np.arctan2(direction[0] * r[:, 1] - direction[1] * r[:, 0], -(r @ direction)) % (2 * np.pi)
    drawn = opened - (1 - angle[:, None] / (2 * np.pi)) * b
    sites = [*keys, *(keys[u] for u in sources)]
    defect_type = engines.FastDefect if fast else ld.Defect
    info = dict(b=b, core=core, cut=cut)
    patch = defect_type("dislocation", sites, drawn, edge, info)
    # A circular crop can separate a few sites of the inserted strip at its outer rim.
    component = max(nx.connected_components(patch.to_networkx()), key=len, default=set())
    if len(component) == len(sites):
        return patch
    keep = sorted(component)
    index = {old: new for new, old in enumerate(keep)}
    edge = {(index[u], index[v]): data for (u, v), data in edge.items() if u in component and v in component}
    return defect_type("dislocation", [sites[u] for u in keep], drawn[keep], edge, info)


def _scanner(fast):
    # Reuse the existing cut enumeration, completion and graph-isomorphism rules. The copied
    # globals are private to this callable, including when two galleries run concurrently.
    primitives = dict(engines._G if fast else vars(ld))
    primitives["dislocation"] = partial(_signed_patch, fast=fast)
    return engines._rebind(ld.dislocation_cores, primitives)


_SCANS = {mode: _scanner(mode) for mode in (False, True)}


def scan(cell, b, *, fast, **options):
    return _SCANS[fast](cell, b, **options)
