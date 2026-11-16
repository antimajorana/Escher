"""Reader and crossing parity checks, split from test_build to keep modules under 600 lines."""
import copy

import numpy as np
import pytest

from papertilling.defects.build import build, lattice_data
from papertilling.defects import fast as fastmod
from papertilling.legacy import fusion_framework as ff
from papertilling.legacy.lab_defects import frame_faces
from test_build import _eq_specs, _EQ_BUILDS


def _same_reading(d, with_cores=True):
    """frame_faces, total_frank, total_charge, cores: fast readers vs legacy on one crystal (cores
    needs the builder's info['reference']: not for lab_defects crystals)."""
    F0, F1 = frame_faces(d), fastmod.frame_faces_fast(d)
    assert F0 == F1
    assert abs(fastmod.total_frank(d, True) - ff.total_frank(d)) < 1e-12
    (H0, S0), (H1, S1) = ff.total_charge(d), fastmod.total_charge(d, True)
    assert np.allclose(H0[0], H1[0], rtol=0, atol=1e-12) and np.allclose(H0[1], H1[1], rtol=0, atol=1e-12)
    assert np.allclose(S0, S1, rtol=0, atol=1e-12)
    C0, C1 = (ff.cores(d), fastmod.cores(d, True)) if with_cores else ([], [])
    assert len(C0) == len(C1)
    for a, b in zip(C0, C1):
        assert a.keys() == b.keys()
        for k in a:
            if isinstance(a[k], np.ndarray):
                assert np.allclose(a[k], b[k], rtol=0, atol=1e-12), k
            else:
                assert a[k] == b[k], (k, a[k], b[k])
    return S0


def test_readers_fast_equal_legacy():
    """Fast readers preserve faces, unreduced Frank angles, holonomy, Burgers sums and cores."""
    crystals = [build("square", *_eq_specs("square", _EQ_BUILDS[1][1])[:2], R=11, centre=(0, 0), picture="raw").defect,
                build("honeycomb", *_eq_specs("honeycomb", "57"), picture="raw").defect]
    lat = lattice_data("shastry_sutherland")
    crystals.append(fastmod.dislocation_fast(lat.cell, lat.cell.a1, core=0.371 * lat.cell.a1 + 0.413 * lat.cell.a2, R=8.0))
    for i, d in enumerate(crystals):
        _same_reading(d, with_cores=i < 2)
    bad = copy.deepcopy(crystals[0])
    k = next(k for k in bad.edge if "g" not in bad.edge[k])
    bad.edge[k]["vec"] = bad.edge[k]["vec"] + np.array([1e-3, 0.0])
    bad.info.pop("_orientable", None)
    assert not np.allclose(_same_reading(bad), _same_reading(crystals[0]), rtol=0, atol=1e-6)


def test_has_crossings_fast_equals_legacy():
    """Crossings agree on known planar cells and a negative control with both square diagonals."""
    import dataclasses
    from papertilling.legacy.wallpaper_lattices import make_cell, has_crossings
    for name in ("square", "honeycomb", "kagome", "shastry_sutherland"):
        c = make_cell(name)
        assert fastmod.has_crossings_fast(c) == has_crossings(c)
    sq = make_cell("square")
    x = dataclasses.replace(sq, bonds=list(sq.bonds) + [(0, 0, 1, 1, 1), (0, 0, 1, -1, 1)])
    assert has_crossings(x) and fastmod.has_crossings_fast(x)


@pytest.mark.slow
def test_has_crossings_fast_all_cells():
    from papertilling.legacy.wallpaper_lattices import CATALOGUE, make_cell, has_crossings
    for name in CATALOGUE:
        c = make_cell(name)
        assert fastmod.has_crossings_fast(c) == has_crossings(c), name
