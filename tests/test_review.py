"""Tests added in the final review (2026-11-13): cases the earlier tests never executed.

  * Lattice.set through a point with NO D-symbol (crossing bonds) and back, and to_dict there;
  * (the slider callback test moved to tests/test_widgets.py);
  * the builder's snapshot/restore path really runs in the fast-vs-legacy equality case that is meant to cover it;
  * the numba kernel's group action arrays are exact integers (no truncation).
Expected values: the start lattice's own key (stored in the catalogue), and the group p4m of the square
lattice at t1 = -1/2 (the special locus printed by Lattice.special()).
"""
import warnings

import numpy as np
import pytest

from papertilling.lattice import load_catalogue, check_family, LatticeWarning


@pytest.fixture(scope="module")
def L():
    return load_catalogue(group="p4g", rule="bonds").lattices[0]          # rotating squares, one free parameter


def _no_key_value(L):
    name = L.family.names[-1]
    for x in np.linspace(L.values[name] - 0.5, L.values[name] + 0.5, 201):
        if check_family(L.family, **{name: float(x)})["key"] is None:
            return name, float(x)
    pytest.skip("no parameter value without a D-symbol in the scanned range")


def test_set_through_point_without_dsymbol(L):
    name, x = _no_key_value(L)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bad = L.set(**{name: x})
    assert bad.key is None and any(issubclass(m.category, LatticeWarning) for m in w)
    assert bad.to_dict()["key"] is None                                   # used to raise TypeError
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        back = bad.set(**{name: L.values[name]})                          # used to raise TypeError
    assert back.key == L.key and back.status["exact"] and back.status["same_key"] is True
    assert not [m for m in w if issubclass(m.category, LatticeWarning)]   # back on the same lattice: no warning


def test_restore_runs_in_equality_case(monkeypatch):
    """The fast-vs-legacy case "square +180 at a site, -90 at a face" (tests/test_build.py, _EQ_BUILDS[-1]) is there
    to cover FastBuilder.snapshot/restore (shared position arrays): check restore is really called."""
    from papertilling.defects import build as B, fast as F
    calls = []
    cls = F.builder_class()
    orig = cls.restore
    monkeypatch.setattr(cls, "restore", lambda self, snap: (calls.append(1), orig(self, snap))[1])
    c = B.build("square", dict(kind="disclination", angle=180, at=(0, 0)), dict(kind="disclination", angle=-90, at=(0.5, 0.5)),
                R=8.5, picture="raw")
    assert calls, "restore was never called: this case no longer covers the snapshot path"
    assert c.total_charge()["kind"] == "disclination"                     # +180 - 90 = +90


@pytest.mark.parametrize("group", ["p2", "cm", "p4g", "p31m", "p6m"])
def test_kernel_action_arrays_are_exact_integers(group):
    from papertilling import kernels_dsymbol as KD
    f = load_catalogue(group=group, rule="bonds").lattices[0].family
    P, Sh, M, flip = KD.action_arrays(f)
    perm, shift = f._action()
    assert np.array_equal(Sh, np.array(shift, float).reshape(Sh.shape))
    assert all(np.array_equal(M[g], np.asarray(Mf, float)) for g, (Mf, _) in enumerate(f._ops))
    assert np.array_equal(P, np.array(perm))
