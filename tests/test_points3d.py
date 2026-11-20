"""3D points rule (C1 WP-P0): Wyckoff positions, the shells-until-connected bonds, the net key.

Expected values are hand-derived or independent of the code under test:
  * Wyckoff multiplicities: International Tables for Crystallography vol. A, the textbook tables of the groups
    (P-1 has 8 inversion centres 1a-1h and 2i; P2_1/c 2a-2d, 4e; Pmmm 27 positions; ...);
  * coordination sequences: sc 4n^2+2 (points with |x|+|y|+|z| = n), bcc 6n^2+2, fcc 10n^2+2, diamond
    4, 12, 24, 42, 64, 92; all four re-derived by an independent Cartesian nearest-neighbour BFS (scratch
    script cs_indep.py, 2026-11-20), and the first one by hand;
  * which point sets have more symmetry than G: hand arguments in the test comments.
"""
import math

import numpy as np
import pytest

from papertilling import _net3d_key as NK
from papertilling import _wyckoff3d as W
from papertilling import points3d as P


def mults(number):
    return [w["mult_conv"] for w in W.wyckoff_positions(number)]


# ---------------------------------------------------------------- Wyckoff positions (ITA vol. A multiplicities)
def test_wyckoff_low_symmetry():
    assert mults(1) == [1]
    assert mults(2) == [1] * 8 + [2]                                   # 8 inversion centres, general 2i
    assert mults(14) == [2, 2, 2, 2, 4]                                # P2_1/c: 2a-2d, 4e
    assert [w["name"] for w in W.wyckoff_positions(14)] == ["2a", "2b", "2c", "2d", "4e"]


def test_wyckoff_pmmm():
    wy = W.wyckoff_positions(47)
    kinds = [w["kind"] for w in wy]
    # 8 points (1a-1h), 12 lines (2i-2t), 6 planes (4u-4z), general 8alpha (spglib writes 'A')
    assert kinds == ["point"] * 8 + ["line"] * 12 + ["plane"] * 6 + ["general"]
    assert mults(47) == [1] * 8 + [2] * 12 + [4] * 6 + [8]
    assert wy[-1]["letter"] == "A"


def test_wyckoff_hexagonal_and_cubic():
    assert mults(194) == [2, 2, 2, 2, 4, 4, 6, 6, 12, 12, 12, 24]      # P6_3/mmc a-l
    assert mults(227) == [8, 8, 16, 16, 32, 48, 96, 96, 192]           # Fd-3m a-i
    fd = W.wyckoff_positions(227)
    assert [w["mult"] for w in fd][:2] == [2, 2]                       # 2 sites per primitive (F) cell
    assert mults(221) == [1, 1, 3, 3, 6, 6, 8, 12, 12, 12, 24, 24, 24, 48]   # Pm-3m a-n


@pytest.mark.slow
def test_wyckoff_all_230_match_spglib_letters():
    # wyckoff_positions raises unless every position gets its own spglib letter, the letters are a..general,
    # and |site-symmetry group| equals the order of spglib's site-symmetry symbol. (# time about 150 s)
    total = 0
    for n in range(1, 231):
        total += len(W.wyckoff_positions(n))
    assert total == 1731                     # the total usually quoted for ITA vol. A (from memory, not looked up)


# ---------------------------------------------------------------- the net state (components, rank, index)
def test_state_hand_cases():
    e = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    assert NK.state(1, [(0, 0, n) for n in e]) == (1, 3, 1)                       # pcu
    assert NK.state(1, [(0, 0, (2, 0, 0)), (0, 0, (0, 1, 0)), (0, 0, (0, 0, 1))]) == (1, 3, 2)   # 2 copies
    assert NK.state(2, [(0, 0, n) for n in e]) == (2, 3, 1)                       # site 1 isolated
    assert NK.state(1, [(0, 0, (1, 0, 0))]) == (1, 1, 0)                          # a chain
    assert NK.lattice_rank_index([(2, 0, 0), (0, 3, 0), (0, 0, 1), (1, 1, 0)]) == (3, 1)   # gcd of minors 1


def test_shells_skip_collinear_repeat():
    # One site in the cell a=(1,0,0), b=(0.3,1.7,0), c=(0.2,0.4,2.9). By hand: shells |a| = 1 (rank 1),
    # |b| = 1.726 (rank 2), |b-a| = 1.838 and |2a| = 2 change nothing (skipped), |c| = 2.934 is the shortest
    # vector with a c component (|c-a| = 3.035, |c-b| = 3.18), so degree 6.
    A = np.array([[1, 0, 0], [0.3, 1.7, 0], [0.2, 0.4, 2.9]])
    bonds, why = P.shell_bonds(np.zeros((1, 3)), A, np.zeros(1, int))
    assert why == "ok"
    assert sorted(n for _, _, n in bonds) == [(0, 0, 1), (0, 1, 0), (1, 0, 0)]


def test_site_on_bond_is_refused():
    # Sites (0,0,0) and (1/2,0,0) of a cubic cell: the bond 0 -> 0 + (1,0,0) passes through site 1.
    F, A = np.array([[0, 0, 0], [0.5, 0, 0]]), np.eye(3)
    I, J, N, L, V = P._pairs_within(F, A, 2.0)
    bad = [(0, 1, (0, 0, 0)), (0, 0, (1, 0, 0)), (0, 0, (0, 1, 0)), (0, 0, (0, 0, 1))]
    assert P._validate(bad, I, J, N, V, np.array([0, 1]), 2)[1] == "a third site lies on a bond"


# ---------------------------------------------------------------- exact group: hand negatives and positives
def _orbit_cell(number, names, c=None):
    wy = {w["name"]: w for w in W.wyckoff_positions(number)}
    g = W.group_entry(number)
    A = W.cell_matrix(1, 1, c or 1, 90, 90, 120 if g["system"] in ("hexagonal", "trigonal") else 90)
    X = np.vstack([W.orbit(number, wy[n]["p0"]) for n in names])
    F, Ap, _ = P.to_primitive(number, X, A)
    return F, Ap


def test_exact_group_hand_cases():
    assert P.exact_group(221, *_orbit_cell(221, ["1a"]))                # sc: exactly Pm-3m
    assert P.exact_group(227, *_orbit_cell(227, ["8a"]))                # diamond: exactly Fd-3m
    assert not P.exact_group(221, *_orbit_cell(221, ["1a", "1b"]))      # 1a + 1b = bcc: Im-3m
    assert not P.exact_group(225, *_orbit_cell(225, ["8c"]))            # 8c = simple cubic of edge a/2
    one = np.zeros((1, 3))
    A = W.cell_matrix(1, 1.1, 1.3, 81, 97, 104)
    assert not P.exact_group(1, one, A)                                  # a Bravais lattice has inversion
    assert not P.exact_group(1, np.array([[0, 0, 0], [0.31, 0.17, 0.43]]), A)   # inversion at the midpoint
    assert P.exact_group(1, np.array([[0, 0, 0], [0.31, 0.17, 0.43], [0.62, 0.11, 0.27]]), A)


def test_hcp_two_phases():
    # P6_3/mmc 2c, a = 1: in-plane neighbours at 1, out-of-plane at sqrt(1/3 + c^2/4); these tie at
    # c = sqrt(8/3) = 1.633. Below: the 6 out-of-plane bonds alone are connected and 3-periodic (degree 6);
    # above: in-plane layers first (rank 2), then the out-of-plane shell (degree 12, hcp).
    for c, deg in ((1.2, 6), (1.8, 12)):
        F, A = _orbit_cell(194, ["2c"], c)
        bonds, why = P.shell_bonds(F, A, np.zeros(len(F), int))
        assert why == "ok" and set(NK.degrees(len(F), bonds).tolist()) == {deg}


# ---------------------------------------------------------------- tiers and the four known first entries
def test_pmmm_choice_counts():
    wy = W.wyckoff_positions(47)
    two = P.configurations(wy, 2)
    assert len(two) == math.comb(27, 2) + 19                             # 370: pairs, plus 19 non-point repeats
    by_m = {m: sum(1 for c in two if sum(wy[i]["mult"] for i in c) == m) for m in (2, 3)}
    assert by_m == {2: math.comb(8, 2), 3: 8 * 12}                       # two points; a point and a line


def sc(n): return 4 * n * n + 2
def bcc(n): return 6 * n * n + 2
def fcc(n): return 10 * n * n + 2


@pytest.mark.parametrize("number,first,also,cs", [
    (221, ["1a"], ["1b"], [sc(n) for n in range(1, 7)]),
    (229, ["2a"], [], [bcc(n) for n in range(1, 7)]),
    (225, ["4a"], ["4b"], [fcc(n) for n in range(1, 7)]),
    (227, ["8a"], ["8b"], [4, 12, 24, 42, 64, 92]),
])
def test_known_first_entries(number, first, also, cs):
    r = P.points3d_lattices(number, n=1)
    e = r["entries"][0]
    assert e["wyckoff"] == first
    assert all(len(a) == 1 for a in e["also"]) and [a[0] for a in e["also"]] == also
    assert e["cs"][0][:6] == cs


def test_p21c_tiers():
    # P2_1/c: 2a-2d are pairs of inversion centres, e.g. (0,0,0), (0,1/2,1/2): a centred lattice, which has
    # more symmetry than P2_1/c, so the first tier (k=1, 2 sites) is empty and every choice is dropped there.
    r = P.points3d_lattices(14)
    t0 = r["tiers"][0]
    assert (t0["k"], t0["m"], t0["choices"]) == (1, 2, 4)
    assert t0["counts"].get("more_symmetry_than_G_early", 0) == 4 and t0["counts"].get("new", 0) == 0
    keys = [e["key"] for e in r["entries"]]
    assert len(r["entries"]) >= 7 and len(set(keys)) == len(keys)
    for e in r["entries"]:
        F, A = np.array(e["F"]), np.array(e["A"])
        assert P.exact_group(14, F, A)
        bonds = [(i, j, tuple(n)) for i, j, n in e["bonds"]]
        assert NK.state(len(F), bonds) == (1, 3, 1)
