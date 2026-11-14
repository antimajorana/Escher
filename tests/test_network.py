"""The network of a lattice (dsymbol.network_key) and "one list entry per network" (author, 2026-11-14).

Expected values come from hand-known facts, not from the code:
  * the square, triangular and honeycomb networks have one kind of corner each: 4.4.4.4, 3.3.3.3.3.3, 6.6.6,
    so their minimal D-symbols have one element with (face size, bonds per site) = (4, 4), (3, 6), (6, 3);
  * the symmetry group of the square network is p4m (8 operations per cell), of the triangular and honeycomb
    networks p6m (12 per cell); p2 has 2 operations per cell. So the square network drawn as an oblique
    lattice (p2, one site per cell) has symmetry index 8 / 2 = 4; drawn with two sites per cell the cell is
    twice as large, index 8; the triangular and honeycomb networks in p2 have index 12 / 2 = 6;
  * the three p2 recipes below are the entries #1, #4, #5 of the earlier (rule B) list, which the author
    reported as the same lattice shown several times.
"""
import warnings

import pytest

from papertilling.dsymbol import dsymbol, network_key, symmetry_index
from papertilling.lattice import Lattice, minimal_lattices
from papertilling.legacy.wallpaper_lattices import make_cell

SQUARE_NET = ((4, 4, 0, 0, 0),)
TRI_NET = ((3, 6, 0, 0, 0),)
HONEY_NET = ((6, 3, 0, 0, 0),)

P2_ONE_SITE = dict(group="p2", sites=[(0, 0)], bonds=[((0, 0), (-1, -1)), ((0, 0), (-1, 0))], a=1, b=1.23, gamma=76)
P2_TWO_SITES_A = dict(group="p2", sites=[(-0.943, -0.751)],
                      bonds=[((0.057, 0.249), (-0.943, -0.751)), ((0.057, 0.249), (0.943, 0.751)), ((0.057, 0.249), (0.943, 1.751))],
                      a=1, b=1.23, gamma=76)
P2_TWO_SITES_B = dict(group="p2", sites=[(-0.646, -0.224)],
                      bonds=[((0.354, 0.776), (0.646, 0.224)), ((0.354, 0.776), (0.646, 1.224)), ((0.354, 0.776), (1.646, 0.224)),
                             ((0.354, 0.776), (1.646, 1.224))], a=1, b=1.23, gamma=76)
P2_TRIANGULAR = dict(group="p2", sites=[(0, 0)], bonds=[((0, 0), (-1, -1)), ((0, 0), (-1, 0)), ((0, 0), (0, -1))], a=1, b=1.23, gamma=76)
P2_BRICK = dict(group="p2", sites=[(-0.029, -0.155)],
                bonds=[((0.971, 0.845), (0.029, 0.155)), ((0.971, 0.845), (0.029, 1.155)), ((0.971, 0.845), (1.029, 0.155))],
                a=1, b=1.23, gamma=76)


def own_net(name):
    return network_key(dsymbol(make_cell(name))["key"])


def test_known_networks_of_catalogue_cells():
    assert own_net("square") == SQUARE_NET
    assert own_net("triangular") == TRI_NET
    assert own_net("honeycomb") == HONEY_NET
    assert len({own_net(n) for n in ("square", "triangular", "honeycomb", "kagome", "shastry_sutherland")}) == 5


def test_one_network_in_three_p2_drawings():
    """The author's case: the parallelogram lattice with one site per cell and its two 2-site drawings."""
    Ls = [Lattice.from_recipe(r) for r in (P2_ONE_SITE, P2_TWO_SITES_A, P2_TWO_SITES_B)]
    assert len({L.key for L in Ls}) == 3                     # rule B told them apart ...
    assert [L.network for L in Ls] == [SQUARE_NET] * 3        # ... but they are one network
    assert [L.symmetry_index for L in Ls] == [4, 8, 8]
    assert [L.vertex_types() for L in Ls] == [["4.4.4.4"]] * 3


def test_symmetry_index_by_hand():
    assert Lattice.from_recipe(P2_TRIANGULAR).network == TRI_NET
    assert Lattice.from_recipe(P2_TRIANGULAR).symmetry_index == 6
    assert Lattice.from_recipe(P2_BRICK).network == HONEY_NET
    assert Lattice.from_recipe(P2_BRICK).symmetry_index == 6
    assert symmetry_index(dsymbol(make_cell("square"))["key"]) == 1          # the square lattice through its own group


def test_negative_control_different_networks():
    a, b = Lattice.from_recipe(P2_ONE_SITE), Lattice.from_recipe(P2_TRIANGULAR)
    assert a.network != b.network and a.network != Lattice.from_recipe(P2_BRICK).network


def test_p2_list_has_seven_different_networks():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimal_lattices("p2", rule="bonds")
    nets = [L.network for L in res.all]
    assert len(res.lattices) == 7 and len(set(nets)) == len(nets)
    assert [L.network for L in res.lattices[:3]] == [SQUARE_NET, TRI_NET, HONEY_NET]
    assert [tuple(L.rank) for L in res.lattices[:3]] == [(1, 2, 1, 2), (1, 3, 1, 3), (1, 3, 2, 3)]      # the simplest drawing of each
    assert len(res.other[0]) >= 2 and all(L.network == SQUARE_NET for L in res.other[0])                 # its other drawings are kept
    assert len(res.drawings) == len(res.all) + sum(len(o) for o in res.other)
    assert all(L.meta["routes_agree"] for L in res.drawings)


@pytest.mark.slow
def test_rule_b_still_available():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimal_lattices("p2", rule="bonds", same="action")
    assert len(res.lattices) == 7 and len({L.network for L in res.lattices}) < 7      # the earlier list repeats networks
