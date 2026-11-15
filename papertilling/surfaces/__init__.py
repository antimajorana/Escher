"""A lattice on other surfaces than the plane.

    from papertilling import named
    from papertilling.surfaces import torus, cylinder, klein_bottle, mobius_band
    S = torus(named("kagome"), (4, 0), (0, 4)); S.checks(); S.plot()

Flat surfaces (cylinder, torus, Moebius band, Klein bottle) carry the lattice exactly: they are quotients of the plane by
symmetries of the lattice (flat.py). Closed surfaces with other Euler characteristics need disclinations whose Frank angles
add up to 360 deg x chi (Gauss-Bonnet); see polyhedra.py (sphere, projective plane) and genus.py.

A refusal says why: SurfaceImpossible (proved from the lattice's symmetry: no mirror or glide line -> no non-orientable
surface; no rotation -> no sphere, no projective plane) or SurfaceNotFound (this construction refused at these settings;
e.hint says what to try). surface_table(L) tries every surface and prints OK / IMPOSSIBLE / NOT FOUND with the reason.
"""
from .base import Surface  # noqa: F401
from .flat import torus, cylinder, klein_bottle, mobius_band, glides  # noqa: F401
from .genus import genus_surface, GenusSurface  # noqa: F401
from .polyhedra import sphere, projective_plane, solid_net, frame_control  # noqa: F401
from .refusal import SurfaceRefused, SurfaceImpossible, SurfaceNotFound, symmetry_facts  # noqa: F401
from .report import surface_table, surfaces, SURFACES  # noqa: F401
