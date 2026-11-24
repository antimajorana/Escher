use super::*;

#[test]
fn cubic_screw_rejoins_the_cut_and_lifts_a_burgers_circuit() {
    for advance in [-2, -1, 1, 2] {
        let patch = build_screw_patch(221, 0, advance).unwrap(); // simple cubic, one site per cell
        let net = patch.net;
        assert_eq!(net.positions.len(), 125);
        assert_eq!(patch.seam_bonds, 2 * (5 - advance.unsigned_abs() as usize));
        assert_eq!(net.bonds.len(), 300 - 2 * advance.unsigned_abs() as usize);
        assert!((patch.advance[2] - f64::from(advance)).abs() < 1e-12);

        // Traverse once counterclockwise around the core plaquette. The first bond crosses the
        // positive-x branch cut. Its target must be in the translated z cell, at the same site.
        let path = [
            [1, 0, 0],
            [1, 1, advance],
            [0, 1, advance],
            [0, 0, advance],
            [1, 0, advance],
        ];
        let ids = path.map(|cell| site_id(cell, 0, 1).unwrap());
        for (step, pair) in ids.windows(2).enumerate() {
            assert!(
                net.bonds.iter().any(|&(u, v, seam)| {
                    ((u == pair[0] && v == pair[1]) || (v == pair[0] && u == pair[1]))
                        && seam == u8::from(step == 0)
                }),
                "missing circuit edge {step} for advance {advance}"
            );
        }
        assert!(
            (net.positions[ids[4]][2] - net.positions[ids[0]][2] - f64::from(advance)).abs()
                < 1e-12
        );
        let unglued = site_id([1, 1, 0], 0, 1).unwrap();
        assert!(
            !net.bonds
                .iter()
                .any(|&(u, v, _)| (u, v) == (ids[0], unglued))
        );
    }
}

#[test]
fn one_screw_preview_builds_for_every_stored_space_group() {
    for number in 1..=230 {
        let patch = build_screw_patch(number, 0, 1)
            .unwrap_or_else(|err| panic!("space group #{number}: {err}"));
        assert!(patch.seam_bonds > 0 && patch.net.defect_axis.is_some());
        assert!(patch.net.positions.iter().flatten().all(|x| x.is_finite()));
        assert!(patch.net.bonds.iter().all(|&(u, v, _)| u != v));
    }
}
