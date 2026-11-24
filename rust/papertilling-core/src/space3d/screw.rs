//! Translation cut and reglue on a finite periodic-net block. Edge endpoints are readdressed in
//! integer cells before applying the helical drawing. The displacement is kinematic.

use std::f64::consts::TAU;

use super::nets3d::Net3dPatch;
use super::points3d::{PointsEntry, points_group};

pub struct ScrewPatch {
    pub net: Net3dPatch,
    pub lattice: [[f64; 3]; 3],
    /// Counterclockwise circuit advance, in Cartesian coordinates.
    pub advance: [f64; 3],
    pub seam_bonds: usize,
}

pub fn build_screw_patch(
    number: u16,
    entry_index: usize,
    periods: i32,
) -> Result<ScrewPatch, String> {
    if !(-2..=2).contains(&periods) || periods == 0 {
        return Err("The axial advance must be ±1 or ±2 primitive periods.".into());
    }
    let group = points_group(number)?;
    let entry = group
        .entries
        .get(entry_index)
        .ok_or_else(|| format!("Unknown lattice entry {} in #{}", entry_index + 1, number))?;
    let name = format!(
        "#{} {} · {}",
        number,
        group.symbol,
        entry.wyckoff.join(" + ")
    );
    build_block(entry, number, name, periods)
}

const RADIUS: i32 = 2;
const WIDTH: usize = (2 * RADIUS + 1) as usize;

fn build_block(
    entry: &PointsEntry,
    number: u16,
    name: String,
    periods: i32,
) -> Result<ScrewPatch, String> {
    let lat = entry.lattice;
    let advance = lat[2].map(|x| f64::from(periods) * x);
    let axis = cut_origin(entry);
    let (ex, ey) = transverse_frame(lat)?;
    let mut positions = Vec::new();
    let mut kinds = Vec::new();
    let mut bonds = Vec::new();
    let mut seam_bonds = 0;

    for i in -RADIUS..=RADIUS {
        for j in -RADIUS..=RADIUS {
            for k in -RADIUS..=RADIUS {
                let cell = [i, j, k];
                for (i, &f) in entry.frac.iter().enumerate() {
                    let f = translated(f, cell);
                    let r = cartesian(lat, [f[0] - axis[0], f[1] - axis[1], 0.0]);
                    let angle = dot(r, ey).atan2(dot(r, ex)).rem_euclid(TAU);
                    let p = cartesian(lat, f);
                    positions.push(core::array::from_fn(|d| p[d] + advance[d] * angle / TAU));
                    kinds.push(entry.labels[i] as u8);
                }
                for &(u, v, offset) in &entry.bonds {
                    // The embedded catalogue stores bounded integer offsets of the neighboring sites.
                    let mut target = core::array::from_fn(|d| cell[d] + offset[d] as i32);
                    let f = translated(entry.frac[u], cell);
                    let g = translated(entry.frac[v], target);
                    let crossing = cut_crossing(f, g, axis);
                    target[2] += periods * crossing;
                    if let Some(to) = site_id(target, v, entry.sites) {
                        let from = site_id(cell, u, entry.sites).expect("cell inside block");
                        bonds.push((from, to, u8::from(crossing != 0)));
                        seam_bonds += usize::from(crossing != 0);
                    }
                }
            }
        }
    }
    if seam_bonds == 0 {
        return Err(
            "This window contains no rejoined seam bond. The preview needs a larger block.".into(),
        );
    }
    let lower = cartesian(
        lat,
        [
            axis[0],
            axis[1],
            -f64::from(RADIUS) - 0.5 + f64::from(periods.min(0)),
        ],
    );
    let upper = cartesian(
        lat,
        [
            axis[0],
            axis[1],
            f64::from(RADIUS) + 0.5 + f64::from(periods.max(0)),
        ],
    );
    let defect_label = format!("Screw dislocation · axial advance {periods:+} primitive a₃");
    let info = format!("{name} · {defect_label} · {seam_bonds} rejoined seam bonds");
    let net = Net3dPatch {
        name,
        space_group_num: number,
        defect_label,
        positions,
        bonds,
        site_kinds: kinds,
        defect_axis: Some((lower, upper)),
        info,
    };
    Ok(ScrewPatch {
        net,
        lattice: lat,
        advance,
        seam_bonds,
    })
}

fn site_id(cell: [i32; 3], site: usize, sites_per_cell: usize) -> Option<usize> {
    if !cell.iter().all(|&n| (-RADIUS..=RADIUS).contains(&n)) {
        return None;
    }
    let [i, j, k] = cell.map(|n| (n + RADIUS) as usize);
    Some(((i * WIDTH + j) * WIDTH + k) * sites_per_cell + site)
}

fn translated(f: [f64; 3], cell: [i32; 3]) -> [f64; 3] {
    core::array::from_fn(|i| f[i] + f64::from(cell[i]))
}

fn cartesian(lat: [[f64; 3]; 3], f: [f64; 3]) -> [f64; 3] {
    core::array::from_fn(|i| (0..3).map(|j| lat[j][i] * f[j]).sum())
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}

fn perpendicular(v: [f64; 3], n: [f64; 3]) -> [f64; 3] {
    let s = dot(v, n);
    core::array::from_fn(|i| v[i] - s * n[i])
}

fn transverse_frame(lat: [[f64; 3]; 3]) -> Result<([f64; 3], [f64; 3]), String> {
    let unit = |v: [f64; 3]| {
        let norm = dot(v, v).sqrt();
        (norm > 1e-9).then(|| v.map(|x| x / norm))
    };
    let err = "The primitive axes do not span a three-dimensional cell.";
    let normal = unit(lat[2]).ok_or(err)?;
    let ex = unit(perpendicular(lat[0], normal)).ok_or(err)?;
    let ey = unit(perpendicular(perpendicular(lat[1], normal), ex)).ok_or(err)?;
    Ok((ex, ey))
}

/// Put the cut between projected site rows, and the axis between projected edge intersections.
fn cut_origin(entry: &PointsEntry) -> [f64; 2] {
    let y = middle_gap(entry.frac.iter().map(|f| f[1]).collect());
    let mut intersections = Vec::new();
    for &(u, v, d) in &entry.bonds {
        let f = entry.frac[u];
        let g = translated(entry.frac[v], d.map(|n| n as i32));
        let low = (f[1].min(g[1]) - y).ceil() as i32;
        let high = (f[1].max(g[1]) - y).floor() as i32;
        for k in low..=high {
            let t = (y + f64::from(k) - f[1]) / (g[1] - f[1]);
            intersections.push((1.0 - t) * f[0] + t * g[0]);
        }
    }
    [middle_gap(intersections), y]
}

fn middle_gap(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.5;
    }
    for value in &mut values {
        *value = value.rem_euclid(1.0);
    }
    values.sort_by(f64::total_cmp);
    let first = values[0];
    let end = first + 1.0;
    values
        .iter()
        .zip(values.iter().skip(1).chain([&end]))
        .max_by(|&(a, b), &(c, d)| (b - a).total_cmp(&(d - c)))
        .map(|(a, b)| ((a + b) * 0.5).rem_euclid(1.0))
        .unwrap()
}

fn cut_crossing(a: [f64; 3], b: [f64; 3], axis: [f64; 2]) -> i32 {
    let ay = a[1] - axis[1];
    let by = b[1] - axis[1];
    if ay.is_sign_positive() == by.is_sign_positive() {
        return 0;
    }
    let t = -ay / (by - ay);
    if (1.0 - t) * a[0] + t * b[0] <= axis[0] {
        return 0;
    }
    if by > ay { 1 } else { -1 }
}

#[cfg(test)]
mod tests;
