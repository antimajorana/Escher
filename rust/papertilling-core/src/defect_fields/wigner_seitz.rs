//! Wigner-Seitz occupancy analysis (`#035`): each actual site belongs to the cell of its nearest reference site
//! (the Wigner-Seitz cell of a site is its Voronoi cell; occupancy 0 marks a vacancy, occupancy >= 2 an interstitial:
//! the usual Wigner-Seitz defect count). Exact nearest-site search; ties go to the lower reference index.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{norm, sub};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WignerSeitz {
    /// Number of actual sites in each reference site's Wigner-Seitz cell.
    pub occupancy: Vec<usize>,
    /// Nearest reference site of each actual site.
    pub assignment: Vec<usize>,
    /// Reference sites with occupancy 0.
    pub vacancies: Vec<usize>,
    /// Reference sites with occupancy >= 2.
    pub interstitial_hosts: Vec<usize>,
}

/// Assign every actual site to its nearest reference site (ties to the lower index) and count occupancies.
pub fn wigner_seitz_occupancy(ref_sites: &[[f64; 2]], actual_sites: &[[f64; 2]]) -> Result<WignerSeitz, String> {
    if ref_sites.is_empty() {
        return Err("no reference sites".to_string());
    }
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in ref_sites {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    // bucket size ~ mean spacing; the extent / n floor keeps collinear (zero-area) reference sets from a tiny `h`
    let (area, ext) = ((hi[0] - lo[0]) * (hi[1] - lo[1]), (hi[0] - lo[0]).max(hi[1] - lo[1]));
    let nr = ref_sites.len() as f64;
    let h = (area / nr).sqrt().max(ext / nr).max(1e-9);
    let key = |p: [f64; 2]| ((p[0] / h).floor() as i64, (p[1] / h).floor() as i64);
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, &p) in ref_sites.iter().enumerate() {
        grid.entry(key(p)).or_default().push(i);
    }
    let max_ring = ((hi[0] - lo[0]).max(hi[1] - lo[1]) / h).ceil() as i64 + 2;
    let mut assignment = Vec::with_capacity(actual_sites.len());
    for &p in actual_sites {
        let (kx, ky) = key(p);
        let mut best: Option<(f64, usize)> = None;
        let mut r = 0i64;
        loop {
            for dx in -r..=r {
                for dy in -r..=r {
                    if dx.abs() != r && dy.abs() != r {
                        continue;
                    }
                    for &i in grid.get(&(kx + dx, ky + dy)).map_or(&[][..], Vec::as_slice) {
                        let d = norm(sub(ref_sites[i], p));
                        if best.is_none_or(|(bd, bi)| d < bd - 1e-12 || ((d - bd).abs() <= 1e-12 && i < bi)) {
                            best = Some((d, i));
                        }
                    }
                }
            }
            // every unvisited cell is at least r * h away
            if best.is_some_and(|(bd, _)| bd <= r as f64 * h) || r > max_ring + (norm(sub(p, lo)) / h) as i64 {
                break;
            }
            r += 1;
        }
        assignment.push(best.map_or(0, |b| b.1));
    }
    let mut occupancy = vec![0usize; ref_sites.len()];
    for &a in &assignment {
        occupancy[a] += 1;
    }
    let vacancies = (0..occupancy.len()).filter(|&i| occupancy[i] == 0).collect();
    let interstitial_hosts = (0..occupancy.len()).filter(|&i| occupancy[i] >= 2).collect();
    Ok(WignerSeitz { occupancy, assignment, vacancies, interstitial_hosts })
}
