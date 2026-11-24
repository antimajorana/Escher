//! Geometry shared by the 2D canvas and the fusion scene builder.

pub(crate) fn cross(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// Minimum interior angle (radians) of a triangle; used to prefer fat ears and avoid razor slivers.
fn min_tri_angle(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    let verts = [a, b, c];
    let mut best = std::f64::consts::PI;
    for i in 0..3 {
        let (o, p, q) = (verts[i], verts[(i + 1) % 3], verts[(i + 2) % 3]);
        let (ux, uy) = (p[0] - o[0], p[1] - o[1]);
        let (vx, vy) = (q[0] - o[0], q[1] - o[1]);
        let denom = ux.hypot(uy) * vx.hypot(vy);
        if denom < 1e-18 {
            return 0.0;
        }
        let cos = ((ux * vx + uy * vy) / denom).clamp(-1.0, 1.0);
        best = best.min(cos.acos());
    }
    best
}

pub(crate) fn orient_ccw(poly: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let area2: f64 = (0..poly.len()).map(|i| cross([0.0, 0.0], poly[i], poly[(i + 1) % poly.len()])).sum();
    if area2 >= 0.0 {
        poly.to_vec()
    } else {
        poly.iter().copied().rev().collect()
    }
}

/// Ear-clipping triangulation of a simple polygon (either orientation), picking the ear with the largest
/// minimum angle at each step so razor-thin slivers (which cause miter spikes in egui's AA) are avoided.
pub(crate) fn triangulate(poly: &[[f64; 2]]) -> Vec<[[f64; 2]; 3]> {
    let ccw = orient_ccw(poly);
    let mut idx: Vec<usize> = (0..ccw.len()).collect();
    let mut out = Vec::new();
    while idx.len() > 3 {
        let n = idx.len();
        let mut best: Option<(usize, f64)> = None;
        for k in 0..n {
            let (a, b, c) = (ccw[idx[(k + n - 1) % n]], ccw[idx[k]], ccw[idx[(k + 1) % n]]);
            if cross(a, b, c) <= 1e-12 {
                continue;
            }
            let empty = !idx.iter().any(|&j| {
                let p = ccw[j];
                let corner = [a, b, c].iter().any(|q| (q[0] - p[0]).hypot(q[1] - p[1]) < 1e-12);
                !corner && cross(a, b, p) >= 0.0 && cross(b, c, p) >= 0.0 && cross(c, a, p) >= 0.0
            });
            if empty {
                let score = min_tri_angle(a, b, c);
                if best.is_none_or(|(_, s)| score > s) {
                    best = Some((k, score));
                }
            }
        }
        // No ear (a degenerate outline): fill what is left as a fan so that no face is left blank.
        let Some((k, _)) = best else {
            out.extend((1..n - 1).map(|j| [ccw[idx[0]], ccw[idx[j]], ccw[idx[j + 1]]]));
            return out;
        };
        out.push([ccw[idx[(k + n - 1) % n]], ccw[idx[k]], ccw[idx[(k + 1) % n]]]);
        idx.remove(k);
    }
    if idx.len() == 3 {
        out.push([ccw[idx[0]], ccw[idx[1]], ccw[idx[2]]]);
    }
    out
}

pub(crate) fn point_in_polygon(pt: [f32; 2], poly: &[[f32; 2]]) -> bool {
    if poly.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let (xi, yi) = (poly[i][0], poly[i][1]);
        let (xj, yj) = (poly[j][0], poly[j][1]);
        let crosses = (yi > pt[1]) != (yj > pt[1]);
        if crosses {
            let x_int = (xj - xi) * (pt[1] - yi) / (yj - yi) + xi;
            if pt[0] < x_int {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}
