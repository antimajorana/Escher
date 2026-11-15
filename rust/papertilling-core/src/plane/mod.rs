//! Euclidean 2D plane module: 17 wallpaper groups, ITA Wyckoff tables, families, tuning, and drawing.

/// The 17 Euclidean plane (wallpaper) groups and exact rational operations (Stage 1).
pub mod groups {
    use serde::{Deserialize, Serialize};

    /// Canonical ITA short Hermann-Mauguin symbols for the 17 wallpaper groups.
    pub const WALLPAPER_GROUPS: [&str; 17] = [
        "p1", "p2", "pm", "pg", "cm", "pmm", "pmg", "pgg", "cmm", "p4", "p4m", "p4g", "p3", "p3m1",
        "p31m", "p6", "p6m",
    ];

    /// Identifier for one of the 17 plane groups.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct PlaneGroupId(pub u8);

    /// Exact 2D fractional affine symmetry operation `x -> m * x + (t_num / t_den)`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub struct RationalOp2D {
        pub m: [[i32; 2]; 2],
        pub t_num: [i32; 2],
        pub t_den: i32,
    }
}

/// International Tables (ITA) 72-row Wyckoff position descriptors (Stage 1).
pub mod wyckoff {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct WyckoffPosition {
        pub group: String,
        pub letter: char,
        pub multiplicity: u8,
        pub site_symmetry: String,
    }
}

/// Parameterised lattice families and enumeration rules (`points`, `sheet`, `bonds`) (Stages 1–2).
pub mod family {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum CatalogueRule {
        Points,
        Sheet,
        Bonds,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct LatticeFamilyStub {
        pub group: String,
        pub rule: CatalogueRule,
        pub parameters: Vec<(String, f64)>,
    }
}

/// Crossing-free parameter intervals and enhanced-symmetry special values (Stage 2).
pub mod tuning {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ParameterBounds {
        pub name: String,
        pub min: f64,
        pub max: f64,
        pub special_values: Vec<f64>,
    }
}

/// 2D relaxation modes, planarity diagnostics, and vector export helpers (Stages 2 & 4).
pub mod draw {
    use faer::Mat;
    use faer::prelude::Solve;
    use faer::sparse::SparseColMat;
    use serde::{Deserialize, Serialize};
    use svg::Document;
    use svg::node::element::Line;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum RelaxationMode {
        GaussNewton,
        Tutte,
        Rigid,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
    pub struct PlanarityCheck {
        pub crossings: usize,
        pub overlaps: usize,
        pub crowded: usize,
        pub folds: usize,
    }

    /// Solve a sparse symmetric positive-definite Laplacian system `L * x = rhs` via sparse Cholesky.
    pub fn solve_tutte_spd(laplacian: &SparseColMat<usize, f64>, rhs: &Mat<f64>) -> Option<Mat<f64>> {
        let llt = laplacian.sp_cholesky(faer::Side::Lower).ok()?;
        Some(llt.solve(rhs))
    }

    /// Render 2D line segments `[(x1, y1, x2, y2)]` to a minimal standalone SVG string.
    pub fn segments_to_svg(segments: &[(f64, f64, f64, f64)], view_box: (f64, f64, f64, f64)) -> String {
        let mut doc = Document::new().set("viewBox", view_box);
        for &(x1, y1, x2, y2) in segments {
            let line = Line::new()
                .set("x1", x1)
                .set("y1", y1)
                .set("x2", x2)
                .set("y2", y2)
                .set("stroke", "#222")
                .set("stroke-width", 0.04);
            doc = doc.add(line);
        }
        doc.to_string()
    }
}
