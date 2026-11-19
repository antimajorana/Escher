//! BDF vertex-completion shell-count table and bounded cache for the H² inspector.

use papertilling_core::hyperbolic::HyperbolicStyle;
use papertilling_core::hyperbolic::inflation::{
    is_hyperbolic, lambda_plus, vertex_completion_counts_budget, vertex_completion_counts_combinatorial,
};

use super::hyperbolic_ui::HyperbolicPanelState;

/// Shells of the inspector's shell table, and the face budget of its geometric fallback.
const SHELL_TABLE_MAX_N: usize = 6;
const SHELL_TABLE_FACE_BUDGET: usize = 20_000;
/// Number of `{p, q}` shell tables kept in [`HyperbolicPanelState::shell_cache`].
pub(super) const SHELL_CACHE_MAX: usize = 16;

/// One row of the shell table: `n`, `|S_n|`, `a_n`, `a_n / a_{n-1}`.
#[derive(Clone, Debug, PartialEq)]
pub struct ShellRow {
    pub n: usize,
    pub cumulative: u64,
    pub shell: u64,
    pub ratio: Option<f64>,
}

/// The inspector's shell table for one `{p, q}`.
#[derive(Clone, Debug, PartialEq)]
pub struct ShellTable {
    pub rows: Vec<ShellRow>,
    pub lambda_plus: f64,
    /// `true`: exact integer counts from the boundary-letter counter; `false`: the geometric counter
    /// (used only where the exact one returns `None`).
    pub exact: bool,
    /// Why the table stops before `n = SHELL_TABLE_MAX_N` (geometric fallback only).
    pub stopped: Option<String>,
}

/// Exact counts from `vertex_completion_counts_combinatorial` (O(n q), microseconds); the geometric
/// counter only if that returns `None` (u64 overflow or a non-convex step, neither of which occurs for
/// p, q <= 12 to n = 6), labelled as such.
pub fn shell_table(p: usize, q: usize) -> Result<ShellTable, String> {
    if !is_hyperbolic(p, q) {
        return Err(format!("{{{p}, {q}}} is not hyperbolic (requires (p-2)(q-2) > 4)"));
    }
    let rows_from = |a: &[u64]| {
        let mut cumulative = 0;
        a.iter()
            .enumerate()
            .map(|(n, &s)| {
                cumulative += s;
                ShellRow { n, cumulative, shell: s, ratio: (n >= 1).then(|| s as f64 / a[n - 1] as f64) }
            })
            .collect()
    };
    if let Some(a) = vertex_completion_counts_combinatorial(p, q, SHELL_TABLE_MAX_N) {
        return Ok(ShellTable { rows: rows_from(&a), lambda_plus: lambda_plus(p, q), exact: true, stopped: None });
    }
    let c = vertex_completion_counts_budget(p, q, SHELL_TABLE_MAX_N, SHELL_TABLE_FACE_BUDGET)?;
    let a: Vec<u64> = c.shells.iter().map(|s| s.shell as u64).collect();
    Ok(ShellTable { rows: rows_from(&a), lambda_plus: c.lambda_plus, exact: false, stopped: c.stopped })
}

impl HyperbolicPanelState {
    /// BDF vertex completion `S_{n+1} = S_n + U(S_n)` from one p-gon (arXiv:1805.02665, Sec. IV-V):
    /// exact counts from the uncapped counter, not from the drawn (capped) patch.
    pub(super) fn show_shell_table(&self, ui: &mut egui::Ui) {
        let Some(((p, q), res)) = self.shell_key.and_then(|k| self.shell_cache.iter().find(|(c, _)| *c == k)) else {
            return;
        };
        ui.separator();
        ui.strong(format!("Vertex completion shells, {{{p},{q}}}, one {p}-gon seed"));
        let table = match res {
            Ok(t) => t,
            Err(e) => {
                ui.small(e);
                return;
            }
        };
        egui::Grid::new("hyp_shell_table").num_columns(4).spacing([10.0, 2.0]).striped(true).show(ui, |ui| {
            for h in ["n", "|S_n|", "a_n", "a_n/a_{n-1}"] {
                ui.small(egui::RichText::new(h).strong());
            }
            ui.end_row();
            for s in &table.rows {
                ui.small(s.n.to_string());
                ui.small(s.cumulative.to_string());
                ui.small(s.shell.to_string());
                ui.small(s.ratio.map_or("-".to_string(), |r| format!("{r:.5}")));
                ui.end_row();
            }
        });
        ui.small(format!(
            "λ+ = γ + √(γ²−1) = {:.5}, γ = ((p−2)(q−2)−2)/2: BDF Eq. 10–11 give it as the growth eigenvalue of the \
             boundary under vertex completion. That the shell ratio a_n/a_{{n-1}} approaches it is observed in this \
             table, not a formula from the paper; λ+ is not a count formula.",
            table.lambda_plus
        ));
        if table.exact {
            ui.small("Counts: exact integers from the boundary-letter counter (no geometry).");
        } else {
            ui.small("Counts: geometric counter (the exact boundary counter returned None for this pair).");
        }
        if let Some(stop) = &table.stopped {
            ui.small(format!("Table stops early: {stop}"));
        }
        if self.style == HyperbolicStyle::RectifiedKagome && self.effective_wedge() == 0 {
            ui.small("Kagome: drawn patch uses the cluster rule on (p.q.p.q) including q-gons; this table shows BDF on {p,q}.");
        }
        if self.effective_wedge() != 0 {
            ui.small(
                "Wedge core Δp ≠ 0: the drawn patch grows copies of the wedge core (vertex completion); \
                 this table (one regular p-gon) does not describe it.",
            );
        }
        if !self.drawn_patch_note.is_empty() {
            ui.small(format!("Drawn patch: {}", self.drawn_patch_note));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    /// Known answers (hand counts, derived in papertilling-core's hyperbolic inflation_tests):
    /// {7,3} a_n = 1, 7, 21, 56; {5,4} a_n = 1, 10, 40; |S_1| = 1 + p (q - 2) for every pair.
    /// Every one of the 92 hyperbolic pairs with p, q in 3..=12 gets an exact table to n = 6; the
    /// first-load time of all 92 is printed.
    #[test]
    fn shell_table_exact_for_all_92_pairs() {
        let t73 = shell_table(7, 3).unwrap();
        assert!(t73.exact);
        assert_eq!(t73.rows.iter().take(4).map(|r| r.shell).collect::<Vec<_>>(), vec![1, 7, 21, 56]);
        assert_eq!(t73.rows.iter().take(4).map(|r| r.cumulative).collect::<Vec<_>>(), vec![1, 8, 29, 85]);
        assert_eq!(t73.rows[2].ratio, Some(3.0));
        let t54 = shell_table(5, 4).unwrap();
        assert_eq!(t54.rows.iter().take(3).map(|r| r.shell).collect::<Vec<_>>(), vec![1, 10, 40]);
        assert!(shell_table(4, 4).is_err());
        let t = Instant::now();
        let mut worst = (0.0, (0, 0));
        let mut pairs = 0;
        for p in 3..=12 {
            for q in (3..=12).filter(|&q| is_hyperbolic(p, q)) {
                let t1 = Instant::now();
                let tab = shell_table(p, q).unwrap();
                let dt = t1.elapsed().as_secs_f64();
                if dt > worst.0 {
                    worst = (dt, (p, q));
                }
                assert!(tab.exact && tab.rows.len() == SHELL_TABLE_MAX_N + 1, "{{{p},{q}}}");
                assert_eq!(tab.rows[1].cumulative, 1 + (p * (q - 2)) as u64, "{{{p},{q}}}");
                pairs += 1;
            }
        }
        assert_eq!(pairs, 92);
        println!("# time shell tables, 92 pairs: {:?}; worst {:.3} ms at {:?}", t.elapsed(), 1e3 * worst.0, worst.1);
    }
}
