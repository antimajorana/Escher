//! Status badges of surface cards, read from the per-lattice surface verdicts (never guessed). WP-C fills in the
//! drawing; the verdict mapping is final.

use crate::backend::VerdictKindUi;
use crate::theme::ThemeTokens;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Badge {
    /// An explicit cell was built (V/E/F counts): constructive.
    ProvedOk,
    /// A symmetry / Gauss-Bonnet proof within the verdict's scope.
    LockedImpossible,
    /// A bounded search found nothing: not a proof.
    Computational,
}

impl Badge {
    pub(crate) fn from_verdict(v: VerdictKindUi) -> Self {
        match v {
            VerdictKindUi::Ok => Self::ProvedOk,
            VerdictKindUi::Impossible => Self::LockedImpossible,
            VerdictKindUi::NotFound => Self::Computational,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::ProvedOk => "[PROVED OK]",
            Self::LockedImpossible => "[LOCKED / IMPOSSIBLE]",
            Self::Computational => "[COMPUTATIONAL]",
        }
    }

    pub(crate) fn colour(self, t: &ThemeTokens) -> egui::Color32 {
        match self {
            Self::ProvedOk => t.status_ok,
            Self::LockedImpossible => t.status_err,
            Self::Computational => t.status_warn,
        }
    }
}

/// The badge as a non-selectable coloured label (so its text is in the accessibility tree).
pub(crate) fn badge_widget(ui: &mut egui::Ui, badge: Badge, tokens: &ThemeTokens) -> egui::Response {
    ui.add(egui::Label::new(egui::RichText::new(badge.label()).small().color(badge.colour(tokens))).selectable(false))
}
