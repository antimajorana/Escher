//! Live search and facet filter of the lattice browser: canonical name, custom alias, ITA symbol or number,
//! crystal system and chirality. Pure functions, so the matching rules are unit-tested without a UI.

use papertilling_core::plane::groups::WALLPAPER_GROUPS;

use crate::backend::{CatalogueItemUi, GroupInfo};

/// Crystal-system facet (the four 2D crystal systems; `None` in [`BrowseFilter::system`] = all).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SystemFacet {
    Oblique,
    Rectangular,
    Square,
    Hexagonal,
}

impl SystemFacet {
    pub const ALL: [Self; 4] = [Self::Oblique, Self::Rectangular, Self::Square, Self::Hexagonal];

    pub fn label(self) -> &'static str {
        match self {
            Self::Oblique => "Oblique",
            Self::Rectangular => "Rectangular",
            Self::Square => "Square",
            Self::Hexagonal => "Hexagonal",
        }
    }

    /// The lower-case name used in [`GroupInfo::crystal_system`].
    pub fn key(self) -> &'static str {
        match self {
            Self::Oblique => "oblique",
            Self::Rectangular => "rectangular",
            Self::Square => "square",
            Self::Hexagonal => "hexagonal",
        }
    }
}

/// Chirality facet of a lattice (its `chiral` flag: an enantiomorph pair in 2D).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChiralFacet {
    #[default]
    Any,
    Chiral,
    Achiral,
}

impl ChiralFacet {
    pub const ALL: [Self; 3] = [Self::Any, Self::Chiral, Self::Achiral];

    pub fn label(self) -> &'static str {
        match self {
            Self::Any => "Any",
            Self::Chiral => "± Chiral",
            Self::Achiral => "Achiral",
        }
    }
}

/// ITA number (`1..=17`) of a wallpaper group symbol as written in this app (`p1` = 1, ..., `p6m` = 17).
pub fn ita_number(symbol: &str) -> Option<u8> {
    WALLPAPER_GROUPS
        .iter()
        .position(|g| *g == symbol)
        .and_then(|i| u8::try_from(i + 1).ok())
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BrowseFilter {
    pub query: String,
    pub system: Option<SystemFacet>,
    pub chirality: ChiralFacet,
}

impl BrowseFilter {
    /// True when no facet and no text narrows the list (the browser then shows the selected group only).
    pub fn is_default(&self) -> bool {
        self.query.trim().is_empty() && self.system.is_none() && self.chirality == ChiralFacet::Any
    }

    /// Group passes the crystal-system facet.
    pub fn group_visible(&self, g: &GroupInfo) -> bool {
        self.system.is_none_or(|s| g.crystal_system == s.key())
    }

    /// Item passes every facet and the text query. `crystal_system` is the system of `item.group`.
    pub fn matches(&self, item: &CatalogueItemUi, crystal_system: &str) -> bool {
        if self.system.is_some_and(|s| s.key() != crystal_system) {
            return false;
        }
        match self.chirality {
            ChiralFacet::Any => {}
            ChiralFacet::Chiral if !item.chiral => return false,
            ChiralFacet::Achiral if item.chiral => return false,
            _ => {}
        }
        query_matches(&self.query, item)
    }
}

/// Every whitespace-separated term must match: a number matches the ITA number of the group exactly
/// (also written `ita17`; `#3` is text, as in the catalogue label `p4 #3 (1,2)`); other terms match case-insensitively inside the custom alias, the
/// canonical name, the named-lattice key, the group symbol or the sheet group.
pub fn query_matches(query: &str, item: &CatalogueItemUi) -> bool {
    let hay = [
        item.label.as_str(),
        item.default_label.as_str(),
        item.named_key.as_deref().unwrap_or(""),
        item.group.as_str(),
        item.sheet_group.as_deref().unwrap_or(""),
    ]
    .map(str::to_lowercase);
    query_terms(query).iter().all(|t| {
        let num = t.strip_prefix("ita").unwrap_or(t);
        // an ITA number is 1..=17 written with digits only; other numbers ("42" in an alias) are text
        if !num.is_empty()
            && num.bytes().all(|b| b.is_ascii_digit())
            && let Ok(n) = num.parse::<u8>()
            && (1..=17).contains(&n)
        {
            return ita_number(&item.group) == Some(n);
        }
        // the group symbol matches exactly when the term is a symbol (so "pm" does not hit "p4m")
        if ita_number(t).is_some() {
            return item.group.eq_ignore_ascii_case(t)
                || item.sheet_group.as_deref().is_some_and(|s| s.eq_ignore_ascii_case(t));
        }
        hay.iter().any(|h| h.contains(t.as_str()))
    })
}

/// Lower-case terms of `query`, with "ita" joined to a following number ("ITA 17" = "ita17").
fn query_terms(query: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut words = query.split_whitespace().map(str::to_lowercase).peekable();
    while let Some(w) = words.next() {
        if w == "ita"
            && let Some(next) = words.peek()
            && next.bytes().all(|b| b.is_ascii_digit())
        {
            let n = words.next().unwrap_or_default();
            out.push(format!("ita{n}"));
        } else {
            out.push(w);
        }
    }
    out
}

/// Crystal system of `group` from the group table (`""` when unknown).
pub fn system_of<'a>(groups: &'a [GroupInfo], group: &str) -> &'a str {
    groups.iter().find(|g| g.symbol == group).map_or("", |g| g.crystal_system)
}
