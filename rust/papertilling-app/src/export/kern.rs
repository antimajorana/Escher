//! Pair kerning of a bundled face from its GPOS `kern` feature (PairAdjustment formats 1 and 2), with the legacy
//! `kern` table as a fallback. ab_glyph reads only the legacy table, which Inter does not ship (GPOS only).

use ttf_parser::gpos::{PairAdjustment, PositioningSubtable};
use ttf_parser::{Face, GlyphId, Tag};

pub(super) struct Kerning {
    face: Face<'static>,
    /// GPOS lookup indices of every `kern` feature record, sorted, without duplicates.
    lookups: Vec<u16>,
}

impl Kerning {
    pub(super) fn new(data: &'static [u8]) -> Option<Self> {
        let face = Face::parse(data, 0).ok()?;
        let mut lookups: Vec<u16> = Vec::new();
        if let Some(gpos) = face.tables().gpos {
            for feature in gpos.features {
                if feature.tag == Tag::from_bytes(b"kern") {
                    lookups.extend(feature.lookup_indices);
                }
            }
        }
        lookups.sort_unstable();
        lookups.dedup();
        Some(Self { face, lookups })
    }

    /// Advance adjustment between glyphs `a` then `b`, in font units (0 when the font has none for the pair).
    pub(super) fn pair(&self, a: u16, b: u16) -> f32 {
        let (a, b) = (GlyphId(a), GlyphId(b));
        let Some(gpos) = self.face.tables().gpos.filter(|_| !self.lookups.is_empty()) else {
            return self.legacy(a, b);
        };
        let mut total = 0_i32;
        for &li in &self.lookups {
            let Some(lookup) = gpos.lookups.get(li) else { continue };
            for sub in lookup.subtables.into_iter::<PositioningSubtable>() {
                let PositioningSubtable::Pair(pa) = sub else { continue };
                let hit = match pa {
                    PairAdjustment::Format1 { coverage, sets } => {
                        coverage.get(a).and_then(|i| sets.get(i)).and_then(|set| set.get(b))
                    }
                    PairAdjustment::Format2 { coverage, classes, matrix } => {
                        coverage.get(a).and_then(|_| matrix.get((classes.0.get(a), classes.1.get(b))))
                    }
                };
                // the first subtable of a lookup that applies to the pair is the only one used
                if let Some((first, _)) = hit {
                    total += i32::from(first.x_advance);
                    break;
                }
            }
        }
        total as f32
    }

    fn legacy(&self, a: GlyphId, b: GlyphId) -> f32 {
        let Some(kern) = self.face.tables().kern else { return 0.0 };
        kern.subtables.into_iter().filter(|s| s.horizontal && !s.variable).find_map(|s| s.glyphs_kerning(a, b)).map_or(0.0, f32::from)
    }
}
