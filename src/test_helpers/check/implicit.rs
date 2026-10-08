//! The canonical form of implicit levels.

use crate::diagram::LevelState;
use crate::diagram::{floor, stored_levels_forced, ImplicitLevel, Tdd};

/// Every implicit level is in the canonical form of [`ImplicitLevel`]: a
/// structural level whose nodes hold two or more pairs each, [`FLOOR`](crate::diagram::FLOOR) or
/// more in all, described in normal form, by the digits a fit reads off its
/// pairs. No stored structural level can be implicit: none of [`FLOOR`](crate::diagram::FLOOR) or
/// more pairs, below 2^31, fits a description of two or more pairs a node.
///
/// An implicit level stores no node where its description implies them
/// (`ImplicitLevel::implies_nodes`), and stores every node's word where
/// it does not.
///
/// The reduction properties of an implicit level are those of every level,
/// which [`check_canonicity`](super::check_canonicity) decides through the
/// level's pairs.
///
/// A structural level's kept live pairs (`HeldPairs`), where it keeps them,
/// are its pairs.
///
/// Holds at operation boundaries, not inside an operation, where a pass may
/// hold a level it changed stored until the operation closes it. Under the
/// stored route a test forces, where nothing closes, only the implicit
/// levels are checked.
pub fn check_implicit_levels(tdd: &Tdd) -> Result<(), String> {
    let forced = stored_levels_forced();
    for (t, level) in tdd.levels.iter().enumerate() {
        if let LevelState::Structural(held) = &level.state
            && let Some(n) = held.get()
            && n as usize != level.live_pairs()
        {
            return Err(format!("level {t}: keeps {n} live pairs of {}", level.live_pairs()));
        }
        match level.implicit() {
            Some(d) => {
                if level.is_marginal() {
                    return Err(format!("level {t}: a marginal level is implicit"));
                }
                if d.pairs_per_node() < 2 || d.pairs() < floor() {
                    return Err(format!(
                        "level {t}: an implicit level of {} nodes of {} pairs",
                        d.nodes(),
                        d.pairs_per_node()
                    ));
                }
                let stored = level.nodes.stored().len();
                if d.implies_nodes() != (stored == 0) {
                    return Err(format!(
                        "level {t}: an implicit level of {} pairs stores {stored} nodes",
                        d.pairs()
                    ));
                }
                let normal = d.normal();
                if &normal != d {
                    return Err(format!("level {t}: a description not in normal form: {d:?}, normal {normal:?}"));
                }
            }
            None if forced => {}
            None => {
                let len = level.pairs.len();
                if (floor()..1 << 31).contains(&len)
                    && let Some(d) = ImplicitLevel::fit(level)
                    && d.pairs_per_node() >= 2
                    && d.pairs() >= floor()
                {
                    return Err(format!(
                        "level {t}: a stored level of {} nodes of {} pairs is affine",
                        d.nodes(),
                        d.pairs_per_node()
                    ));
                }
            }
        }
    }
    Ok(())
}
