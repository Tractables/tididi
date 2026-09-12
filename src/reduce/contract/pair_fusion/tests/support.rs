//! The whole-diagram fusion sweep the canonical-form tests pin.

use super::*;

/// The full unfiltered sweep, for the tests that pin fusion-canonicality on a
/// whole diagram; production uses `fuse_pairs_at_parents`.
pub(crate) fn fuse_pairs(eng: &Engine, tdd: &mut Tdd) -> Result<PairFusionStats, ApplyError> {
    let mut scratch = take_scratch(eng);
    let r = fuse_pairs_inner(eng, tdd, None, &mut scratch);
    return_scratch(eng, scratch);
    r
}
