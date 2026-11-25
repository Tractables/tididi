use super::*;

/// Sum out `levels` and restore the marginal invariants, on any vtree: the
/// tests' way to a summed internal level on a home, which no entry makes.
pub(crate) fn marginalize_levels(eng: &Engine, f: &mut Tdd, levels: &[VtreeIdx]) -> Result<(), OperationError> {
    let _op = eng.limits().enter()?;
    f.check_level_indices(levels)?;
    let vtree = std::sync::Arc::clone(&f.vtree);
    f.levels.hold_counts(f.levels.is_closed());
    f.levels.forget_pair_total();
    let done = super::evaluate_levels(eng, f, levels, &vtree).and_then(|()| restore_marginal_invariants(eng, f, levels, &vtree));
    if done.is_err() {
        f.levels.forget_left_pairs();
    }
    done
}

mod alloc_guard;
mod deadline;
mod deep;
mod primitives;
pub(crate) mod region;
mod region_canon;
mod region_entries;
mod region_ops;
mod region_reduce;
mod weighted_projection;
