//! Building one node's output pairs against the clause's virtual nodes.

use super::*;

/// Inner loop for the both-relevant pair-accumulation step.
///
/// Accumulates `c_t` pairs of types 1/2/3 directly onto `level.pairs` and
/// `d_t` pairs into `clause_dt_pairs`. The caller records
/// `level.pairs.len()` and clears `clause_t3_buf` and `clause_dt_pairs`
/// before the call, and emits the node afterwards.
#[inline(always)]
// The per-level scratch buffers are passed as separate parameters so the
// borrow checker can split them; bundling them in a struct would force one
// shared borrow across the level loop.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_both_rel_pairs(
    eng: &Engine,
    inputs: &[InputPair],
    left_grid_base: usize,
    right_grid_base: usize,
    compute_dt: bool,
    cd_map: &[[u32; 2]],
    level: &mut TddLevel,
    clause_t3_buf: &mut Vec<InputPair>,
    clause_dt_pairs: &mut Vec<InputPair>,
) -> Result<(), ApplyError> {
    let lim = eng.limits();
    let mut prev_left = u32::MAX;
    for p in inputs {
        if p.left.0 != prev_left {
            level.pairs.extend_from_slice(clause_t3_buf);
            clause_t3_buf.clear();
            prev_left = p.left.0;
        }
        let [l_ct, l_dt] = cd_map[left_grid_base + p.left.idx()];
        let [r_ct, r_dt] = cd_map[right_grid_base + p.right.idx()];
        if l_ct != NO_PRODUCT {
            if r_ct != NO_PRODUCT {
                level.pairs.push(InputPair {
                    left: NodeIdx(l_ct),
                    right: NodeIdx(r_ct),
                });
            }
            if r_dt != NO_PRODUCT {
                level.pairs.push(InputPair {
                    left: NodeIdx(l_ct),
                    right: NodeIdx(r_dt),
                });
            }
        }
        if l_dt != NO_PRODUCT && r_ct != NO_PRODUCT {
            lim.try_push(clause_t3_buf, InputPair {
                left: NodeIdx(l_dt),
                right: NodeIdx(r_ct),
            })?;
        }
        if compute_dt && l_dt != NO_PRODUCT && r_dt != NO_PRODUCT {
            lim.try_push(clause_dt_pairs, InputPair {
                left: NodeIdx(l_dt),
                right: NodeIdx(r_dt),
            })?;
        }
    }
    level.pairs.extend_from_slice(clause_t3_buf);
    Ok(())
}

/// Inner loop for the single-relevant pair-accumulation step.
///
/// Accumulates `c_t` pairs directly onto `level.pairs` and `d_t` pairs into
/// `clause_dt_pairs`. The caller records `level.pairs.len()` and clears
/// `clause_dt_pairs` before the call, and emits the node afterwards.
///
/// `left_rel` says which child is the relevant one; `left_grid_base` and
/// `right_grid_base` are the `cd_map` offsets of the children. The irrelevant
/// side's map is not filled, so its raw pair index is used directly.
#[inline(always)]
// The per-level scratch buffers are passed as separate parameters so the
// borrow checker can split them; bundling them in a struct would force one
// shared borrow across the level loop.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_single_rel_pairs(
    eng: &Engine,
    inputs: &[InputPair],
    left_rel: bool,
    left_grid_base: usize,
    right_grid_base: usize,
    compute_dt: bool,
    cd_map: &[[u32; 2]],
    level: &mut TddLevel,
    clause_dt_pairs: &mut Vec<InputPair>,
) -> Result<(), ApplyError> {
    let lim = eng.limits();
    for p in inputs {
        let e = if left_rel {
            cd_map[left_grid_base + p.left.idx()]
        } else {
            cd_map[right_grid_base + p.right.idx()]
        };
        let (l, r) = if left_rel { (e[0], p.right.0) } else { (p.left.0, e[0]) };
        if l != NO_PRODUCT && r != NO_PRODUCT {
            level.pairs.push(InputPair {
                left: NodeIdx(l),
                right: NodeIdx(r),
            });
        }
        if compute_dt {
            let (l, r) = if left_rel { (e[1], p.right.0) } else { (p.left.0, e[1]) };
            if l != NO_PRODUCT && r != NO_PRODUCT {
                lim.try_push(clause_dt_pairs, InputPair {
                    left: NodeIdx(l),
                    right: NodeIdx(r),
                })?;
            }
        }
    }
    Ok(())
}
