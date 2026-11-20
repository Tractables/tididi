//! Rebuilding a spine level as the accumulator level conjoined with the clause.

use super::*;

/// The [`ClauseWork`] fields a spine level's rebuild uses, borrowed apart.
///
/// The per-node loops must not read `cd_map` through
/// `&mut ClauseWork`: a pair buffer's push that grows calls out of line with
/// a pointer into the working set, after which the compiler reloads every
/// field it reads through it, per node and per pair (about 2% more
/// instructions on a workload of large clause conjunctions). Held here,
/// the slice stays out of the buffers' reach.
pub(super) struct ClauseTables<'a> {
    pub(super) cd_map: &'a mut [[u32; 2]],
    pub(super) level_base: &'a [usize],
    pub(super) need_dt: &'a [bool],
    pub(super) on_spine: &'a [bool],
    pub(super) t3_buf: &'a mut Vec<ChildPair>,
    pub(super) dt_pairs: &'a mut Vec<ChildPair>,
    /// The emptied arenas of the spine level rebuilt last, which the next
    /// one builds its output in where they fit: a walk over small levels
    /// allocates no arena per level.
    pub(super) spare_nodes: &'a mut Vec<EncodedNode>,
    pub(super) spare_pairs: &'a mut Vec<ChildPair>,
    /// The highest level with both children on the spine, the one that
    /// builds no `d_t`; per side, how often it may have left a node of that
    /// child unnamed (a `d` half beside a partner with no `c` half); and its
    /// input pairs.
    pub(super) top: Option<(VtreeIdx, [usize; 2], usize)>,
}

/// The child-map offsets and worst-case output pairs per input pair for a
/// spine level.
#[derive(Clone, Copy)]
pub(super) struct SpineCtx {
    pub(super) left_grid_base: usize,
    pub(super) right_grid_base: usize,
    pub(super) pair_mult: usize,
}

/// Conjoin one accumulator node with the clause's virtual `c_t` (and `d_t`
/// where the level needs it), writing the emitted node indices into `slot`.
/// Return how many pairs the two nodes hold.
///
/// The caller has already put the level's growth mode in place.
pub(super) fn conjoin_node_with_clause<const LEFT: bool, const RIGHT: bool, const DT: bool>(
    eng: &Engine,
    inputs: &[ChildPair],
    ctx: SpineCtx,
    level: &mut TddLevel,
    slot: usize,
    tables: &mut ClauseTables<'_>,
) -> Result<usize, OperationError> {
    // Reserve this node's whole worst case before emitting any of it, so
    // the direct c_t pushes stay infallible `Vec::push`es.
    reserve_pairs_for_emit(eng, level, ctx.pair_mult * inputs.len())?;
    let ct_start = level.pairs.stored_len();
    if DT {
        tables.dt_pairs.clear();
    }

    // The virtual c_t has three products when both children carry clause
    // variables, and one otherwise; each builder leaves its pairs sorted.
    if LEFT && RIGHT {
        tables.t3_buf.clear();
        build_both_rel_pairs::<DT>(eng, inputs, ctx, level, tables)?;
    } else {
        build_single_rel_pairs::<LEFT, DT>(eng, inputs, ctx, level, tables)?;
    }
    let written = level.pairs.stored_len() - ct_start + if DT { tables.dt_pairs.len() } else { 0 };
    let ct = emit_from(eng, level, ct_start)?;
    // The parent's both-relevant pass requires the c_t index below d_t. The
    // d_t pairs fit the reservation above: at most one per input pair, which
    // `pair_mult` counts.
    let dt = if DT {
        let dt_start = level.pairs.stored_len();
        level.pairs.stored_mut().extend_from_slice(tables.dt_pairs);
        emit_from(eng, level, dt_start)?
    } else {
        NO_PRODUCT
    };
    tables.cd_map[slot] = [ct, dt];
    Ok(written)
}

/// Whether a spare arena of `capacity` serves a reserve of `need`: it holds
/// at most twice what a fresh reserve allocates, which is at least four
/// entries (`Vec`'s smallest allocation of 8-byte elements), so a level keeps
/// no more slack than a reserve's growth leaves.
fn fits_reserve(capacity: usize, need: usize) -> bool {
    capacity <= 2 * need.max(4)
}

/// Rebuild one spine level as the conjunction of the accumulator's level with
/// the clause, filling this level's `cd_map` block. Returns whether a pair
/// of the accumulator's level was dropped where the level has one relevant
/// child and computes no `d_t`: only there can a node of the other child
/// that a pair named be named by none after.
pub(super) fn rebuild_spine_level(
    eng: &Engine,
    t: VtreeIdx,
    vtree: &Vtree,
    levels: &mut [TddLevel],
    tables: &mut ClauseTables<'_>,
) -> Result<bool, OperationError> {
    let lim = eng.limits();
    let t_idx = t.idx();
    let (left, right) = vtree.children(t);
    let left_idx = left.idx();
    let right_idx = right.idx();
    let base = tables.level_base[t_idx];
    let compute_dt = tables.need_dt[t_idx];
    let left_rel = tables.on_spine[left_idx];
    let right_rel = tables.on_spine[right_idx];
    let both_rel = left_rel && right_rel;
    let left_grid_base = tables.level_base[left_idx];
    let right_grid_base = tables.level_base[right_idx];

    // `old` holds the accumulator's pairs for this level; the emptied
    // `levels[t_idx]` receives the conjoined output.
    let mut old = std::mem::take(&mut levels[t_idx]);
    let k = old.slot_count();
    let in_pairs = old.pairs.len();
    let level = &mut levels[t_idx];
    // At most a c_t and a d_t node per accumulator node.
    let node_cap = if compute_dt { 2 * k } else { k };
    if fits_reserve(tables.spare_nodes.capacity(), node_cap) {
        std::mem::swap(level.nodes.stored_mut(), tables.spare_nodes);
    }
    lim.reserve(level.nodes.stored_mut(), node_cap)?;
    // Worst-case output pairs per input pair: up to 3 c_t pairs for a
    // both-relevant node, plus 1 d_t pair when `compute_dt`. The input pairs
    // are the arena's and at most one more per node, the pair a single-pair
    // node stores inline, which the arena does not hold. The arena is sized
    // at up to twice its input and topped up per node, so the peak never
    // holds a whole-level worst case of 3 or 4 beside the still-live `old`. Twice
    // covers a level with one relevant child whole: sized at the input
    // count, such a level with `d_t` outgrew its arena near its last node
    // and copied it, once per spine level and clause, which a long run of
    // clauses paid as a second pass over the level. Growth near the budget
    // keeps the input count.
    let inputs = in_pairs.saturating_add(k);
    let pair_mult = (if both_rel { 3 } else { 1 }) + usize::from(compute_dt);
    let worst = inputs.saturating_mul(pair_mult);
    lim.begin_level(Some(worst as u128));
    let first_reserve = if lim.bounded_growth() {
        inputs
    } else {
        inputs.saturating_mul(pair_mult.min(2))
    };
    // An empty input pair arena is not read, so it is the output's.
    if old.pairs.is_empty() && old.pairs.implicit().is_none() {
        std::mem::swap(level.pairs.stored_mut(), old.pairs.stored_mut());
    } else if fits_reserve(tables.spare_pairs.capacity(), first_reserve) {
        std::mem::swap(level.pairs.stored_mut(), tables.spare_pairs);
    }
    lim.reserve(level.pairs.stored_mut(), first_reserve)?;
    let ctx = SpineCtx {
        left_grid_base,
        right_grid_base,
        pair_mult,
    };
    let dropped = match (left_rel, right_rel, compute_dt) {
        (true, true, true) => {
            rebuild_nodes::<true, true, true>(eng, &old, ctx, level, base, tables)?
        }
        (true, true, false) => {
            tables.top = Some((t, [0; 2], 0));
            rebuild_nodes::<true, true, false>(eng, &old, ctx, level, base, tables)?
        }
        (true, false, true) => {
            rebuild_nodes::<true, false, true>(eng, &old, ctx, level, base, tables)?
        }
        (true, false, false) => {
            rebuild_nodes::<true, false, false>(eng, &old, ctx, level, base, tables)?
        }
        (false, true, true) => {
            rebuild_nodes::<false, true, true>(eng, &old, ctx, level, base, tables)?
        }
        (false, true, false) => {
            rebuild_nodes::<false, true, false>(eng, &old, ctx, level, base, tables)?
        }
        (false, false, _) => unreachable!("a spine level has a relevant child"),
    };
    let old_value_ref_sides = old.value_ref_sides;
    // A spare the level did not take is freed with `old`.
    old.clear();
    std::mem::swap(old.nodes.stored_mut(), tables.spare_nodes);
    std::mem::swap(old.pairs.stored_mut(), tables.spare_pairs);
    drop(old);
    // Trim the slack the per-node top-up growth left behind.
    level.shrink_arrays();

    // The rebuilt level copied the irrelevant side's pair refs verbatim,
    // inline marginal counts included, but started with zero `value_ref_sides`.
    // The relevant side is never marginal (`require_structure_at` rejects a
    // marginal clause leaf, `plan_cd_map_bases` a marginal spine level), so
    // the input level's markers carry over exactly.
    level.value_ref_sides = old_value_ref_sides;
    Ok(dropped && !both_rel && !compute_dt)
}

/// Rebuild the nodes with the level's relevant children and complement demand fixed.
/// The separate frame keeps level setup out of the specialized pair loops.
/// Each node charges the work clock a unit for each pair it reads and each
/// pair it writes. Returns whether the nodes wrote fewer pairs than they read.
#[inline(never)]
fn rebuild_nodes<const LEFT: bool, const RIGHT: bool, const DT: bool>(
    eng: &Engine,
    old: &TddLevel,
    ctx: SpineCtx,
    level: &mut TddLevel,
    base: usize,
    tables: &mut ClauseTables<'_>,
) -> Result<bool, OperationError> {
    // The one call site keeps `conjoin_node_with_clause` inlined here.
    // An implicit level's node's pairs are generated into `buf`: a stored
    // node's off its range, and the nodes the description implies, which
    // follow none that is stored, in order off the description.
    let nodes = old.nodes.stored();
    let stored = old.stored();
    let (mut buf, mut cursor) = (Vec::new(), None);
    let mut gate = eng.limits().gate();
    let (mut read, mut wrote) = (0usize, 0usize);
    for i in 0..old.slot_count() {
        let inputs: &[ChildPair] = match (nodes.get(i), stored) {
            (Some(node), Some(s)) => s.of(node),
            (Some(node), None) => {
                buf.clear();
                buf.extend(old.pairs_iter_of(node));
                &buf
            }
            (None, _) => old.described_read_next(&mut cursor, i, &mut buf),
        };
        if inputs.is_empty() {
            // Dead accumulator node: nothing emitted, and this is the one
            // write of its map entry.
            tables.cd_map[base + i] = [NO_PRODUCT, NO_PRODUCT];
            continue;
        }
        let written =
            conjoin_node_with_clause::<LEFT, RIGHT, DT>(eng, inputs, ctx, level, base + i, tables)?;
        gate.poll((inputs.len() + written) as u64)?;
        read += inputs.len();
        wrote += written;
    }
    gate.flush()?;
    Ok(wrote < read)
}

/// Emit the node whose pairs sit at `level.pairs[pair_start..]` and return
/// its index, or [`NO_PRODUCT`] when there are none.
///
/// The pairs are not deduplicated: at a level whose subtree includes a
/// marginal child the accumulator's pair list may be a multiset, two equal
/// pairs each carrying one summed-out family's contribution, and dropping one
/// loses count.
fn emit_from(eng: &Engine, level: &mut TddLevel, pair_start: usize) -> Result<u32, OperationError> {
    Ok(finish_node(eng, level, pair_start)?.map_or(NO_PRODUCT, |node| node.0))
}
