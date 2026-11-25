//! The pointwise sum of two count diagrams on one vtree.
//!
//! A diagram denotes a count for every assignment of its structural
//! variables, the ones no summed-out level holds: the number of assignments
//! of the summed-out variables that extend it to a model. A pair's count is
//! the product of its sides' and a node's the sum of its pairs', a summed-out
//! side contributing its value. The sum `f + g` adds the two counts.
//!
//! The levels split three ways, the same for both operands, which must have
//! summed out the same levels:
//!
//! - a **summed-out level** holds `f`'s values and then `g`'s, and `g`'s
//!   references to a value slot move past `f`'s;
//! - a **Boolean level**, a structural one with no summed-out level under it,
//!   is the overlay of `or` (`overlay.rs`): each node of either operand is
//!   split by the other operand's nodes into disjoint pieces;
//! - a **count level**, a structural one with a summed-out level under it,
//!   holds a copy of every node of `f` and every node of `g`, each pair
//!   rewritten onto the pieces of its sides. Its nodes need not be disjoint:
//!   a count level's nodes already overlap wherever the summed-out variables
//!   told them apart, and the copies of `f` and of `g` are told apart the
//!   same way.
//!
//! The output node holds the pairs of both operands' output nodes. A
//! structural side two of its pairs share is where the counts add: where the
//! other side is summed out, reduction's pair fusion merges the two pairs
//! into one whose value is the sum; elsewhere twin contraction merges the
//! copies the pairs name, so a level stays a copy of each operand only where
//! the two differ.
//!
//! With no summed-out level every count is zero or one, and the sum is the
//! disjunction of two functions with no common model: the overlay's
//! disjunction, refusing a cell of both operands at the root.

use std::borrow::Cow;
use std::sync::Arc;

use crate::Engine;
use crate::diagram::{
    Assembly, ChildDecoder, ChildPair, ChildSide, CountOverflow, EncodedChildRef, LEAF_COUNTS,
    NodeIdx, Tdd, TddLevel, TddNodeId, ValueRef,
};
use crate::limits::{Limits, OperationError, PollGate};
use crate::reduce::ReductionPlan;
use crate::value::{COUNT_OVERFLOW, Count, CountRead, CountVec};
use crate::vtree::{Vtree, VtreeIdx};

use super::overlay::{ABSENT, LevelOverlay, Overlay, Pieces, Scratch, leaf_index, overlay_on};

/// Return `f + g`, the pointwise sum of two count diagrams on one vtree or
/// on projections of one home.
///
/// A diagram denotes a count for each assignment of its structural
/// variables: one for a model and zero otherwise while no level is summed
/// out, and after [`Tdd::marginalize`] the number of assignments of
/// the summed-out variables that extend it to a model. The result's count is
/// `f`'s plus `g`'s on every assignment. For `f` and `g` summed out from
/// Boolean functions `F` and `G` over the same subtrees, it is `F ⊎ G`
/// summed out over them: the bag union of their models, which counts a model
/// of both twice, as `F ∨ G` does not.
///
/// Both operands must have summed out the same levels. With no summed-out
/// level, every count is zero or one, so the operands must share no model
/// and the sum is their disjunction, [`or`](crate::or). A false operand
/// returns the other, whatever its levels.
///
/// On projections of one home ([`Vtree::home`](crate::Vtree::home)) an
/// operand records only the variables [`Tdd::marginalize`] summed out on its
/// own projection, as regions; a summed-out variable its projection lacks
/// stays free, and a region whose count is `2^k` everywhere for its `k`
/// variables is dropped, its variables free. Operands on two projections
/// must record the same regions, each over the same variables in both:
/// they are added on the projection on both leaf sets, which holds each
/// region once. Operands that record different regions are refused, even
/// where one recorded none because its region was dropped; operands of
/// which neither records a summed-out variable are added as a disjunction.
/// Operands with no common model can be added before the sum-out, and the
/// sum summed out.
///
/// Uses the shared vtree's execution context automatically.
///
/// Both operands are consumed on success and on error. The result has the
/// operands' summed-out levels and keeps their weights. Of two satisfiable
/// operands without a summed-out level it is not minimized;
/// [`Engine::minimize`] certifies it. With a false operand or a summed-out
/// level it is minimized, and certified only where no level is summed out.
///
/// ```
/// use tididi::vtree::{Home, VarId};
/// use tididi::{add, or, Tdd, Vtree};
///
/// // With nothing summed out, two functions with no common model add to
/// // their disjunction.
/// let vtree = Home::new(Vtree::balanced(4));
/// let a = Tdd::cube(&vtree, [1, 2])?;
/// let b = Tdd::cube(&vtree, [-1, 3])?;
/// assert!(add(a.clone(), b.clone())?.equivalent(&or(a, b)?)?);
///
/// // On projections of a home, x1 summed out of x1 ∨ x3 and of x1 ∧ x4:
/// // the sum is on the projection on x1's region, x3 and x4.
/// let home = Home::new(Vtree::balanced(4));
/// let mut f = Tdd::clause(&Home::new(Vtree::projection(&home, [VarId(1), VarId(3)])?), [1, 3])?;
/// let mut g = Tdd::cube(&Home::new(Vtree::projection(&home, [VarId(1), VarId(4)])?), [1, 4])?;
/// f.marginalize(|v| v == VarId(1))?;
/// g.marginalize(|v| v == VarId(1))?;
/// let sum = add(f, g)?;
/// assert_eq!(sum.vtree().num_leaves(), 2);
/// // Under x3 and x4 true, x1 counts 2 + 1 for each value of the free x2.
/// let mut counter = sum.counter()?;
/// counter.observe([3, 4])?;
/// assert_eq!(counter.model_count()?, 6u32.into());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// The work follows the operands' pairs. A Boolean level, one with no
/// summed-out level under it, splits each operand's nodes by the other's as
/// [`or`](crate::or) does; a level above a summed-out one keeps both
/// operands' nodes, and reduction merges the ones that coincide.
///
/// # Errors
///
/// [`OperationError::VtreeMismatch`] for operands neither on one vtree nor on
/// projections of one home; [`OperationError::MarginalLevel`] naming the lowest level that one
/// operand has summed out and the other has not, an internal summed-out
/// level of an operand that would move to a larger projection, or a region
/// of one operand on two projections that the other does not record over
/// the same variables;
/// [`OperationError::SharedModel`] when neither operand has a summed-out
/// level and they share a model, whose count of two only a summed-out level
/// could hold; [`OperationError::IncompatibleWeights`] for a summed-out
/// level holding weights, which this does not add; and [`OperationError::IndexOverflow`] when a summed-out
/// level's values, `f`'s and `g`'s together, outgrow the slots a pair side
/// addresses, or a level's nodes outgrow a 32-bit index. Counts do not overflow: past `u128` they are held exactly.
/// Allocation refusals and installed limits propagate as
/// [`OperationError::OverBudget`], [`OperationError::Stopped`] and
/// [`OperationError::OutputCap`].
pub fn add(f: Tdd, g: Tdd) -> Result<Tdd, OperationError> {
    let context = Arc::clone(f.context());
    context.run(|eng| eng.add(f, g))
}

impl Engine {
    /// Run [`add`] using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// Returns the operation's errors, plus [`OperationError::Stopped`] or
    /// [`OperationError::OutputCap`] when an installed limit refuses the work.
    pub fn add(&self, f: Tdd, g: Tdd) -> Result<Tdd, OperationError> {
        let _op = self.limits().enter()?;
        add_on(self, f, g)
    }
}

/// `f + g` of two owned operands on one vtree: the overlay, or the sum of
/// summed-out diagrams, minimized.
fn add_on(eng: &Engine, f: Tdd, g: Tdd) -> Result<Tdd, OperationError> {
    if !f.is_zero() && !g.is_zero() && (f.has_marginal_level() || g.has_marginal_level()) && crate::apply::extend::needed([&f, &g]) {
        crate::apply::extend::refuse_other_homes([&f, &g])?;
        refuse_unshared_sums(&f, &g)?;
    }
    if !f.is_zero() && !g.is_zero() && !f.has_marginal_level() && !g.has_marginal_level() {
        return overlay_on(eng, Overlay::Add, f, g);
    }
    let [f, g] = crate::apply::extend::on_one_sharing(eng, [Cow::Owned(f), Cow::Owned(g)])?.map(Cow::into_owned);
    eng.limits().check_stop()?;
    let mut result = match (f.is_zero(), g.is_zero()) {
        (true, _) => g,
        (false, true) => f,
        (false, false) => {
            if f.weight_table().is_some() {
                return Err(OperationError::IncompatibleWeights);
            }
            // A level under a summed-out parent is folded into it, in
            // whatever state it was left.
            let folded = |t: VtreeIdx| {
                f.vtree
                    .node(t)
                    .parent()
                    .is_some_and(|p| summed(&f, p) && summed(&g, p))
            };
            let differs = |t: &VtreeIdx| summed(&f, *t) != summed(&g, *t) && !folded(*t);
            if let Some(t) = f.vtree.bottomup().find(differs) {
                return Err(OperationError::MarginalLevel(t));
            }
            sum_levels(eng, &f, &g)?
        }
    };
    eng.reduce(&mut result, ReductionPlan::default())?;
    crate::apply::extend::trimmed(eng, result)
}

/// What a pair side of a count level is rewritten onto.
enum Side<'a> {
    /// A structural child: the output nodes each operand node splits into.
    Pieces(&'a Pieces),
    /// A summed-out child: `f`'s values where they were, `g`'s slots
    /// `shift` on, past `f`'s.
    Values { shift: u32 },
}

/// How a count level reads the child on one side: its values when it is
/// summed out, else its pieces, `own` for an internal child and `leaf` for a
/// leaf.
fn side_of<'a>(summed: bool, shift: u32, own: &'a Option<Pieces>, leaf: &'a Pieces) -> Side<'a> {
    match summed {
        true => Side::Values { shift },
        false => Side::Pieces(own.as_ref().unwrap_or(leaf)),
    }
}

/// The output references one side of an operand's pair stands for:
/// `pieces` when they are nodes, else the one value reference written into
/// `one`.
fn side_refs<'s>(
    lim: &Limits,
    side: &'s Side,
    raw: EncodedChildRef,
    of_g: bool,
    one: &'s mut [u32; 1],
) -> Result<&'s [u32], OperationError> {
    match side {
        Side::Pieces(p) if of_g => Ok(p.of_g(raw.0)),
        Side::Pieces(p) => Ok(p.of_f(raw.0)),
        Side::Values { shift } => {
            one[0] = match ChildDecoder::marginal().value(raw) {
                ValueRef::Slot(slot) if of_g && *shift != 0 => {
                    ValueRef::slot_raw(lim.next_slot(slot as usize + *shift as usize)?)
                }
                _ => raw.raw(),
            };
            Ok(&one[..])
        }
    }
}

/// Push into `pairs` the pairs of node `a` of `level`, an operand's level
/// (`g`'s when `of_g`), each rewritten onto the output references of its
/// sides, charging the work to the level's `poll`.
fn rewrite_node(
    eng: &Engine,
    level: &TddLevel,
    a: usize,
    (left, right): (&Side, &Side),
    of_g: bool,
    pairs: &mut Vec<ChildPair>,
    poll: &mut PollGate<'_>,
) -> Result<(), OperationError> {
    let lim = eng.limits();
    let (mut one_l, mut one_r) = ([0u32], [0u32]);
    for p in level.pairs_iter_of_idx(a) {
        let xs = side_refs(lim, left, p.left, of_g, &mut one_l)?;
        let ys = side_refs(lim, right, p.right, of_g, &mut one_r)?;
        for &x in xs {
            for &y in ys {
                lim.try_push(
                    pairs,
                    ChildPair::new(EncodedChildRef::from_raw(x), EncodedChildRef::from_raw(y)),
                )?;
            }
        }
        poll.poll((xs.len() * ys.len()) as u64 + 1)?;
    }
    Ok(())
}

/// The value of an operand's output, a slot of its summed-out root level.
fn root_value(level: &TddLevel, slot: NodeIdx) -> Count {
    let counts = level.marginal_counts().expect("a count-marginal root");
    CountRead::from_slot(counts, level.marginal_counts_big(), slot.idx()).to_count()
}

/// The sum of `f` and `g` whose roots are summed out: the root holds the sum
/// of their values.
fn summed_root(eng: &Engine, mut assembly: Assembly<'_>, f: &Tdd, g: &Tdd) -> Result<Tdd, OperationError> {
    let root = f.vtree.root();
    let value = count_sum(
        root_value(&f.levels[root.idx()], f.output.local),
        root_value(&g.levels[root.idx()], g.output.local),
    );
    let mut column = CountVec::try_with_capacity(eng, 1)?;
    column.push(eng, value)?;
    let (counts, big) = column.into_parts();
    assembly.levels_mut()[root.idx()].become_marginal(counts, big);
    let output = TddNodeId { vtree: root, local: NodeIdx(0) };
    assembly.finish_asserted(output, None)
}

/// The sum of two exact counts.
fn count_sum(a: Count, b: Count) -> Count {
    match (&a, &b) {
        (Count::Fast(x), Count::Fast(y))
            if x.checked_add(*y).is_some_and(|s| s != COUNT_OVERFLOW) =>
        {
            Count::Fast(x + y)
        }
        _ => {
            let big = |c: Count| match c {
                Count::Fast(v) => num_bigint::BigUint::from(v),
                Count::Big(v) => v,
            };
            Count::Big(big(a) + big(b))
        }
    }
}

/// A summed-out level's store holding `f`'s values and then `g`'s: the fast
/// column and its overflow table.
fn joined_store(
    eng: &Engine,
    fl: &TddLevel,
    gl: &TddLevel,
) -> Result<(Vec<u128>, Option<CountOverflow>), OperationError> {
    let (fc, gc) = (
        fl.marginal_counts().unwrap_or(&[]),
        gl.marginal_counts().unwrap_or(&[]),
    );
    let width = fc.len() + gc.len();
    if width > 0 {
        eng.limits().next_slot(width - 1)?;
    }
    let mut counts: Vec<u128> = Vec::new();
    eng.limits().reserve_exact(&mut counts, width)?;
    counts.extend_from_slice(fc);
    counts.extend_from_slice(gc);
    let mut big: Option<CountOverflow> = None;
    for (store, from, at) in [(fl, fc, 0), (gl, gc, fc.len())] {
        for (i, &c) in from.iter().enumerate() {
            if c == COUNT_OVERFLOW {
                let value = store
                    .marginal_counts_big()
                    .and_then(|b| b.get(i))
                    .expect("an overflowed count has its exact value");
                big.get_or_insert_with(CountOverflow::default).try_insert(
                    eng,
                    at + i,
                    value.clone(),
                )?;
            }
        }
    }
    Ok((counts, big))
}

/// Refuse operands on two projections where one has summed out a leaf the
/// other lacks, or a region the other has not with the same anchor and
/// extent: the other reads those variables as true, or as other sums, which
/// the sum cannot add to the region's values.
#[cold]
fn refuse_unshared_sums(f: &Tdd, g: &Tdd) -> Result<(), OperationError> {
    for (a, b) in [(f, g), (g, f)] {
        let (av, bv) = (a.vtree(), b.vtree());
        let lacks = |t: VtreeIdx| match av.node(t).is_region() {
            true => !bv.regions().any(|u| bv.region_at(u) == av.region_at(t)),
            false => bv.node_at_home(av.home_node(t)).is_none_or(|u| !summed(b, u)),
        };
        if let Some(t) = av.bottomup().find(|&t| av.node(t).is_leaf() && summed(a, t) && lacks(t)) {
            return Err(OperationError::MarginalLevel(t));
        }
    }
    Ok(())
}

/// Whether a level of `vtree` is summed out in `f`, which the caller has
/// checked `g` agrees with.
fn summed(f: &Tdd, t: VtreeIdx) -> bool {
    f.levels[t.idx()].is_marginal()
}

/// The sum of two nonfalse unweighted operands that have summed out the
/// same levels, at least one, before its reduction.
fn sum_levels(eng: &Engine, f: &Tdd, g: &Tdd) -> Result<Tdd, OperationError> {
    let vtree: Arc<Vtree> = Arc::clone(&f.vtree);
    let root = vtree.root();
    let lim = eng.limits();
    let n = vtree.num_nodes();
    // Whether a level is summed out or has a summed-out level under it.
    let mut counted: Vec<bool> = Vec::new();
    lim.try_resize(&mut counted, n, false)?;
    for t in vtree.bottomup() {
        counted[t.idx()] = summed(f, t)
            || (!vtree.node(t).is_leaf() && {
                let (l, r) = vtree.children(t);
                counted[l.idx()] || counted[r.idx()]
            });
    }
    // Where `g`'s value slots start at each summed-out level.
    let mut shift: Vec<u32> = Vec::new();
    lim.try_resize(&mut shift, n, 0)?;
    let under_summed = |t: VtreeIdx| vtree.node(t).parent().is_some_and(|p| summed(f, p));

    let mut assembly = Assembly::new(eng, &vtree)?;
    // A summed-out leaf holds its label column, or nothing when every
    // reference to it is inline; it is never shifted.
    for (leaf, _) in vtree.leaf_bottomup() {
        if !summed(f, leaf) {
            continue;
        }
        let column = !under_summed(leaf)
            && [f, g].iter().any(|d| {
                d.levels[leaf.idx()]
                    .marginal_counts()
                    .is_some_and(|c| !c.is_empty())
            });
        let mut store: Vec<u128> = Vec::new();
        if column {
            lim.reserve_exact(&mut store, LEAF_COUNTS.len())?;
            store.extend_from_slice(&LEAF_COUNTS);
        }
        let levels = assembly.levels_mut();
        levels[leaf.idx()].become_marginal(store, None);
    }
    // A region holds values, as a summed-out internal level does; no
    // summed-out level is above it.
    for region in vtree.regions() {
        if region == root {
            return summed_root(eng, assembly, f, g);
        }
        let (fl, gl) = (&f.levels[region.idx()], &g.levels[region.idx()]);
        let (counts, big) = joined_store(eng, fl, gl)?;
        shift[region.idx()] = fl.marginal_counts().map_or(0, <[u128]>::len) as u32;
        assembly.levels_mut()[region.idx()].become_marginal(counts, big);
    }

    let mut built: Vec<Option<Pieces>> = Vec::new();
    lim.reserve_exact(&mut built, n)?;
    built.resize_with(n, || None);
    let leaves = Pieces::leaves();
    let mut spare: Vec<Pieces> = Vec::new();
    let mut scratch = Scratch::default();
    let mut pairs: Vec<ChildPair> = Vec::new();
    let mut out_nodes = 0u64;

    for (t, left, right) in vtree.internal_bottomup() {
        let (fl, gl) = (&f.levels[t.idx()], &g.levels[t.idx()]);
        if summed(f, t) {
            if t == root {
                return summed_root(eng, assembly, f, g);
            }
            let levels = assembly.levels_mut();
            // Under a summed-out parent a level's values are folded into the
            // parent's, and its store stays empty.
            let (counts, big) = match under_summed(t) {
                true => (Vec::new(), None),
                false => joined_store(eng, fl, gl)?,
            };
            shift[t.idx()] = fl.marginal_counts().map_or(0, <[u128]>::len) as u32;
            levels[t.idx()].become_marginal(counts, big);
            continue;
        }
        let take = |built: &mut [Option<Pieces>], child: VtreeIdx| {
            (!vtree.node(child).is_leaf() && !summed(f, child)).then(|| {
                built[child.idx()]
                    .take()
                    .expect("a child level is visited before its parent")
            })
        };
        let (lo, ro) = (take(&mut built, left), take(&mut built, right));
        let ls = side_of(
            summed(f, left),
            shift[left.idx()],
            &lo,
            &leaves[leaf_index(fl, gl, true)],
        );
        let rs = side_of(
            summed(f, right),
            shift[right.idx()],
            &ro,
            &leaves[leaf_index(fl, gl, false)],
        );
        let levels = assembly.levels_mut();
        let out = &mut levels[t.idx()];
        let mut pieces = spare.pop().unwrap_or_default();
        pieces.clear();
        if !counted[left.idx()] && !counted[right.idx()] {
            // A Boolean level: the overlay's disjunction. The root has a
            // summed-out level under it, so it is never one.
            debug_assert!(
                t != root,
                "a Boolean root in a diagram with a summed-out level"
            );
            let (Side::Pieces(lp), Side::Pieces(rp)) = (&ls, &rs) else {
                unreachable!("a Boolean level has structural children")
            };
            (LevelOverlay { fl, gl, lp, rp }).inner_level(
                eng,
                Overlay::Or,
                out,
                &mut scratch,
                &mut pieces,
            )?;
        } else {
            for (child_side, child) in [(ChildSide::Left, left), (ChildSide::Right, right)] {
                if summed(f, child) {
                    out.set_has_value_refs(
                        child_side,
                        fl.has_value_refs(child_side) || gl.has_value_refs(child_side),
                    );
                }
            }
            let mut poll = lim.gate();
            if t == root {
                pairs.clear();
                rewrite_node(
                    eng,
                    fl,
                    f.output.local.idx(),
                    (&ls, &rs),
                    false,
                    &mut pairs,
                    &mut poll,
                )?;
                rewrite_node(
                    eng,
                    gl,
                    g.output.local.idx(),
                    (&ls, &rs),
                    true,
                    &mut pairs,
                    &mut poll,
                )?;
                poll.flush()?;
                crate::diagram::sort_pairs(&mut pairs);
                let local = out.push_node(lim, &pairs)?;
                lim.level_done(out_nodes + 1)?;
                return assembly.finish_asserted(TddNodeId { vtree: root, local }, None);
            }
            // A count level: a copy of each node of `f`, then of `g`.
            let (fw, gw) = (fl.slot_count(), gl.slot_count());
            for (level, width, of_g) in [(fl, fw, false), (gl, gw, true)] {
                pieces.side_mut(of_g).begin_exact(lim, width)?;
                for a in 0..width {
                    pairs.clear();
                    rewrite_node(eng, level, a, (&ls, &rs), of_g, &mut pairs, &mut poll)?;
                    crate::diagram::sort_pairs(&mut pairs);
                    let o = out.push_node(lim, &pairs)?.0;
                    let (f_node, g_node) = if of_g {
                        (ABSENT, a as u32)
                    } else {
                        (a as u32, ABSENT)
                    };
                    pieces.push_piece(lim, o, f_node, g_node, of_g)?;
                    pieces.side_mut(of_g).close(lim)?;
                }
            }
            poll.flush()?;
        }
        out_nodes += pieces.f_part.len() as u64;
        lim.level_done(out_nodes)?;
        built[t.idx()] = Some(pieces);
        spare.extend(lo);
        spare.extend(ro);
    }
    unreachable!("the bottom-up walk ends at the internal root")
}

#[cfg(test)]
#[path = "tests/add.rs"]
mod tests;
