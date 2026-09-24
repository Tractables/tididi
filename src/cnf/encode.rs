//! The equivalence encoding: one walk over the reachable internal nodes,
//! bottom-up, emitting each node's definition through the sink.
//!
//! Every node's variable is allocated before any definition, and a definition
//! refers only to lower levels, so a stop between two nodes leaves every node
//! before it defined in full.

use crate::diagram::{ChildDecoder, EncodedChildRef, LeafLabel, Tdd, ValueRef, ZERO};
use crate::limits::{Limits, OperationError};
use crate::vtree::VtreeIdx;
use crate::Engine;

use super::{ClauseSink, CnfEncoding, EncodePoint};

/// A table entry with no literal: an unreachable or deleted node, a summed-out
/// value of count zero, or a node from the stop point on.
pub(super) const NONE: i32 = 0;
/// A table entry for a reachable summed-out value with a positive count, which
/// reads as true and has no variable.
pub(super) const TRUE: i32 = i32::MIN;
/// How many stored slots of a level pass between two [`EncodePoint::Node`]
/// polls.
const NODE_POLL_STRIDE: usize = 1024;

/// Refuse the two values a literal table cannot hold.
pub(super) fn check_literal(literal: i32) -> Result<i32, OperationError> {
    if literal == NONE || literal == TRUE { Err(OperationError::InvalidLiteral(literal)) } else { Ok(literal) }
}

/// One side of a pair that can be true: a literal, or true itself.
#[derive(Clone, Copy)]
enum Side {
    True,
    Literal(i32),
}

/// An all-`fill` table with one entry per reference slot of every level.
pub(super) fn table<T: Clone>(lim: &Limits, f: &Tdd, fill: T) -> Result<Vec<Vec<T>>, OperationError> {
    let mut rows = Vec::new();
    lim.reserve_exact(&mut rows, f.levels.len())?;
    for i in 0..f.levels.len() {
        let mut row = Vec::new();
        lim.try_resize(&mut row, f.reference_slot_count(VtreeIdx(i as u32)), fill.clone())?;
        rows.push(row);
    }
    Ok(rows)
}

/// Whether a side into a summed-out level can be true: its count is positive.
fn value_positive(f: &Tdd, child: VtreeIdx, raw: EncodedChildRef) -> bool {
    match ChildDecoder::marginal().value(raw) {
        ValueRef::Inline(count) => count > 0,
        ValueRef::Slot(slot) => f.levels[child.idx()].marginal_counts().expect("weighted levels are refused")[slot as usize] != 0,
    }
}

/// What a side of a pair reads as, `None` when it cannot be true. `row` is the
/// child level's literal table.
fn side(row: &[i32], raw: EncodedChildRef, marginal: bool) -> Option<Side> {
    if !marginal {
        let literal = row[ChildDecoder::structural().node(raw).idx()];
        debug_assert!(literal != NONE && literal != TRUE, "a reachable child has a literal");
        return Some(Side::Literal(literal));
    }
    match ChildDecoder::marginal().value(raw) {
        ValueRef::Inline(count) => (count > 0).then_some(Side::True),
        ValueRef::Slot(slot) => (row[slot as usize] == TRUE).then_some(Side::True),
    }
}

/// Emit `x ↔ l ∧ r`, leaving out a side that is true.
fn define_and<K: ClauseSink + ?Sized>(sink: &mut K, x: i32, l: Side, r: Side) {
    match (l, r) {
        (Side::Literal(a), Side::Literal(b)) => {
            sink.clause(&[-x, a]);
            sink.clause(&[-x, b]);
            sink.clause(&[x, -a, -b]);
        }
        (Side::True, Side::Literal(w)) | (Side::Literal(w), Side::True) => {
            sink.clause(&[-x, w]);
            sink.clause(&[x, -w]);
        }
        (Side::True, Side::True) => sink.clause(&[x]),
    }
}

/// The encoding of [`CnfScheme::Equivalence`](super::CnfScheme::Equivalence).
pub(super) fn equivalence<K: ClauseSink + ?Sized>(eng: &Engine, f: &Tdd, activation: i32, sink: &mut K) -> Result<CnfEncoding, OperationError> {
    let lim = eng.limits();
    let vtree = &*f.vtree;
    let mut reachable = table(lim, f, false)?;
    if !f.is_zero() {
        reachable[f.output.vtree.idx()][f.output.local.idx()] = true;
        f.propagate_reachability(&mut reachable);
    }
    let mut literals = table(lim, f, NONE)?;

    // Leaves: one true variable, made at the first structural leaf.
    let mut truth = None;
    for (t, var) in vtree.leaf_bottomup() {
        if f.levels[t.idx()].is_marginal() { continue; }
        let t_literal = match truth {
            Some(literal) => literal,
            None => {
                let literal = check_literal(sink.fresh_var())?;
                sink.clause(&[literal]);
                *truth.insert(literal)
            }
        };
        let x = check_literal(sink.leaf_literal(var))?;
        let row = &mut literals[t.idx()];
        row[LeafLabel::One as usize] = t_literal;
        row[LeafLabel::Pos as usize] = x;
        row[LeafLabel::Neg as usize] = -x;
    }
    // Summed-out levels, leaf or internal: a reachable value with a positive
    // count reads as true.
    for (i, level) in f.levels.iter().enumerate() {
        let Some(counts) = level.marginal_counts() else { continue };
        for (slot, &count) in counts.iter().enumerate() {
            if reachable[i][slot] && count != 0 { literals[i][slot] = TRUE; }
        }
    }
    // Every node's variable, before any definition.
    for (t, _, _) in vtree.internal_bottomup() {
        for (i, _) in f.levels[t.idx()].nodes().iter().enumerate() {
            if reachable[t.idx()][i] { literals[t.idx()][i] = check_literal(sink.fresh_var())?; }
        }
    }
    let erasure_certified = erasure_certified(lim, f, &reachable)?;

    let mut stop = None;
    let mut complete = 0;
    let mut encoded = 0u64;
    let mut live: Vec<(Side, Side)> = Vec::new();
    let mut zs: Vec<i32> = Vec::new();
    let mut body: Vec<i32> = Vec::new();
    let mut gate = lim.gate();
    'levels: for (t, left, right) in vtree.internal_bottomup() {
        let at = EncodePoint::Level { level: t, complete };
        if sink.poll(at).is_break() {
            stop = Some(at);
            break;
        }
        let level = &f.levels[t.idx()];
        let (left_marginal, right_marginal) = (f.levels[left.idx()].is_marginal(), f.levels[right.idx()].is_marginal());
        for (i, _) in level.nodes().iter().enumerate() {
            if i % NODE_POLL_STRIDE == 0 {
                let at = EncodePoint::Node { level: t, slot: i };
                if sink.poll(at).is_break() {
                    stop = Some(at);
                    break 'levels;
                }
            }
            gate.poll(1)?;
            if !reachable[t.idx()][i] { continue; }
            let y = literals[t.idx()][i];
            encoded += 1;
            let pairs = level.pairs_of_idx(i);
            live.clear();
            for pair in pairs {
                if pair.left == ZERO.into() || pair.right == ZERO.into() { continue; }
                let sides = side(&literals[left.idx()], pair.left, left_marginal).zip(side(&literals[right.idx()], pair.right, right_marginal));
                if let Some(sides) = sides { lim.try_push(&mut live, sides)?; }
            }
            if live.is_empty() {
                sink.clause(&[-y]);
            } else if pairs.len() == 1 {
                define_and(sink, y, live[0].0, live[0].1);
            } else {
                zs.clear();
                for &(l, r) in &live {
                    let z = check_literal(sink.fresh_var())?;
                    define_and(sink, z, l, r);
                    lim.try_push(&mut zs, z)?;
                }
                body.clear();
                lim.reserve(&mut body, zs.len() + 1)?;
                body.push(-y);
                body.extend_from_slice(&zs);
                sink.clause(&body);
                for &z in &zs { sink.clause(&[y, -z]); }
            }
        }
        complete += 1;
    }
    gate.flush()?;
    lim.discard(live);
    lim.discard(zs);
    lim.discard(body);
    let skipped = match stop {
        Some(at) => forget_from(f, &mut literals, at),
        None => 0,
    };
    Ok(CnfEncoding { activation, literals, reachable, stop, encoded, skipped, erasure_certified })
}

/// Clear every entry of the internal levels from `at` on, in
/// `internal_bottomup` order, and count the entries that held something.
fn forget_from(f: &Tdd, literals: &mut [Vec<i32>], at: EncodePoint) -> u64 {
    let (level, mut start) = match at {
        EncodePoint::Level { level, .. } => (level, 0),
        EncodePoint::Node { level, slot } => (level, slot),
    };
    let mut cleared = 0;
    for (t, _, _) in f.vtree.internal_bottomup().skip_while(|&(t, _, _)| t != level) {
        for entry in &mut literals[t.idx()][start..] {
            if *entry != NONE {
                *entry = NONE;
                cleared += 1;
            }
        }
        start = 0;
    }
    cleared
}

/// Whether reading summed-out children as true keeps each level's reachable
/// nodes disjoint; see [`CnfEncoding::erasure_certified`]. Every reachable
/// pair at a level with a summed-out child is checked.
fn erasure_certified(lim: &Limits, f: &Tdd, reachable: &[Vec<bool>]) -> Result<bool, OperationError> {
    // The first node, at the level being checked, whose pair puts each
    // structural child beside a summed-out side that can be true.
    let mut first: Vec<u32> = Vec::new();
    let mut certified = true;
    'levels: for (t, left, right) in f.vtree.internal_bottomup() {
        let level = &f.levels[t.idx()];
        let (left_marginal, right_marginal) = (f.levels[left.idx()].is_marginal(), f.levels[right.idx()].is_marginal());
        if level.is_marginal() || !(left_marginal || right_marginal) { continue; }
        let (summed, explicit) = if left_marginal { (left, right) } else { (right, left) };
        if !(left_marginal && right_marginal) {
            first.clear();
            lim.try_resize(&mut first, f.reference_slot_count(explicit), u32::MAX)?;
        }
        for (i, _) in level.nodes().iter().enumerate() {
            if !reachable[t.idx()][i] { continue; }
            for pair in level.pairs_of_idx(i) {
                if pair.left == ZERO.into() || pair.right == ZERO.into() { continue; }
                if left_marginal && right_marginal {
                    if value_positive(f, left, pair.left) && value_positive(f, right, pair.right) {
                        certified = false;
                        break 'levels;
                    }
                    continue;
                }
                let (value, structural) = if left_marginal { (pair.left, pair.right) } else { (pair.right, pair.left) };
                if !value_positive(f, summed, value) { continue; }
                let owner = &mut first[ChildDecoder::structural().node(structural).idx()];
                if *owner == u32::MAX {
                    *owner = i as u32;
                } else if *owner != i as u32 {
                    certified = false;
                    break 'levels;
                }
            }
        }
    }
    lim.discard(first);
    Ok(certified)
}
