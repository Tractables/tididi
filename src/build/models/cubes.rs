//! Canonical construction from a set of cubes, each fixing some of the
//! constrained variables and leaving the others free.
//!
//! The atoms of a vtree node follow from its parent's as they do for rows
//! (see `split`), but a node's values are *pieces*: cubes over its
//! constrained variables, every value of a piece in one atom. Below the node
//! over every constrained variable the pieces of a node are disjoint; the
//! input cubes, all in the one atom there, may overlap.
//!
//! A piece is the product of a high part, over the left child's variables,
//! and a low part, over the right child's. Parts of different pieces can
//! overlap where one leaves free what another fixes, so each side's distinct
//! parts are first *refined*: cut into disjoint cubes, the side's pieces,
//! such that every part is the union of the pieces it covers. A parent piece
//! then covers a grid of a high piece times a low piece, all in its atom.
//! The completions of a value in a high piece are the pairs of a low value
//! and the atom it forms with the high one, and since the low pieces are
//! disjoint, two high pieces have equal completions exactly when they meet
//! the same low pieces in the same atoms. So a side's atoms are its pieces
//! numbered by the sets of (other piece, atom) pairs they meet, and the
//! parent's triples are the grid's (atom, high atom, low atom), once each.
//!
//! The refinement walks the parts' bits from the highest down, keeping a bit
//! free while no part fixes it and fixed while every part fixes it alike,
//! and splits the parts at the first bit where they differ, a part that
//! leaves it free going to both sides. A side whose parts are pairwise
//! nested or disjoint within each block of bits, as intervals aligned to
//! powers of two are, refines into few pieces, and parts that are such
//! intervals and pairwise disjoint, or fix every bit, are the pieces
//! already.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::diagram::{Assembly, NodeIdx, Tdd, NEG_LEAF_IDX, ONE_LEAF_IDX, POS_LEAF_IDX};
use crate::limits::{Charged, Limits, OperationError};
use crate::vtree::{VarId, Vtree, VtreeIdx};
use crate::Engine;

use super::layout::Layout;
use super::rows::words_per_row;
use crate::sort::Radix;
use super::split::{Decomposition, Plan};

impl Tdd {
    /// The canonical diagram whose models are the union of `cubes`, each an
    /// assignment to some of `vars`, every other variable of `vtree` free.
    ///
    /// A cube is `2 * w` words, `w = vars.len().div_ceil(64).max(1)`: `w`
    /// words of values, then `w` words that say which variables the cube
    /// fixes, both packed as a row of [`from_models`](Self::from_models)
    /// packs them. Where bit `i` of the second half is set the cube fixes
    /// `vars[i]` to bit `i` of the first; where it is clear the cube leaves
    /// `vars[i]` free, whatever the first half holds there. A cube that
    /// fixes every variable is a row, and `from_cubes` over such cubes is
    /// `from_models` over the rows. Cubes may overlap and repeat. Bits at or
    /// past `vars.len()` are ignored in both halves.
    ///
    /// Empty `cubes` gives the constant-false diagram, and an empty `vars`
    /// with at least one cube gives the constant-true diagram.
    ///
    /// The result is canonical for `vtree`, the diagram `from_models` builds
    /// from every assignment the cubes cover. Construction splits each
    /// vtree node's pieces, disjoint cubes over its variables, into its
    /// children's from the root down, as `from_models` splits a node's
    /// values, and stores the nodes bottom-up. The pieces of a node are the
    /// parts its parent's pieces have there, cut where they overlap, so
    /// cubes that fix a leading part of the bits of each block of variables
    /// — intervals aligned to powers of two, or products of them — take
    /// work in proportion to the cubes and the vtree's depth rather than
    /// to the assignments they cover. Cubes whose free variables cross each
    /// other arbitrarily can cut into many more pieces.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tididi::{Tdd, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// let vtree = Arc::new(Vtree::balanced(4));
    /// let vars = [VarId(1), VarId(2), VarId(3), VarId(4)];
    /// // Values below 8 with bit 3 clear, and the value 13 (bits 0, 2, 3).
    /// let cubes = [0b0000, 0b1000, 0b1101, 0b1111];
    /// let f = Tdd::from_cubes(&vtree, &vars, &cubes)?;
    /// println!("Assignments covered: {}", f.model_count()?);
    /// # assert_eq!(f.model_count()?, 9u32.into());
    /// # tididi::test_helpers::assert_canonical(&f);
    /// # Ok::<(), tididi::OperationError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`OperationError::VariableNotInVtree`] for a variable of `vars` that is
    /// not a leaf of `vtree`, [`OperationError::DuplicateVariable`] for a
    /// repeated one, [`OperationError::RaggedRows`] when `cubes.len()` is not a
    /// multiple of the words per cube, [`OperationError::OverBudget`] for a
    /// refused allocation, [`OperationError::IndexOverflow`] when a level would
    /// outgrow the index that addresses it, and [`OperationError::Stopped`]
    /// when an armed stop fires.
    pub fn from_cubes(vtree: &Arc<Vtree>, vars: &[VarId], cubes: &[u64]) -> Result<Tdd, OperationError> {
        vtree.context().run(|eng| eng.from_cubes(vtree, vars, cubes))
    }
}

impl Engine {
    /// Run [`Tdd::from_cubes`] using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// As [`Tdd::from_cubes`].
    pub fn from_cubes(&self, vtree: &Arc<Vtree>, vars: &[VarId], cubes: &[u64]) -> Result<Tdd, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        let w = words_per_row(vars.len());
        if !cubes.len().is_multiple_of(2 * w) {
            return Err(OperationError::RaggedRows { words: cubes.len(), per_row: 2 * w });
        }
        let mut layout = self.scratch.model_layout.checkout(self);
        layout.prepare_for(lim, vtree, vars)?;
        if cubes.is_empty() || vars.is_empty() {
            return crate::build::constant_on(self, vtree, !cubes.is_empty());
        }
        if cubes.len() / (2 * w) > NodeIdx::MAX_LIVE {
            return Err(OperationError::IndexOverflow);
        }
        let mut scratch = Scratch::default();
        let root = root_pieces(lim, &mut scratch, vars.len(), &layout, cubes, w)?;
        let mut plans = plan(lim, &mut scratch, vtree, &layout, root)?;
        scratch.discard(lim);
        let mut assembly = Assembly::new(self, vtree)?;
        let output = super::fill(self, &mut assembly, vtree, &layout, &mut plans)?;
        // The levels are canonical as built: seat them with nothing to reduce.
        let (levels, _) = assembly.parts_mut();
        Ok(crate::build::seat_canonical(self, vtree, std::mem::take(levels), output))
    }
}

/// Disjoint cubes over one vtree node's constrained variables, below the
/// node over all of them, and the atom of each. At that node the cubes are
/// the input's, which may overlap, all in the one atom.
struct Pieces {
    /// Words per value and per mask, low word first.
    words: usize,
    /// Each piece's fixed values, `words` apiece, clear where it is free.
    value: Vec<u64>,
    /// Each piece's fixed variables, `words` apiece.
    fixed: Vec<u64>,
    /// The atom of each piece.
    atom: Vec<u32>,
    /// How many atoms there are.
    atoms: u32,
}

impl Pieces {
    fn len(&self) -> usize {
        self.atom.len()
    }
}

impl Charged for Pieces {
    fn charged_bytes(&self) -> u64 {
        self.value.charged_bytes() + self.fixed.charged_bytes() + self.atom.charged_bytes()
    }
}

/// A charged vector of `len` copies of `fill`.
fn filled<T: Clone>(lim: &Limits, len: usize, fill: T) -> Result<Vec<T>, OperationError> {
    let mut out = Vec::new();
    lim.try_resize(&mut out, len, fill)?;
    Ok(out)
}

/// The buffers one split reuses from the last.
#[derive(Default)]
struct Scratch {
    /// The radix sort's buffers.
    radix: Radix,
    /// One-word sort keys.
    keys: Vec<u64>,
}

impl Scratch {
    /// Drop the buffers and hand their charge back.
    fn discard(self, lim: &Limits) {
        self.radix.discard(lim);
        lim.discard(self.keys);
    }
}

/// Bits needed to write every index below `count`.
fn index_bits(count: usize) -> usize {
    (usize::BITS - count.saturating_sub(1).leading_zeros()) as usize
}

/// Cubes from which one-word cubes are re-encoded through byte tables, as
/// rows are.
const BYTE_TABLE_MIN_CUBES: usize = 64;

/// The input cubes in leaf order, values clear where they are free, each
/// once.
fn root_pieces(
    lim: &Limits,
    s: &mut Scratch,
    num_vars: usize,
    layout: &Layout,
    cubes: &[u64],
    w: usize,
) -> Result<Pieces, OperationError> {
    let n = cubes.len() / (2 * w);
    let mut value = filled(lim, n * w, 0u64)?;
    let mut fixed = filled(lim, n * w, 0u64)?;
    let mut gate = lim.gate();
    if w == 1 && n >= BYTE_TABLE_MIN_CUBES {
        // One table per byte of the input word holds where that byte's bits
        // land, so a word is re-encoded by eight lookups.
        let mut table = filled(lim, 8 * 256, 0u64)?;
        for (bit, &to) in layout.position.iter().enumerate() {
            let (byte, within) = (bit / 8, bit % 8);
            for (byte_value, entry) in table[byte * 256..][..256].iter_mut().enumerate() {
                if (byte_value >> within) & 1 == 1 {
                    *entry |= 1u64 << to;
                }
            }
        }
        let map = |word: u64| (0..8).fold(0, |acc, byte| acc | table[byte * 256 + ((word >> (8 * byte)) & 0xff) as usize]);
        let live = if num_vars >= 64 { !0u64 } else { (1u64 << num_vars) - 1 };
        gate.poll(2 * n as u64)?;
        for k in 0..n {
            let care = cubes[2 * k + 1] & live;
            fixed[k] = map(care);
            value[k] = map(cubes[2 * k] & care);
        }
        lim.discard(table);
    } else {
        for k in 0..n {
            gate.poll(2 * w as u64)?;
            let (row, mask) = cubes[2 * k * w..2 * (k + 1) * w].split_at(w);
            for j in 0..w {
                let used = num_vars - j * 64;
                let live = if used >= 64 { !0 } else { (1u64 << used) - 1 };
                let mut care = mask[j] & live;
                let ones = row[j] & care;
                while care != 0 {
                    let bit = care.trailing_zeros() as usize;
                    care &= care - 1;
                    let to = layout.position[j * 64 + bit] as usize;
                    fixed[k * w + to / 64] |= 1u64 << (to % 64);
                    value[k * w + to / 64] |= ((ones >> bit) & 1) << (to % 64);
                }
            }
        }
    }
    gate.flush()?;
    let mut out = Pieces { words: w, value: Vec::new(), fixed: Vec::new(), atom: Vec::new(), atoms: 1 };
    if w == 1 {
        // A one-word cube sorts as its value above its mask, in one word
        // through the radix sort where both fit, else as a pair.
        let distinct = if 2 * num_vars <= 64 {
            let mut keys = value;
            for (key, &mask) in keys.iter_mut().zip(&fixed) {
                *key = *key << num_vars | mask;
            }
            lim.discard(fixed);
            s.radix.sort(lim, &mut keys, 0, 2 * num_vars)?;
            keys.dedup();
            let mask = if num_vars == 0 { 0 } else { !0u64 >> (64 - num_vars) };
            lim.reserve_exact(&mut out.value, keys.len())?;
            lim.reserve_exact(&mut out.fixed, keys.len())?;
            out.value.extend(keys.iter().map(|&key| key >> num_vars));
            out.fixed.extend(keys.iter().map(|&key| key & mask));
            keys.len()
        } else {
            let mut keys: Vec<u128> = Vec::new();
            lim.reserve_exact(&mut keys, n)?;
            keys.extend(value.iter().zip(&fixed).map(|(&v, &f)| (v as u128) << 64 | f as u128));
            lim.discard(value);
            lim.discard(fixed);
            lim.gate().poll(n as u64)?;
            keys.sort_unstable();
            keys.dedup();
            lim.reserve_exact(&mut out.value, keys.len())?;
            lim.reserve_exact(&mut out.fixed, keys.len())?;
            out.value.extend(keys.iter().map(|&key| (key >> 64) as u64));
            out.fixed.extend(keys.iter().map(|&key| key as u64));
            let len = keys.len();
            lim.discard(keys);
            len
        };
        out.atom = filled(lim, distinct, 0u32)?;
        return Ok(out);
    }
    let (order, ids) = distinct(lim, s, 64 * w, w, &value, &fixed)?;
    lim.reserve_exact(&mut out.value, order.len() * w)?;
    lim.reserve_exact(&mut out.fixed, order.len() * w)?;
    for &k in &order {
        out.value.extend_from_slice(&value[k as usize * w..][..w]);
        out.fixed.extend_from_slice(&fixed[k as usize * w..][..w]);
    }
    out.atom = filled(lim, order.len(), 0u32)?;
    for buf in [value, fixed] {
        lim.discard(buf);
    }
    lim.discard(order);
    lim.discard(ids);
    Ok(out)
}

/// The distinct cubes of `(value, fixed)`, `words` apiece and at most
/// `width` bits wide: the first position of each distinct cube in
/// ascending order, and each position's index among them.
fn distinct(
    lim: &Limits,
    s: &mut Scratch,
    width: usize,
    words: usize,
    value: &[u64],
    fixed: &[u64],
) -> Result<(Vec<u32>, Vec<u32>), OperationError> {
    let n = value.len() / words;
    let mut ids = filled(lim, n, 0u32)?;
    let mut firsts = Vec::new();
    lim.reserve_exact(&mut firsts, n)?;
    let index = index_bits(n);
    if words == 1 && 2 * width + index <= 64 {
        // The value above the mask above the position, in one word: a stable
        // radix sort of the cube bits leaves each cube's positions ascending.
        let keys = &mut s.keys;
        keys.clear();
        lim.reserve_exact(keys, n)?;
        keys.extend((0..n).map(|k| ((value[k] << width | fixed[k]) << index) | k as u64));
        s.radix.sort(lim, keys, index, 2 * width)?;
        let mut last = u64::MAX;
        for &key in keys.iter() {
            let (cube, k) = (key >> index, (key & ((1u64 << index) - 1)) as usize);
            if firsts.is_empty() || cube != last {
                firsts.push(k as u32);
                last = cube;
            }
            ids[k] = firsts.len() as u32 - 1;
        }
        return Ok((firsts, ids));
    }
    let at = |k: u32| (&value[k as usize * words..][..words], &fixed[k as usize * words..][..words]);
    let key = |k: u32| {
        let (v, f) = at(k);
        v.iter().rev().chain(f.iter().rev())
    };
    let mut order = Vec::new();
    lim.reserve_exact(&mut order, n)?;
    order.extend(0..n as u32);
    lim.gate().poll(n as u64)?;
    if words == 1 {
        order.sort_unstable_by_key(|&k| (value[k as usize], fixed[k as usize]));
    } else {
        order.sort_unstable_by(|&a, &b| key(a).cmp(key(b)));
    }
    for (i, &k) in order.iter().enumerate() {
        if i == 0 || at(order[i - 1]) != at(k) {
            firsts.push(k);
        }
        ids[k as usize] = firsts.len() as u32 - 1;
    }
    lim.discard(order);
    Ok((firsts, ids))
}

/// The plan of every split vtree node, from the pieces of the node over
/// every constrained variable, indexed by vtree node.
fn plan(
    lim: &Limits,
    s: &mut Scratch,
    vtree: &Vtree,
    layout: &Layout,
    root: Pieces,
) -> Result<Vec<Option<Plan>>, OperationError> {
    let mut plans: Vec<Option<Plan>> = Vec::new();
    lim.reserve_exact(&mut plans, vtree.num_nodes())?;
    plans.resize_with(vtree.num_nodes(), || None);
    let mut pending = Vec::new();
    lim.try_push(&mut pending, (split_node(vtree, layout, vtree.root()), root))?;
    let mut planned = 0u64;
    while let Some((t, pieces)) = pending.pop() {
        lim.check_stop()?;
        if vtree.node(t).is_leaf() {
            plans[t.idx()] = Some(Plan::Leaf(leaf_nodes(lim, &pieces)?));
            lim.discard(pieces);
            continue;
        }
        let (high, low) = vtree.children(t);
        let widths = (layout.count[low.idx()] as usize, layout.count[high.idx()] as usize);
        let (split, low_pieces, high_pieces) = split(lim, s, &pieces, widths)?;
        planned += split.atoms() as u64;
        lim.check_output_cap(planned)?;
        lim.discard(pieces);
        plans[t.idx()] = Some(Plan::Branch(split));
        lim.try_push(&mut pending, (split_node(vtree, layout, low), low_pieces))?;
        lim.try_push(&mut pending, (split_node(vtree, layout, high), high_pieces))?;
    }
    Ok(plans)
}

/// The node at or below the constrained node `t` whose atoms are `t`'s, as
/// `split` finds it for rows.
fn split_node(vtree: &Vtree, layout: &Layout, mut t: VtreeIdx) -> VtreeIdx {
    while !vtree.node(t).is_leaf() {
        let (left, right) = vtree.children(t);
        match (layout.count[left.idx()], layout.count[right.idx()]) {
            (0, _) => t = right,
            (_, 0) => t = left,
            _ => break,
        }
    }
    t
}

/// A leaf's atoms as the leaf nodes that stand for them: a free piece is
/// the true node, and a fixed piece the literal of its value, unless both
/// values share an atom.
fn leaf_nodes(lim: &Limits, pieces: &Pieces) -> Result<Vec<NodeIdx>, OperationError> {
    let mut locals = Vec::new();
    lim.reserve_exact(&mut locals, 2)?;
    if pieces.atoms == 1 && (pieces.len() == 2 || pieces.fixed[0] & 1 == 0) {
        locals.push(ONE_LEAF_IDX);
        return Ok(locals);
    }
    locals.resize(pieces.atoms as usize, ONE_LEAF_IDX);
    for (k, &atom) in pieces.atom.iter().enumerate() {
        locals[atom as usize] = if pieces.value[k] & 1 == 0 { NEG_LEAF_IDX } else { POS_LEAF_IDX };
    }
    Ok(locals)
}

/// Words a value of `width` bits occupies.
fn words_for(width: usize) -> usize {
    width.div_ceil(64).max(1)
}

/// Copy the `width` bits of `src` from bit `start` on into `dst`, low word
/// first, clearing the bits of the last word past `width`.
fn extract(src: &[u64], start: usize, width: usize, dst: &mut [u64]) {
    let (skip, shift) = (start / 64, start % 64);
    for (k, out) in dst.iter_mut().enumerate() {
        let low = src.get(skip + k).copied().unwrap_or(0) >> shift;
        let high = match shift {
            0 => 0,
            _ => src.get(skip + k + 1).copied().unwrap_or(0) << (64 - shift),
        };
        *out = low | high;
    }
    let tail = width % 64;
    if let (true, Some(last)) = (tail != 0, dst.last_mut()) {
        *last &= (1u64 << tail) - 1;
    }
    if width == 0 {
        dst.fill(0);
    }
}

/// One side's distinct parts, the pieces they refine into, and which
/// pieces each part covers.
struct Side {
    /// The distinct part of each parent piece.
    part_of: Vec<u32>,
    /// The pieces, as values and masks of `words` words.
    pieces: Pieces,
    /// Where each part's pieces start in `cover`, with one entry past the
    /// last part.
    starts: Vec<u32>,
    /// The pieces each part covers.
    cover: Vec<u32>,
}

impl Side {
    fn cover(&self, part: u32) -> &[u32] {
        &self.cover[self.starts[part as usize] as usize..self.starts[part as usize + 1] as usize]
    }

    fn discard(self, lim: &Limits) {
        lim.discard(self.part_of);
        lim.discard(self.starts);
        lim.discard(self.cover);
    }
}

/// Split a node's pieces into its children's, whose widths are `(low,
/// high)` bits: the low part of a piece is over its right child's
/// variables.
///
/// A parent piece covers a grid of child pieces, and the grid can hold as
/// many cells as the values it stands for, so it is never listed. One side,
/// the *coarse* one, lists instead the other side's parts each of its
/// pieces meets, with their atoms: the parts of the parent pieces over it,
/// fewer than the cells. Pieces that list the same parts have equal
/// completions; lists that differ are expanded into the other side's pieces
/// once each, and those sets number the coarse side's atoms. The other
/// side's atoms then refine one partition of its pieces by each of those
/// sets, which also lists the triples. The coarse side is the one whose
/// pieces meet the fewer parts.
fn split(
    lim: &Limits,
    s: &mut Scratch,
    parent: &Pieces,
    (low_width, high_width): (usize, usize),
) -> Result<(Decomposition, Pieces, Pieces), OperationError> {
    let n = parent.len();
    let mut low = side(lim, s, parent, 0, low_width)?;
    let mut high = side(lim, s, parent, low_width, high_width)?;
    if low.pieces.len() > NodeIdx::MAX_LIVE || high.pieces.len() > NodeIdx::MAX_LIVE {
        return Err(OperationError::IndexOverflow);
    }
    let incidence = |side: &Side| (0..n).map(|e| side.cover(side.part_of[e]).len()).sum::<usize>();
    let (high_incidence, low_incidence) = (incidence(&high), incidence(&low));
    let high_coarse = high_incidence <= low_incidence;
    let (coarse, fine) = if high_coarse { (&high, &low) } else { (&low, &high) };
    let (coarse_atom, sets) = coarse_atoms(lim, s, parent, coarse, fine, high_incidence.min(low_incidence))?;
    let (fine_atom, fine_atoms) = refine_by_sets(lim, fine.pieces.len(), &sets)?;

    // The triples, once each: every coarse atom's set names its pairs.
    let coarse_atoms = sets.len() as u32;
    let (high_atoms, low_atoms) = if high_coarse { (coarse_atoms, fine_atoms) } else { (fine_atoms, coarse_atoms) };
    let pairs = sets.iter().enumerate().flat_map(|(a, set)| {
        let fine_atom = &fine_atom;
        set.iter().map(move |&item| {
            let (f, atom) = ((item >> 32) as usize, item as u32);
            let (h, l) = if high_coarse { (a as u32, fine_atom[f]) } else { (fine_atom[f], a as u32) };
            [atom, h, l]
        })
    });
    let triples = sorted_triples(lim, s, sets.items.len(), [parent.atoms, high_atoms, low_atoms], pairs)?;

    let (high_atom, low_atom) = match high_coarse {
        true => (coarse_atom, fine_atom),
        false => (fine_atom, coarse_atom),
    };
    sets.discard(lim);
    high.pieces.atom = high_atom;
    high.pieces.atoms = high_atoms;
    low.pieces.atom = low_atom;
    low.pieces.atoms = low_atoms;
    let (high_pieces, low_pieces) = (std::mem::replace(&mut high.pieces, empty()), std::mem::replace(&mut low.pieces, empty()));
    high.discard(lim);
    low.discard(lim);
    let split = Decomposition::Triples { atoms: parent.atoms as usize, triples };
    Ok((split, low_pieces, high_pieces))
}

/// The `count` triples `each` lists, `[atom, high atom, low atom]` below
/// `atoms` apiece, sorted and once each: packed into one word and radix
/// sorted where the three fit, else into two.
fn sorted_triples(
    lim: &Limits,
    s: &mut Scratch,
    count: usize,
    atoms: [u32; 3],
    each: impl Iterator<Item = [u32; 3]>,
) -> Result<Vec<[u32; 3]>, OperationError> {
    let bits = atoms.map(|a| index_bits(a as usize));
    let mut triples = Vec::new();
    let mut gate = lim.gate();
    gate.poll(count as u64)?;
    if bits.iter().sum::<usize>() <= 64 {
        let (mid, low) = (bits[1] + bits[2], bits[2]);
        let keys = &mut s.keys;
        keys.clear();
        lim.reserve_exact(keys, count)?;
        keys.extend(each.map(|[a, h, l]| (a as u64) << mid | (h as u64) << low | l as u64));
        s.radix.sort(lim, keys, 0, mid + bits[0])?;
        keys.dedup();
        let mask = |width: usize| if width == 0 { 0 } else { !0u64 >> (64 - width) };
        lim.reserve_exact(&mut triples, keys.len())?;
        triples.extend(keys.iter().map(|&k| [(k >> mid) as u32, ((k >> low) & mask(bits[1])) as u32, (k & mask(low)) as u32]));
    } else {
        let mut keys: Vec<u128> = Vec::new();
        lim.reserve_exact(&mut keys, count)?;
        keys.extend(each.map(|[a, h, l]| (a as u128) << 64 | (h as u128) << 32 | l as u128));
        keys.sort_unstable();
        keys.dedup();
        lim.reserve_exact(&mut triples, keys.len())?;
        triples.extend(keys.iter().map(|&k| [(k >> 64) as u32, (k >> 32) as u32, k as u32]));
        lim.discard(keys);
    }
    gate.flush()?;
    Ok(triples)
}

/// No pieces.
fn empty() -> Pieces {
    Pieces { words: 1, value: Vec::new(), fixed: Vec::new(), atom: Vec::new(), atoms: 0 }
}

/// Lists of items, one after another.
struct Lists {
    /// Where each list starts, with one entry past the last list.
    starts: Vec<u32>,
    /// The items.
    items: Vec<u64>,
}

impl Lists {
    fn len(&self) -> usize {
        self.starts.len() - 1
    }

    fn get(&self, k: usize) -> &[u64] {
        &self.items[self.starts[k] as usize..self.starts[k + 1] as usize]
    }

    fn iter(&self) -> impl Iterator<Item = &[u64]> {
        (0..self.len()).map(|k| self.get(k))
    }

    fn discard(self, lim: &Limits) {
        lim.discard(self.starts);
        lim.discard(self.items);
    }

    /// `count` lists from their `total` `(list, item)` entries, grouped by
    /// list in ascending order.
    fn gather(
        lim: &Limits,
        count: usize,
        total: usize,
        entries: impl Iterator<Item = (u32, u64)>,
    ) -> Result<Lists, OperationError> {
        let mut out = Lists { starts: Vec::new(), items: Vec::new() };
        lim.reserve_exact(&mut out.starts, count + 1)?;
        lim.reserve_exact(&mut out.items, total)?;
        for (list, item) in entries {
            while out.starts.len() <= list as usize {
                out.starts.push(out.items.len() as u32);
            }
            out.items.push(item);
        }
        while out.starts.len() <= count {
            out.starts.push(out.items.len() as u32);
        }
        Ok(out)
    }
}

/// The atoms of the coarse side's pieces, and the set of `(fine piece,
/// parent atom)` pairs, packed high and low, that each atom's pieces meet.
fn coarse_atoms(
    lim: &Limits,
    s: &mut Scratch,
    parent: &Pieces,
    coarse: &Side,
    fine: &Side,
    incidence: usize,
) -> Result<(Vec<u32>, Lists), OperationError> {
    let n = parent.len();
    // Each coarse piece's (fine part, atom) pairs, ascending, once each:
    // packed into one word and radix sorted where the three fit.
    let bits = [coarse.pieces.len(), fine.starts.len() - 1, parent.atoms as usize].map(index_bits);
    let mut gate = lim.gate();
    gate.poll(incidence as u64)?;
    let lists = if bits.iter().sum::<usize>() <= 64 {
        let (piece, part) = (bits[1] + bits[2], bits[2]);
        let keys = &mut s.keys;
        keys.clear();
        lim.reserve_exact(keys, incidence)?;
        for e in 0..n {
            let pair = (fine.part_of[e] as u64) << part | parent.atom[e] as u64;
            keys.extend(coarse.cover(coarse.part_of[e]).iter().map(|&c| (c as u64) << piece | pair));
        }
        s.radix.sort(lim, keys, 0, piece + bits[0])?;
        keys.dedup();
        let atom_mask = if part == 0 { 0 } else { !0u64 >> (64 - part) };
        let part_mask = if bits[1] == 0 { 0 } else { !0u64 >> (64 - bits[1]) };
        let entries = keys.iter().map(|&k| ((k >> piece) as u32, ((k >> part) & part_mask) << 32 | (k & atom_mask)));
        Lists::gather(lim, coarse.pieces.len(), keys.len(), entries)?
    } else {
        let mut entries: Vec<u128> = Vec::new();
        lim.reserve_exact(&mut entries, incidence)?;
        for e in 0..n {
            let pair = (fine.part_of[e] as u128) << 32 | parent.atom[e] as u128;
            entries.extend(coarse.cover(coarse.part_of[e]).iter().map(|&c| (c as u128) << 64 | pair));
        }
        entries.sort_unstable();
        entries.dedup();
        let lists = Lists::gather(lim, coarse.pieces.len(), entries.len(), entries.iter().map(|&g| ((g >> 64) as u32, g as u64)))?;
        lim.discard(entries);
        lists
    };
    let (list_id, firsts) = number_lists(lim, &lists)?;

    // Each distinct list as the fine pieces its parts cover, which lists
    // that differ can share.
    let mut expanded = Lists { starts: Vec::new(), items: Vec::new() };
    lim.reserve_exact(&mut expanded.starts, firsts.len() + 1)?;
    let total: usize = firsts.iter().flat_map(|&k| lists.get(k as usize)).map(|&item| fine.cover((item >> 32) as u32).len()).sum();
    gate.poll(total as u64)?;
    lim.reserve_exact(&mut expanded.items, total)?;
    expanded.starts.push(0);
    for &k in &firsts {
        let start = expanded.items.len();
        for &item in lists.get(k as usize) {
            let (part, atom) = ((item >> 32) as u32, item & 0xffff_ffff);
            expanded.items.extend(fine.cover(part).iter().map(|&f| (f as u64) << 32 | atom));
        }
        // A part's cover ascends, so a list of one part, or of parts whose
        // covers follow each other, is in order already.
        let set = &mut expanded.items[start..];
        if !set.is_sorted() {
            set.sort_unstable();
        }
        let mut kept = 0;
        for at in 0..set.len() {
            if at == 0 || set[at] != set[kept - 1] {
                set[kept] = set[at];
                kept += 1;
            }
        }
        expanded.items.truncate(start + kept);
        expanded.starts.push(expanded.items.len() as u32);
    }
    gate.flush()?;
    lists.discard(lim);
    let (set_id, set_firsts) = number_lists(lim, &expanded)?;
    let mut atom = Vec::new();
    lim.reserve_exact(&mut atom, list_id.len())?;
    atom.extend(list_id.iter().map(|&l| set_id[l as usize]));
    lim.discard(list_id);
    lim.discard(firsts);
    lim.discard(set_id);
    // Each atom's set, once.
    let mut sets = Lists { starts: Vec::new(), items: Vec::new() };
    lim.reserve_exact(&mut sets.starts, set_firsts.len() + 1)?;
    sets.starts.push(0);
    for &k in &set_firsts {
        let set = expanded.get(k as usize);
        lim.reserve(&mut sets.items, set.len())?;
        sets.items.extend_from_slice(set);
        sets.starts.push(sets.items.len() as u32);
    }
    lim.discard(set_firsts);
    expanded.discard(lim);
    Ok((atom, sets))
}

/// The atoms of `count` fine pieces: two share an atom when every set
/// holds both or neither, and both beside the same parent atom. Each set
/// splits the classes it meets, by parent atom. Returns each piece's atom,
/// numbered by first piece, and how many there are.
fn refine_by_sets(lim: &Limits, count: usize, sets: &Lists) -> Result<(Vec<u32>, u32), OperationError> {
    let mut class = filled(lim, count, 0u32)?;
    // Per class, the last set that met it, the first atom beside which it
    // met it there, and the class those pieces moved to. A class the same
    // set meets beside a second atom is rare, and goes through `more`,
    // emptied only by a set that uses it.
    let mut met: Vec<[u32; 3]> = Vec::new();
    lim.reserve(&mut met, count.min(sets.items.len()) + 1)?;
    met.push([u32::MAX; 3]);
    let mut more: FxHashMap<u64, u32> = FxHashMap::default();
    let mut gate = lim.gate();
    for (stamp, set) in sets.iter().enumerate() {
        gate.poll(set.len() as u64)?;
        let stamp = stamp as u32;
        let mut spilled = false;
        for &item in set {
            let (f, atom) = ((item >> 32) as usize, item as u32);
            let c = class[f] as usize;
            let [last, first_atom, first_to] = met[c];
            let to = if last == stamp && first_atom == atom {
                first_to
            } else if last == stamp {
                if !spilled {
                    more.clear();
                    spilled = true;
                }
                let key = (c as u64) << 32 | atom as u64;
                match more.get(&key) {
                    Some(&to) => to,
                    None => {
                        let to = met.len() as u32;
                        lim.try_push(&mut met, [u32::MAX; 3])?;
                        lim.reserve_map(&mut more, 1)?;
                        more.insert(key, to);
                        to
                    }
                }
            } else {
                let to = met.len() as u32;
                lim.try_push(&mut met, [u32::MAX; 3])?;
                met[c] = [stamp, atom, to];
                to
            };
            class[f] = to;
        }
    }
    gate.flush()?;
    lim.discard(more);
    let classes = met.len() as u32;
    lim.discard(met);
    let mut number = filled(lim, classes as usize, u32::MAX)?;
    let mut atoms = 0u32;
    for c in class.iter_mut() {
        if number[*c as usize] == u32::MAX {
            number[*c as usize] = atoms;
            atoms += 1;
        }
        *c = number[*c as usize];
    }
    lim.discard(number);
    Ok((class, atoms))
}

/// Number lists by content, in order of first appearance, every list
/// non-empty. Returns each list's number and the first list of each
/// number.
fn number_lists(lim: &Limits, lists: &Lists) -> Result<(Vec<u32>, Vec<u32>), OperationError> {
    let count = lists.len();
    let mut ids = Vec::new();
    lim.reserve_exact(&mut ids, count)?;
    let mut head: FxHashMap<u64, u32> = FxHashMap::default();
    lim.reserve_map(&mut head, count)?;
    let mut next: Vec<u32> = Vec::new();
    let mut first: Vec<u32> = Vec::new();
    let mut gate = lim.gate();
    for k in 0..count {
        let list = lists.get(k);
        gate.poll(list.len() as u64)?;
        debug_assert!(!list.is_empty(), "every piece meets another");
        let hash = list.iter().fold(0x243f_6a88_85a3_08d3u64, |h, &x| (h.rotate_left(5) ^ x).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95));
        let chain = head.get(&hash).copied().unwrap_or(u32::MAX);
        let mut candidate = chain;
        while candidate != u32::MAX && lists.get(first[candidate as usize] as usize) != list {
            candidate = next[candidate as usize];
        }
        if candidate == u32::MAX {
            candidate = first.len() as u32;
            lim.try_push(&mut first, k as u32)?;
            lim.try_push(&mut next, chain)?;
            head.insert(hash, candidate);
        }
        ids.push(candidate);
    }
    gate.flush()?;
    lim.discard(head);
    lim.discard(next);
    Ok((ids, first))
}

/// The parts of the parent's pieces over bits `start..start + width`, their
/// distinct values, and the pieces those refine into.
fn side(lim: &Limits, s: &mut Scratch, parent: &Pieces, start: usize, width: usize) -> Result<Side, OperationError> {
    let n = parent.len();
    let (w, pw) = (words_for(width), parent.words);
    let mut value = filled(lim, n * w, 0u64)?;
    let mut fixed = filled(lim, n * w, 0u64)?;
    let mut gate = lim.gate();
    for e in 0..n {
        gate.poll(w as u64)?;
        extract(&parent.value[e * pw..][..pw], start, width, &mut value[e * w..][..w]);
        extract(&parent.fixed[e * pw..][..pw], start, width, &mut fixed[e * w..][..w]);
    }
    gate.flush()?;
    let (firsts, part_of) = distinct(lim, s, width, w, &value, &fixed)?;
    let mut parts = Pieces { words: w, value: Vec::new(), fixed: Vec::new(), atom: Vec::new(), atoms: 0 };
    lim.reserve_exact(&mut parts.value, firsts.len() * w)?;
    lim.reserve_exact(&mut parts.fixed, firsts.len() * w)?;
    for &k in &firsts {
        parts.value.extend_from_slice(&value[k as usize * w..][..w]);
        parts.fixed.extend_from_slice(&fixed[k as usize * w..][..w]);
    }
    lim.discard(value);
    lim.discard(fixed);
    lim.discard(firsts);
    let (pieces, starts, cover) = refine(lim, width, parts)?;
    Ok(Side { part_of, pieces, starts, cover })
}

/// Cut distinct cubes of `width` bits, ascending by value, into disjoint
/// pieces, each cube the union of the pieces it covers. Returns the pieces
/// (their atoms unset), and where each cube's pieces start in the list of
/// covered pieces, with that list.
fn refine(lim: &Limits, width: usize, parts: Pieces) -> Result<(Pieces, Vec<u32>, Vec<u32>), OperationError> {
    let (k, w) = (parts.value.len() / parts.words, parts.words);
    let full = |j: usize| {
        let bits = width.saturating_sub(64 * j).min(64);
        if bits == 64 { !0u64 } else { (1u64 << bits) - 1 }
    };
    // The parts ascend by value. Where each fixes a leading run of the bits
    // and leaves the rest free, an interval aligned to a power of two, and
    // each such interval ends before the next begins, they are disjoint
    // already.
    let aligned_and_apart = w == 1 && {
        let mut end = 0u128;
        (0..k).all(|p| {
            let (value, free) = (parts.value[p], !parts.fixed[p] & full(0));
            let apart = free & free.wrapping_add(1) == 0 && (p == 0 || value as u128 >= end);
            end = value as u128 + free as u128 + 1;
            apart
        })
    };
    if aligned_and_apart || (0..k).all(|p| (0..w).all(|j| parts.fixed[p * w + j] == full(j))) {
        // Distinct cubes that fix every bit are disjoint already.
        let mut starts = Vec::new();
        lim.reserve_exact(&mut starts, k + 1)?;
        starts.extend(0..=k as u32);
        let mut cover = Vec::new();
        lim.reserve_exact(&mut cover, k)?;
        cover.extend(0..k as u32);
        let mut parts = parts;
        parts.atom = filled(lim, k, 0u32)?;
        return Ok((parts, starts, cover));
    }

    // Depth first over the bits, highest first, each frame a set of cubes
    // that agree on the bits above `bit` and the piece they share there.
    struct Frame {
        items: (usize, usize),
        bit: usize,
        value: usize,
    }
    let mut arena: Vec<u32> = Vec::new();
    lim.reserve(&mut arena, k)?;
    arena.extend(0..k as u32);
    // Path values and masks of the open frames, two `w`-word slots apiece.
    let mut paths: Vec<u64> = Vec::new();
    lim.try_resize(&mut paths, 2 * w, 0u64)?;
    let mut stack = vec![Frame { items: (0, k), bit: width, value: 0 }];
    let mut out = Pieces { words: w, value: Vec::new(), fixed: Vec::new(), atom: Vec::new(), atoms: 0 };
    // Each piece's cubes, as ranges of the arena.
    let mut members: Vec<(u32, u32)> = Vec::new();
    let (mut or_f, mut and_f, mut or_v, mut and_v) = (vec![0u64; w], vec![0u64; w], vec![0u64; w], vec![0u64; w]);
    let mut split_bits = vec![0u64; w];
    let mut masks = vec![0u64; w];
    let mut gate = lim.gate();
    while let Some(frame) = stack.pop() {
        let (from, len) = frame.items;
        gate.poll(len as u64)?;
        let below = |j: usize| {
            let bits = frame.bit.saturating_sub(64 * j).min(64);
            if bits == 64 { !0u64 } else { (1u64 << bits) - 1 }
        };
        or_f.fill(0);
        and_f.fill(!0);
        or_v.fill(0);
        and_v.fill(!0);
        for &p in &arena[from..from + len] {
            let p = p as usize;
            for j in 0..w {
                let (f, v) = (parts.fixed[p * w + j], parts.value[p * w + j]);
                or_f[j] |= f;
                and_f[j] &= f;
                or_v[j] |= v;
                and_v[j] &= v;
            }
        }
        for j in 0..w {
            and_f[j] &= below(j);
            or_f[j] &= below(j);
            split_bits[j] = ((or_f[j] & !and_f[j]) | (and_f[j] & (or_v[j] ^ and_v[j]))) & below(j);
        }
        if len > 1 && (0..w).all(|j| and_f[j] == below(j)) {
            // Every cube here fixes every bit left, so the cubes of one
            // value below the path are one piece, and different values are
            // disjoint: no bit needs splitting one at a time.
            let start = arena.len();
            lim.reserve(&mut arena, len)?;
            arena.extend_from_within(from..from + len);
            for (j, mask) in masks.iter_mut().enumerate() {
                *mask = below(j);
            }
            let values = &parts.value;
            let lower = |p: u32| {
                let masks = &masks;
                (0..w).rev().map(move |j| values[p as usize * w + j] & masks[j])
            };
            arena[start..].sort_unstable_by(|&a, &b| lower(a).cmp(lower(b)));
            let mut at = start;
            while at < arena.len() {
                let mut end = at + 1;
                while end < arena.len() && lower(arena[end]).eq(lower(arena[at])) {
                    end += 1;
                }
                let p = arena[at] as usize;
                lim.reserve_exact(&mut out.value, w)?;
                lim.reserve_exact(&mut out.fixed, w)?;
                for j in 0..w {
                    out.value.push(paths[frame.value + j] | (parts.value[p * w + j] & below(j)));
                }
                for j in 0..w {
                    out.fixed.push(paths[frame.value + w + j] | below(j));
                }
                lim.try_push(&mut members, (at as u32, (end - at) as u32))?;
                at = end;
            }
            continue;
        }
        let top = (0..w).rev().find(|&j| split_bits[j] != 0).map(|j| 64 * j + 63 - split_bits[j].leading_zeros() as usize);
        // The bits above the split, or every bit left, are shared: fixed
        // where every cube fixes them, alike, and free where none does.
        let lowest = top.map_or(0, |s| s + 1);
        let path = frame.value;
        for j in 0..w {
            let keep = below(j) & !lower_bits(j, lowest);
            paths[path + w + j] |= and_f[j] & keep;
            paths[path + j] |= or_v[j] & and_f[j] & keep;
        }
        match top {
            None => {
                lim.reserve_exact(&mut out.value, w)?;
                lim.reserve_exact(&mut out.fixed, w)?;
                out.value.extend_from_slice(&paths[path..path + w]);
                out.fixed.extend_from_slice(&paths[path + w..path + 2 * w]);
                lim.try_push(&mut members, (from as u32, len as u32))?;
            }
            Some(s) => {
                let (j, b) = (s / 64, s % 64);
                // A cube free at the split goes to both sides.
                for side in [1u64, 0] {
                    let start = arena.len();
                    for i in from..from + len {
                        let p = arena[i] as usize;
                        let (f, v) = (parts.fixed[p * w + j] >> b & 1, parts.value[p * w + j] >> b & 1);
                        if f == 0 || v == side {
                            lim.try_push(&mut arena, p as u32)?;
                        }
                    }
                    let slot = paths.len();
                    lim.reserve(&mut paths, 2 * w)?;
                    paths.extend_from_within(path..path + 2 * w);
                    paths[slot + w + j] |= 1 << b;
                    paths[slot + j] |= side << b;
                    stack.push(Frame { items: (start, arena.len() - start), bit: s, value: slot });
                }
            }
        }
    }
    gate.flush()?;

    // Invert the members into each cube's covered pieces.
    let mut starts = filled(lim, k + 1, 0u32)?;
    for &(from, len) in &members {
        for &p in &arena[from as usize..(from + len) as usize] {
            starts[p as usize + 1] += 1;
        }
    }
    for p in 0..k {
        starts[p + 1] += starts[p];
    }
    let mut cover = filled(lim, starts[k] as usize, 0u32)?;
    let mut at = Vec::new();
    lim.reserve_exact(&mut at, k + 1)?;
    at.extend_from_slice(&starts);
    for (piece, &(from, len)) in members.iter().enumerate() {
        for &p in &arena[from as usize..(from + len) as usize] {
            cover[at[p as usize] as usize] = piece as u32;
            at[p as usize] += 1;
        }
    }
    out.atom = filled(lim, members.len(), 0u32)?;
    lim.discard(at);
    lim.discard(arena);
    lim.discard(paths);
    lim.discard(members);
    lim.discard(parts);
    Ok((out, starts, cover))
}

/// The bits of word `j` below bit `bit` of a multi-word value.
fn lower_bits(j: usize, bit: usize) -> u64 {
    let bits = bit.saturating_sub(64 * j).min(64);
    if bits == 64 { !0 } else { (1u64 << bits) - 1 }
}

#[cfg(test)]
#[path = "tests/cubes.rs"]
mod tests;
