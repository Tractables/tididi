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


use rustc_hash::FxHashMap;

use crate::Engine;
use crate::diagram::{NEG_LEAF_IDX, NodeIdx, ONE_LEAF_IDX, POS_LEAF_IDX, Tdd};
use crate::limits::{Charged, Limits, OperationError};
use crate::vtree::{Home, VarId};

use super::filled;
use super::layout::Layout;
use super::rows::{extract, index_bits, words_per_row};
use super::plan::{Decomposition, Emitted, plan, sorted_triples};
use super::split::dedup_tail;
use crate::sort::Radix;

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
    /// The result is certified ([`is_canonical`](Self::is_canonical)).
    ///
    /// ```
    /// use tididi::vtree::Home;
    /// use tididi::{Tdd, Vtree};
    /// use tididi::vtree::VarId;
    ///
    /// let vtree = Home::new(Vtree::balanced(4));
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
    pub fn from_cubes(
        vtree: &Home,
        vars: &[VarId],
        cubes: &[u64],
    ) -> Result<Tdd, OperationError> {
        vtree
            .context()
            .run(|eng| eng.from_cubes(vtree, vars, cubes))
    }
}

impl Engine {
    /// Run [`Tdd::from_cubes`] using this batch's scratch and resource limits.
    ///
    /// # Errors
    ///
    /// As [`Tdd::from_cubes`].
    pub fn from_cubes(
        &self,
        vtree: &Home,
        vars: &[VarId],
        cubes: &[u64],
    ) -> Result<Tdd, OperationError> {
        let lim = self.limits();
        let _op = lim.enter()?;
        let w = words_per_row(vars.len());
        if !cubes.len().is_multiple_of(2 * w) {
            return Err(OperationError::RaggedRows {
                words: cubes.len(),
                per_row: 2 * w,
            });
        }
        let layout = Layout::new(lim, vtree, vars)?;
        if cubes.is_empty() || vars.is_empty() {
            return crate::build::constant_on(self, vtree, !cubes.is_empty());
        }
        if cubes.len() / (2 * w) > NodeIdx::MAX_LIVE {
            return Err(OperationError::IndexOverflow);
        }
        let mut scratch = Scratch::default();
        let root = root_pieces(lim, &mut scratch, vars.len(), &layout, cubes, w)?;
        let mut plans = plan(
            lim,
            vtree,
            &layout,
            root,
            |pieces| leaf_nodes(lim, pieces),
            |pieces, widths| split(lim, &mut scratch, pieces, widths),
        )?;
        scratch.discard(lim);
        super::emit::fill(self, vtree, &layout, &mut plans)
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
    /// No pieces of `words` words.
    fn empty(words: usize) -> Pieces {
        Pieces {
            words,
            value: Vec::new(),
            fixed: Vec::new(),
            atom: Vec::new(),
            atoms: 0,
        }
    }

    /// The pieces of `value` and `fixed`, `words` apiece, at `firsts`, in
    /// that order, with no atoms yet.
    fn at(lim: &Limits, words: usize, value: &[u64], fixed: &[u64], firsts: &[u32]) -> Result<Pieces, OperationError> {
        let mut out = Pieces::empty(words);
        lim.reserve_exact(&mut out.value, firsts.len() * words)?;
        lim.reserve_exact(&mut out.fixed, firsts.len() * words)?;
        for &k in firsts {
            out.value.extend_from_slice(&value[k as usize * words..][..words]);
            out.fixed.extend_from_slice(&fixed[k as usize * words..][..words]);
        }
        Ok(out)
    }

    fn len(&self) -> usize {
        self.atom.len()
    }
}

impl Charged for Pieces {
    fn charged_bytes(&self) -> u64 {
        self.value.charged_bytes() + self.fixed.charged_bytes() + self.atom.charged_bytes()
    }
}

/// The buffers one split reuses from the last.
#[derive(Default)]
struct Scratch {
    /// The radix sort's buffers.
    radix: Radix,
    /// One-word sort keys.
    keys: Vec<u64>,
    /// Two-word sort keys.
    wide: Vec<u128>,
}

impl Scratch {
    /// Drop the buffers and hand their charge back.
    fn discard(self, lim: &Limits) {
        self.radix.discard(lim);
        lim.discard(self.keys);
        lim.discard(self.wide);
    }
}

/// Cubes from which one-word cubes are re-encoded through byte tables.
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
        let table = layout.byte_table(lim, 1)?;
        let live = lower_bits(0, num_vars);
        gate.poll(2 * n as u64)?;
        for k in 0..n {
            let care = cubes[2 * k + 1] & live;
            fixed[k] = Layout::reencode_word(&table, care);
            value[k] = Layout::reencode_word(&table, cubes[2 * k] & care);
        }
        lim.discard(table);
    } else {
        for k in 0..n {
            gate.poll(2 * w as u64)?;
            let (row, mask) = cubes[2 * k * w..2 * (k + 1) * w].split_at(w);
            for j in 0..w {
                let mut care = mask[j] & lower_bits(j, num_vars);
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
    let (order, ids) = distinct(lim, s, num_vars, w, &value, &fixed)?;
    let mut out = Pieces { atoms: 1, ..Pieces::at(lim, w, &value, &fixed, &order)? };
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
    let index = index_bits(n) as usize;
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
    let at = |k: u32| {
        (
            &value[k as usize * words..][..words],
            &fixed[k as usize * words..][..words],
        )
    };
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

/// A leaf's atoms as the leaf nodes that stand for them: a free piece is
/// the true node, and a fixed piece the literal of its value, unless both
/// values share an atom.
fn leaf_nodes(lim: &Limits, pieces: &Pieces) -> Result<Vec<NodeIdx>, OperationError> {
    let mut locals = Vec::new();
    if pieces.atoms == 1 && (pieces.len() == 2 || pieces.fixed[0] & 1 == 0) {
        lim.try_push(&mut locals, ONE_LEAF_IDX)?;
        return Ok(locals);
    }
    lim.try_resize(&mut locals, pieces.atoms as usize, ONE_LEAF_IDX)?;
    for (k, &atom) in pieces.atom.iter().enumerate() {
        locals[atom as usize] = if pieces.value[k] & 1 == 0 {
            NEG_LEAF_IDX
        } else {
            POS_LEAF_IDX
        };
    }
    Ok(locals)
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
    let incidence = |side: &Side| {
        (0..n)
            .map(|e| side.cover(side.part_of[e]).len())
            .sum::<usize>()
    };
    let (high_incidence, low_incidence) = (incidence(&high), incidence(&low));
    let high_coarse = high_incidence <= low_incidence;
    let (coarse, fine) = if high_coarse {
        (&high, &low)
    } else {
        (&low, &high)
    };
    let (coarse_atom, sets) = coarse_atoms(
        lim,
        s,
        parent,
        coarse,
        fine,
        high_incidence.min(low_incidence),
    )?;
    let (fine_atom, fine_atoms) = refine_by_sets(lim, fine.pieces.len(), &sets)?;

    // The triples, once each: every coarse atom's set names its pairs.
    let coarse_atoms = sets.len() as u32;
    let (high_atoms, low_atoms) = if high_coarse {
        (coarse_atoms, fine_atoms)
    } else {
        (fine_atoms, coarse_atoms)
    };
    // A coarse atom's set at a time: the coarse side's atoms ascend.
    let each = |emit: &mut dyn FnMut([u32; 3])| {
        for (a, set) in sets.iter().enumerate() {
            for &item in set {
                let (f, atom) = ((item >> 32) as usize, item as u32);
                let (h, l) = if high_coarse {
                    (a as u32, fine_atom[f])
                } else {
                    (fine_atom[f], a as u32)
                };
                emit([atom, h, l]);
            }
        }
    };
    let emitted = if high_coarse { Emitted::ByMid } else { Emitted::ByLast };
    let atoms = (parent.atoms, high_atoms, low_atoms);
    let triples = sorted_triples(lim, &mut s.radix, &mut s.keys, &mut s.wide, atoms, emitted, each)?;

    let (high_atom, low_atom) = match high_coarse {
        true => (coarse_atom, fine_atom),
        false => (fine_atom, coarse_atom),
    };
    sets.discard(lim);
    high.pieces.atom = high_atom;
    high.pieces.atoms = high_atoms;
    low.pieces.atom = low_atom;
    low.pieces.atoms = low_atoms;
    let (high_pieces, low_pieces) = (
        std::mem::replace(&mut high.pieces, Pieces::empty(1)),
        std::mem::replace(&mut low.pieces, Pieces::empty(1)),
    );
    high.discard(lim);
    low.discard(lim);
    let split = Decomposition::Triples {
        atoms: parent.atoms as usize,
        triples,
    };
    Ok((split, low_pieces, high_pieces))
}

/// Lists of items, one after another.
#[derive(Default)]
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
        // Every start is at most `total`.
        lim.list_index(total)?;
        let mut out = Lists::default();
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
    // Each coarse piece's (fine part, atom) pairs, ascending, once each.
    let mut gate = lim.gate();
    gate.poll(incidence as u64)?;
    let each = |emit: &mut dyn FnMut([u32; 3])| {
        for e in 0..n {
            for &c in coarse.cover(coarse.part_of[e]) {
                emit([c, fine.part_of[e], parent.atom[e]]);
            }
        }
    };
    let atoms = (coarse.pieces.len() as u32, (fine.starts.len() - 1) as u32, parent.atoms);
    let incidences = sorted_triples(lim, &mut s.radix, &mut s.keys, &mut s.wide, atoms, Emitted::Unordered, each)?;
    let entries = incidences.iter().map(|&[c, part, atom]| (c, u64::from(part) << 32 | u64::from(atom)));
    let lists = Lists::gather(lim, coarse.pieces.len(), incidences.len(), entries)?;
    lim.discard(incidences);
    let (list_id, firsts) = number_lists(lim, &lists)?;
    gate.flush()?;
    let expanded = expand_coarse_lists(lim, &lists, &firsts, fine)?;
    lists.discard(lim);
    let (set_id, set_firsts) = number_lists(lim, &expanded)?;
    let mut atom = Vec::new();
    lim.reserve_exact(&mut atom, list_id.len())?;
    atom.extend(list_id.iter().map(|&l| set_id[l as usize]));
    lim.discard(list_id);
    lim.discard(firsts);
    lim.discard(set_id);
    // Each atom's set, once.
    let mut sets = Lists::default();
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
fn refine_by_sets(
    lim: &Limits,
    count: usize,
    sets: &Lists,
) -> Result<(Vec<u32>, u32), OperationError> {
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
        let hash = list.iter().fold(0x243f_6a88_85a3_08d3u64, |h, &x| {
            (h.rotate_left(5) ^ x).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95)
        });
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
fn side(
    lim: &Limits,
    s: &mut Scratch,
    parent: &Pieces,
    start: usize,
    width: usize,
) -> Result<Side, OperationError> {
    let n = parent.len();
    let (w, pw) = (words_per_row(width), parent.words);
    let mut value = filled(lim, n * w, 0u64)?;
    let mut fixed = filled(lim, n * w, 0u64)?;
    let mut gate = lim.gate();
    for e in 0..n {
        gate.poll(w as u64)?;
        extract(
            &parent.value[e * pw..][..pw],
            start,
            width,
            &mut value[e * w..][..w],
        );
        extract(
            &parent.fixed[e * pw..][..pw],
            start,
            width,
            &mut fixed[e * w..][..w],
        );
    }
    gate.flush()?;
    let (firsts, part_of) = distinct(lim, s, width, w, &value, &fixed)?;
    let parts = Pieces::at(lim, w, &value, &fixed, &firsts)?;
    lim.discard(value);
    lim.discard(fixed);
    lim.discard(firsts);
    let (pieces, starts, cover) = refine(lim, width, parts)?;
    Ok(Side {
        part_of,
        pieces,
        starts,
        cover,
    })
}

/// Expand each distinct coarse list in `firsts` into the sorted, deduplicated
/// fine pieces its parts cover.
fn expand_coarse_lists(
    lim: &Limits,
    lists: &Lists,
    firsts: &[u32],
    fine: &Side,
) -> Result<Lists, OperationError> {
    let mut expanded = Lists::default();
    lim.reserve_exact(&mut expanded.starts, firsts.len() + 1)?;
    let total: usize = firsts
        .iter()
        .flat_map(|&k| lists.get(k as usize))
        .map(|&item| fine.cover((item >> 32) as u32).len())
        .sum();
    // The sets keep at most `total` items, so every start is at most it.
    lim.list_index(total)?;
    let mut gate = lim.gate();
    gate.poll(total as u64)?;
    lim.reserve_exact(&mut expanded.items, total)?;
    expanded.starts.push(0);
    for &k in firsts {
        let start = expanded.items.len();
        for &item in lists.get(k as usize) {
            let (part, atom) = ((item >> 32) as u32, item & 0xffff_ffff);
            expanded
                .items
                .extend(fine.cover(part).iter().map(|&f| (f as u64) << 32 | atom));
        }
        // A part's cover ascends, so a list of one part, or of parts whose
        // covers follow each other, is in order already.
        let set = &mut expanded.items[start..];
        if !set.is_sorted() {
            set.sort_unstable();
        }
        dedup_tail(&mut expanded.items, start);
        expanded.starts.push(expanded.items.len() as u32);
    }
    gate.flush()?;
    Ok(expanded)
}

/// Cut distinct cubes of `width` bits, ascending by value, into disjoint
/// pieces, each cube the union of the pieces it covers. Returns the pieces
/// (their atoms unset), and where each cube's pieces start in the list of
/// covered pieces, with that list.
fn refine(
    lim: &Limits,
    width: usize,
    parts: Pieces,
) -> Result<(Pieces, Vec<u32>, Vec<u32>), OperationError> {
    let (k, w) = (parts.value.len() / parts.words, parts.words);
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
    let mut stack = vec![Frame {
        items: (0, k),
        bit: width,
        value: 0,
    }];
    let mut out = Pieces::empty(w);
    // Each piece's cubes, as ranges of the arena: a cube free at a split
    // is listed on both sides, so the arena can outgrow a 32-bit offset.
    let mut members: Vec<(u32, u32)> = Vec::new();
    let span = |lim: &Limits, from: usize, end: usize| {
        lim.list_index(end).map(|end| (from as u32, end - from as u32))
    };
    let (mut or_f, mut and_f, mut or_v, mut and_v) =
        (vec![0u64; w], vec![0u64; w], vec![0u64; w], vec![0u64; w]);
    let mut split_bits = vec![0u64; w];
    let mut gate = lim.gate();
    while let Some(frame) = stack.pop() {
        let (from, len) = frame.items;
        gate.poll(len as u64)?;
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
            let below = lower_bits(j, frame.bit);
            and_f[j] &= below;
            or_f[j] &= below;
            split_bits[j] = ((or_f[j] & !and_f[j]) | (and_f[j] & (or_v[j] ^ and_v[j]))) & below;
        }
        let top = (0..w)
            .rev()
            .find(|&j| split_bits[j] != 0)
            .map(|j| 64 * j + 63 - split_bits[j].leading_zeros() as usize);
        // The bits above the split, or every bit left, are shared: fixed
        // where every cube fixes them, alike, and free where none does.
        let lowest = top.map_or(0, |s| s + 1);
        let path = frame.value;
        for j in 0..w {
            let keep = lower_bits(j, frame.bit) & !lower_bits(j, lowest);
            paths[path + w + j] |= and_f[j] & keep;
            paths[path + j] |= or_v[j] & and_f[j] & keep;
        }
        match top {
            None => {
                lim.reserve_exact(&mut out.value, w)?;
                lim.reserve_exact(&mut out.fixed, w)?;
                out.value.extend_from_slice(&paths[path..path + w]);
                out.fixed.extend_from_slice(&paths[path + w..path + 2 * w]);
                lim.try_push(&mut members, span(lim, from, from + len)?)?;
            }
            Some(s) => {
                let (j, b) = (s / 64, s % 64);
                // A cube free at the split goes to both sides.
                for side in [1u64, 0] {
                    let start = arena.len();
                    for i in from..from + len {
                        let p = arena[i] as usize;
                        let (f, v) = (
                            parts.fixed[p * w + j] >> b & 1,
                            parts.value[p * w + j] >> b & 1,
                        );
                        if f == 0 || v == side {
                            lim.try_push(&mut arena, p as u32)?;
                        }
                    }
                    let slot = paths.len();
                    lim.reserve(&mut paths, 2 * w)?;
                    paths.extend_from_within(path..path + 2 * w);
                    paths[slot + w + j] |= 1 << b;
                    paths[slot + j] |= side << b;
                    stack.push(Frame {
                        items: (start, arena.len() - start),
                        bit: s,
                        value: slot,
                    });
                }
            }
        }
    }
    gate.flush()?;

    let (starts, cover) = invert_piece_members(lim, k, &arena, &members)?;
    out.atom = filled(lim, members.len(), 0u32)?;
    lim.discard(arena);
    lim.discard(paths);
    lim.discard(members);
    lim.discard(parts);
    Ok((out, starts, cover))
}

/// Invert each piece's cube slice `(from, len)` in `arena` into each cube's
/// covered pieces `(starts, cover)`.
fn invert_piece_members(
    lim: &Limits,
    k: usize,
    arena: &[u32],
    members: &[(u32, u32)],
) -> Result<(Vec<u32>, Vec<u32>), OperationError> {
    // Every start is at most the cover's length, the members' total.
    lim.list_index(members.iter().map(|&(_, len)| len as usize).sum())?;
    let mut starts = filled(lim, k + 1, 0u32)?;
    for &(from, len) in members {
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
    lim.discard(at);
    Ok((starts, cover))
}

/// The bits of word `j` below bit `bit` of a multi-word value.
fn lower_bits(j: usize, bit: usize) -> u64 {
    let bits = bit.saturating_sub(64 * j).min(64);
    if bits == 64 { !0 } else { (1u64 << bits) - 1 }
}

#[cfg(test)]
#[path = "tests/cubes.rs"]
mod tests;
