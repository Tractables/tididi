//! Disjunction and difference in skip form, by an overlay of the operands
//! that never complements a level.
//!
//! As the stored overlay ([`crate::apply::overlay`]) does, each class of one
//! operand at a level is split by the classes of the other into pieces: the
//! product `a ∧ b`, the residue `a ∧ ¬G` outside every class of `g`, and for
//! a disjunction the residue `¬F ∧ b`. A cell of a class, a pair of child
//! pieces, lies in one class of the other operand or in none, which that
//! operand's pairs at the level answer, so a residue is collected from its
//! cells and never computed as a complement. The pieces of a level are
//! pairwise disjoint, and each class is the union of its own.
//!
//! In skip form a level's classes are its stored nodes and the references
//! that skip through it, each the one pair `true` on the skipped variable
//! makes ([`SkipForm::pairs_at`]). Where neither operand stores a node at a
//! level, every class there skips through it toward the same child, and so
//! does every piece: the pieces of that child are the level's. The walk
//! visits only the levels where an operand stores a node or where a skip
//! ends on a leaf label (two leaf children), from the leaves up.
//!
//! The walk is the larger operand's, `P` (the one with more pairs), and the
//! result keeps its numbering. Every reference of `P` carries its pieces as
//! of the highest level visited that read it, each with the class of the
//! other operand, `S`, it lies in. A class of `P` whose cells all lie in one
//! class of `S`, or all outside `S`, is one piece: the class itself, which
//! keeps its slot, so the parents that name it see no change. Where its
//! sides kept their references too the node is `P`'s own, and a level whose
//! slots are all `P`'s own is `P`'s level, shared. A class that splits gives
//! its slot to its first stored piece and appends the others after the
//! level's slots, so a change climbs the vtree only through classes that
//! split. At a level with one leaf child a skip of `P` is rebuilt only where
//! `S`'s stored pairs there name the class of one of its pieces, found from
//! `S`'s classes; elsewhere the skip's pieces are its child's.
//!
//! `S` keeps no products, which are `P`'s pieces; it keeps only its
//! residues, its parts outside every class of `P`, and only where the result
//! needs them (a disjunction, or the difference `S ∧ ¬P`). A residue is the
//! residues of its cells' sides, and the cells of the products of its sides
//! that no pair of `P` at the level holds; against a leaf child those are
//! the literals the pairs of `P` naming a class leave out, so only the
//! classes `P` names there partly add to the residues.
//!
//! Against a leaf child a cell's side is a set of literals: the other
//! operand's labels that pair with the cell's inner piece split it, and the
//! literals none of them holds go to the residue.
//!
//! The raw result is a skip form of the function whose classes at each level
//! are these pieces; its output is the union of the outputs' pieces (for a
//! difference, the part of `f`'s output outside `g`), one node at the level
//! where the two outputs' pieces were last split. The contraction then
//! merges equal pieces and makes the result canonical.

use std::cmp::Reverse;

use rustc_hash::{FxHashMap, FxHashSet};

use super::contract::contract_in;
use super::{cells_label, filled, ix, literal_cells, mk, normalize, vt, Ref, SLevel, Shape, SkipForm, NONE, POS, ZREF};
use crate::limits::Limits;
use crate::OperationError;

/// Which Boolean combination the overlay builds.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Combine {
    /// `f ∨ g`.
    Or,
    /// `f ∧ ¬g`.
    AndNot,
}

/// The class of a piece in an operand where it lies in none of its classes.
const ABSENT: Ref = ZREF;

/// An operand's references numbered densely: stored node `i` of level `t` is
/// `base[t] + i`, label `k` of leaf `c` is `base[c] + k`, for the levels in
/// `cone` (every level without one) and the levels and leaves their pairs
/// and the output name; the others have no numbers, so that a small operand
/// on a large vtree numbers little.
struct Ids {
    base: Vec<u32>,
    len: usize,
}

impl Ids {
    fn of(s: &SkipForm, cone: Option<&[bool]>, lim: &Limits) -> Result<Ids, OperationError> {
        let sh = &*s.shape;
        let mut base = filled(lim, sh.num_nodes(), NONE)?;
        let mut n = 0u32;
        for &t in &s.active {
            if inside(cone, t) {
                base[t as usize] = n;
                n += s.levels[t as usize].width() as u32;
            }
        }
        let mut named = |x: Ref| {
            let v = vt(x) as usize;
            if base[v] == NONE {
                base[v] = n;
                n += if sh.is_leaf(v as u32) { sh.labels[v] } else { s.levels[v].width() as u32 };
            }
        };
        named(s.out);
        for &t in &s.active {
            if inside(cone, t) {
                for &(a, b) in &s.levels[t as usize].pairs {
                    named(a);
                    named(b);
                }
            }
        }
        Ok(Ids { base, len: n as usize })
    }

    /// Whether `x` has a number: the operand names it or its level.
    #[inline]
    fn has(&self, x: Ref) -> bool {
        self.base[vt(x) as usize] != NONE
    }

    #[inline]
    fn id(&self, x: Ref) -> usize {
        debug_assert!(self.base[vt(x) as usize] != NONE, "an unnumbered reference");
        (self.base[vt(x) as usize] + ix(x)) as usize
    }
}

/// Per reference of `P`, what the walk reads of it together, in one line:
/// its pieces (each the result's reference and the class of `S` it lies in,
/// or [`ABSENT`]), the stamps of the last levels that named it and rebuilt
/// it, and the level a skip to it is read at.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct Rec {
    /// The one piece where `n == 1`; where `n > 1`, `one.0` is the start of
    /// the list in the arena ([`Big::put`]).
    one: (Ref, Ref),
    n: u32,
    /// `stamp << 2 | cells`: the stamp of the last level whose pairs name
    /// it, with the literal cells they give it there against a leaf child.
    named: u32,
    /// For a reference a pair names through a skip, the child level the pair
    /// reads it at: it is a class from there down to its own level, and no
    /// other level's pairs name it. `NONE` for the others.
    from: u32,
    /// The stamp of the last level where it skips through and its pieces
    /// are rebuilt.
    mark: u32,
}

const REC: Rec = Rec { one: (ZREF, ZREF), n: 0, named: 0, from: NONE, mark: 0 };

impl Rec {
    /// Its pieces, `arena` holding those of more than one.
    #[inline]
    fn pieces<'a>(&'a self, arena: &'a [(Ref, Ref)]) -> &'a [(Ref, Ref)] {
        if self.n <= 1 {
            &std::slice::from_ref(&self.one)[..self.n as usize]
        } else {
            &arena[self.one.0 as usize..self.one.0 as usize + self.n as usize]
        }
    }
}

/// Whether level `t` is in `cone`, every level where there is none.
#[inline]
fn inside(cone: Option<&[bool]>, t: u32) -> bool {
    cone.is_none_or(|c| c[t as usize])
}

/// Per level whose children are both leaves, the references to its leaves
/// that `s` reads there through a skip from above (from the output and the
/// pairs of the levels in `cone`); and, where `from` is given, per reference
/// such a pair names through a skip the child level it is read at.
fn read_through(s: &SkipForm, ids: &Ids, cone: Option<&[bool]>, mut from: Option<&mut [Rec]>, lim: &Limits) -> Result<FxHashMap<u32, Vec<Ref>>, OperationError> {
    let sh = &*s.shape;
    let mut skips: FxHashMap<u32, Vec<Ref>> = FxHashMap::default();
    // A reference read through a skip, at child level `c`. Every pair
    // reads it from the same level: where `from` is given, a leaf label is
    // listed the first time only.
    let mut note = |c: u32, x: Ref| -> Result<(), OperationError> {
        if let Some(from) = from.as_deref_mut() {
            let k = &mut from[ids.id(x)].from;
            debug_assert!(*k == NONE || *k == c, "a reference read from two levels");
            let first = *k == NONE;
            *k = c;
            if !first {
                return Ok(());
            }
        }
        if sh.is_leaf(vt(x)) {
            lim.try_push(skips.entry(sh.parent[vt(x) as usize]).or_default(), x)?;
        }
        Ok(())
    };
    if vt(s.out) != sh.root {
        note(sh.root, s.out)?;
    }
    for &t in &s.active {
        if !inside(cone, t) {
            continue;
        }
        let (l, r) = sh.kids[t as usize];
        for &(a, b) in &s.levels[t as usize].pairs {
            if vt(a) != l {
                note(l, a)?;
            }
            if vt(b) != r {
                note(r, b)?;
            }
        }
    }
    for v in skips.values_mut() {
        v.sort_unstable();
        v.dedup();
    }
    Ok(skips)
}

fn discard_skips(skips: FxHashMap<u32, Vec<Ref>>, lim: &Limits) {
    for (_, v) in skips {
        lim.discard(v);
    }
}

// Tests have [`Big::room`] and [`Small::note`] move the live lists to a new
// buffer at every growth, and count the moves; production moves them by the
// lists' sizes alone.
#[cfg(test)]
use super::tests::{compactions_forced, note_compaction};

#[cfg(not(test))]
#[inline(always)]
fn compactions_forced() -> bool {
    false
}

#[cfg(not(test))]
#[inline(always)]
fn note_compaction(_lists: usize) {}

/// The operand with more pairs, `P`, whose numbering the result keeps.
struct Big<'a> {
    s: &'a SkipForm,
    ids: Ids,
    /// Per reference, its record.
    rec: Vec<Rec>,
    /// The lists of more than one piece, and the room of lists replaced.
    arena: Vec<(Ref, Ref)>,
    /// The pieces of the lists the records name in `arena`.
    live: usize,
    skips: FxHashMap<u32, Vec<Ref>>,
    /// The level where the output's pieces were last written.
    out_at: u32,
}

impl<'a> Big<'a> {
    /// The walk's operand, numbered in `cone` (the levels the walk visits).
    fn new(s: &'a SkipForm, cone: Option<&[bool]>, lim: &Limits) -> Result<Big<'a>, OperationError> {
        let ids = Ids::of(s, cone, lim)?;
        let mut rec = filled(lim, ids.len, REC)?;
        let skips = read_through(s, &ids, cone, Some(&mut rec), lim)?;
        Ok(Big { s, ids, rec, arena: Vec::new(), live: 0, skips, out_at: NONE })
    }

    /// The pieces of `x` as of the highest level visited that reads it.
    #[inline]
    fn pieces(&self, x: Ref) -> &[(Ref, Ref)] {
        self.rec[self.ids.id(x)].pieces(&self.arena)
    }

    /// Whether `x`'s pieces are `x` itself, in one class of `S`: that class.
    #[inline]
    fn whole(&self, x: Ref) -> Option<Ref> {
        let r = &self.rec[self.ids.id(x)];
        (r.n == 1 && r.one.0 == x).then_some(r.one.1)
    }

    /// Make `items` the pieces of the reference numbered `id`. A list of
    /// more than one piece takes the room of the list it replaces where that
    /// is as long, or the arena's tail where the list it replaces ends it;
    /// otherwise it is appended. A walk sets a skip's pieces again at every
    /// level that rebuilds them, so lists appended without reuse grew the
    /// arena many times past the pieces it holds.
    #[inline]
    fn put(&mut self, id: usize, items: &[(Ref, Ref)], lim: &Limits) -> Result<(), OperationError> {
        let r = self.rec[id];
        let n = items.len();
        let mut at = None;
        if r.n > 1 {
            let (s, k) = (r.one.0 as usize, r.n as usize);
            self.live -= k;
            if s + k == self.arena.len() {
                self.arena.truncate(s);
            } else if n <= k {
                at = Some(s);
            }
        }
        match items {
            [] => {}
            &[x] => self.rec[id].one = x,
            _ => {
                let s = match at {
                    Some(s) => {
                        self.arena[s..s + n].copy_from_slice(items);
                        s
                    }
                    None => {
                        // The record names no list while the arena makes
                        // room, which may move the others.
                        self.rec[id].n = 0;
                        self.room(n, lim)?;
                        self.arena.extend_from_slice(items);
                        self.arena.len() - n
                    }
                };
                self.rec[id].one.0 = s as Ref;
                self.live += n;
            }
        }
        self.rec[id].n = n as u32;
        Ok(())
    }

    /// Room for `n` more pieces at the arena's end. Where at most half of a
    /// full arena is live, the live lists move to a new arena twice their
    /// size, so the arena stays within twice the pieces it holds; but only
    /// where the arena is at least half as long as the records, so that the
    /// pass over them costs little next to the pieces appended since the
    /// last.
    fn room(&mut self, n: usize, lim: &Limits) -> Result<(), OperationError> {
        let cap = self.arena.capacity();
        if cap - self.arena.len() >= n {
            return Ok(());
        }
        let need = self.live + n;
        let forced = compactions_forced();
        if !forced && (2 * need > cap || self.rec.len() > 2 * cap) {
            return lim.reserve(&mut self.arena, n);
        }
        note_compaction(0);
        let mut fresh = Vec::new();
        lim.reserve_exact(&mut fresh, 2 * need)?;
        for r in self.rec.iter_mut().filter(|r| r.n > 1) {
            let s = r.one.0 as usize;
            r.one.0 = fresh.len() as Ref;
            fresh.extend_from_slice(&self.arena[s..s + r.n as usize]);
        }
        lim.discard(std::mem::replace(&mut self.arena, fresh));
        Ok(())
    }

    /// Make `items` the pieces of `x` at level `t`, noting `x` under each
    /// class of `S` they lie in where `x` is a class above `t` too.
    fn set(&mut self, x: Ref, items: &[(Ref, Ref)], t: u32, small: &mut Small, lim: &Limits) -> Result<(), OperationError> {
        let id = self.ids.id(x);
        self.put(id, items, lim)?;
        if x == self.s.out {
            self.out_at = t;
        }
        let from = self.rec[id].from;
        if from != NONE && from != t {
            for &(_, y) in items {
                if y != ABSENT {
                    small.note(y, x, lim)?;
                }
            }
        }
        Ok(())
    }

    /// Note that the current level's pairs (stamped `stamp`) name `e` with
    /// the literal cells `m`, returning whether they had not named it yet.
    #[inline]
    fn name(&mut self, e: Ref, m: u8, stamp: u32) -> bool {
        debug_assert!(stamp < 1 << 30, "a stamp past the record's");
        let k = &mut self.rec[self.ids.id(e)].named;
        let first = *k >> 2 != stamp;
        if first {
            *k = stamp << 2;
        }
        *k |= m as u32;
        first
    }

    /// The literal cells the current level's pairs give `e` ([`Big::name`]).
    #[inline]
    fn cells(&self, e: Ref) -> u8 {
        (self.rec[self.ids.id(e)].named & 3) as u8
    }

    fn discard(self, lim: &Limits) {
        lim.discard(self.into_ids(lim).base);
    }

    /// The numbering, the rest dropped.
    fn into_ids(self, lim: &Limits) -> Ids {
        lim.discard(self.rec);
        lim.discard(self.arena);
        discard_skips(self.skips, lim);
        self.ids
    }
}

/// Per reference of `S`, what the walk reads of it together, in one line.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct SRec {
    /// The result's reference for its residue as of the highest level
    /// visited that reads it, or `ZREF` where it has none.
    res: Ref,
    /// The piece of the current pair's right side in it, or `ZREF`
    /// ([`inner`]'s cells by left side, where `P`'s pieces outside `S` are
    /// not kept).
    seen: Ref,
    /// Where the operand's stored pairs at the current level name it in
    /// [`index_inner`]'s index: the stamp of the level, start and count.
    run: (u32, u32, u32),
    /// The first entry of `inv` (`NONE` for none): the skips of `P` that had
    /// a piece in it when their pieces were set, each entry with the next.
    /// Read only where the reference's own pairs name it, and checked there
    /// against the skip's pieces; `WALKED` once read, at the one level whose
    /// pairs name it.
    head: u32,
}

const SREC: SRec = SRec { res: ZREF, seen: ZREF, run: (0, 0, 0), head: NONE };
/// The head of a class's list once [`lopsided`] has read it.
const WALKED: u32 = NONE - 1;

/// The other operand, `S`: its classes split `P`'s, and it keeps only its
/// residues, its parts outside every class of `P`.
struct Small<'a> {
    s: &'a SkipForm,
    ids: Ids,
    /// Per reference, its record.
    rec: Vec<SRec>,
    skips: FxHashMap<u32, Vec<Ref>>,
    inv: Vec<(Ref, u32)>,
    /// The entries of `inv` on lists already read.
    walked: usize,
    /// The level where the output's residue was last written.
    out_at: u32,
}

impl<'a> Small<'a> {
    /// The walk's other operand.
    fn new(s: &'a SkipForm, lim: &Limits) -> Result<Small<'a>, OperationError> {
        let ids = Ids::of(s, None, lim)?;
        let skips = read_through(s, &ids, None, None, lim)?;
        let rec = filled(lim, ids.len, SREC)?;
        Ok(Small { s, ids, rec, skips, inv: Vec::new(), walked: 0, out_at: NONE })
    }

    /// The entries of this operand's sorted index of the level stamped
    /// `stamp` ([`index_inner`], [`index_left`]) whose first side is `e`. A
    /// conjunction's `true` beyond the cone ([`conjoin`]) may be a class this
    /// operand never names: no pair names it.
    #[inline]
    fn named<'i, T>(&self, idx: &'i [T], e: Ref, stamp: u32) -> &'i [T] {
        if !self.ids.has(e) {
            return &[];
        }
        let (st, lo, n) = self.rec[self.ids.id(e)].run;
        if st == stamp { &idx[lo as usize..(lo + n) as usize] } else { &[] }
    }

    #[inline]
    fn res(&self, x: Ref) -> Ref {
        self.rec[self.ids.id(x)].res
    }

    /// Make `r` the residue of `x` at level `t`.
    fn set_res(&mut self, x: Ref, r: Ref, t: u32) {
        self.rec[self.ids.id(x)].res = r;
        if x == self.s.out {
            self.out_at = t;
        }
    }

    /// Note that a piece of `P`'s skip `x` lies in class `y`; nothing where
    /// this operand never names `y`, whose pairs never read it.
    fn note(&mut self, y: Ref, x: Ref, lim: &Limits) -> Result<(), OperationError> {
        if !self.ids.has(y) {
            return Ok(());
        }
        let forced = compactions_forced();
        // A compaction reads every record: only where they are no more than
        // twice the entries, as in [`Big::room`].
        let len = self.inv.len();
        if len == self.inv.capacity() && len > 0 && (forced || (2 * self.walked >= len && self.rec.len() <= 2 * len)) {
            self.compact(lim)?;
        }
        let h = &mut self.rec[self.ids.id(y)].head;
        let next = if *h == WALKED { NONE } else { *h };
        lim.try_push(&mut self.inv, (x, next))?;
        *h = (self.inv.len() - 1) as u32;
        Ok(())
    }

    /// Move the lists not yet read to a new `inv` with as much room again,
    /// each in its order.
    fn compact(&mut self, lim: &Limits) -> Result<(), OperationError> {
        let live = self.inv.len() - self.walked;
        let mut fresh = Vec::new();
        lim.reserve_exact(&mut fresh, 2 * live + 1)?;
        for r in &mut self.rec {
            let mut k = r.head;
            if k == NONE || k == WALKED {
                continue;
            }
            let a = fresh.len();
            while k != NONE {
                let (x, next) = self.inv[k as usize];
                fresh.push((x, NONE));
                k = next;
            }
            fresh[a..].reverse();
            for (i, e) in fresh.iter_mut().enumerate().skip(a + 1) {
                e.1 = (i - 1) as u32;
            }
            r.head = (fresh.len() - 1) as u32;
        }
        debug_assert_eq!(fresh.len(), live, "an entry on no list or on a list read");
        note_compaction(1);
        lim.discard(std::mem::replace(&mut self.inv, fresh));
        self.walked = 0;
        Ok(())
    }

    fn discard(self, lim: &Limits) {
        lim.discard(self.ids.base);
        lim.discard(self.rec);
        lim.discard(self.inv);
        discard_skips(self.skips, lim);
    }
}

/// A slot of `P` whose node stands.
const KEEP: u32 = NONE;
/// A slot of `P` whose class has no stored piece: an empty node until the
/// result is collected.
const DEAD: u32 = NONE - 1;

/// The result's level being built: `P`'s slots, each unchanged, rewritten or
/// dead, and the nodes appended after them.
#[derive(Default)]
struct Build {
    t: u32,
    /// `P`'s width at the level: appended node `k` is `width + k`.
    width: u32,
    /// Per slot of `P`, once one changes: [`KEEP`], [`DEAD`], or its node in
    /// `rew`.
    slot: Vec<u32>,
    rew: SLevel,
    app: SLevel,
}

impl Build {
    fn begin(&mut self, t: u32, width: u32) {
        self.t = t;
        self.width = width;
        self.slot.clear();
        for l in [&mut self.rew, &mut self.app] {
            l.start.clear();
            l.pairs.clear();
        }
    }

    fn mark(&mut self, j: u32, v: u32, lim: &Limits) -> Result<(), OperationError> {
        if self.slot.is_empty() {
            lim.try_resize(&mut self.slot, self.width as usize, KEEP)?;
        }
        debug_assert_eq!(self.slot[j as usize], KEEP, "a slot changed twice");
        self.slot[j as usize] = v;
        Ok(())
    }

    /// Slot `j` holds `pairs`, a piece of `P`'s class there, as its own.
    fn rewrite(&mut self, j: u32, pairs: &[(Ref, Ref)], lim: &Limits) -> Result<Ref, OperationError> {
        let k = self.rew.push(pairs, lim)?;
        self.mark(j, k, lim)?;
        Ok(mk(self.t, j))
    }

    fn kill(&mut self, j: u32, lim: &Limits) -> Result<(), OperationError> {
        self.mark(j, DEAD, lim)
    }

    fn append(&mut self, pairs: &[(Ref, Ref)], lim: &Limits) -> Result<Ref, OperationError> {
        let k = self.app.push(pairs, lim)?;
        Ok(mk(self.t, self.width + k))
    }

    /// The result's reference for a new piece at the level whose cells are
    /// `pairs`: the node they make, appended, or the reference it resolves
    /// to where it skips.
    fn piece(&mut self, sh: &Shape, pairs: &mut Vec<(Ref, Ref)>, lim: &Limits) -> Result<Ref, OperationError> {
        match normalize(sh, self.t, pairs) {
            Err(x) => Ok(x),
            Ok(()) => self.append(pairs, lim),
        }
    }

    /// The result's reference for a piece of `P`'s class at `slot` (`None`
    /// for a skip): the class's slot for its first stored piece (`taken`
    /// notes it), the others appended, or where a piece skips, the reference
    /// it resolves to.
    fn place(&mut self, sh: &Shape, slot: Option<u32>, taken: &mut bool, pairs: &mut Vec<(Ref, Ref)>, lim: &Limits) -> Result<Ref, OperationError> {
        match normalize(sh, self.t, pairs) {
            Err(x) => Ok(x),
            Ok(()) => match slot {
                Some(j) if !*taken => {
                    *taken = true;
                    self.rewrite(j, pairs, lim)
                }
                _ => self.append(pairs, lim),
            },
        }
    }

    /// Install the level in `out`: `P`'s own, shared, where no slot changed
    /// and nothing was appended; else built from `P`'s nodes, the rewritten
    /// ones and the appended ones. Returns whether a slot died.
    fn finish(&mut self, out: &mut SkipForm, p: &SkipForm, lim: &Limits) -> Result<bool, OperationError> {
        let t = self.t as usize;
        let added = self.app.width();
        let src = &p.levels[t];
        if self.slot.is_empty() && added == 0 {
            if self.width > 0 {
                out.levels.share(t, &p.levels, lim)?;
                out.active.push(self.t);
                super::trace::count(0, src.pairs.len() as u64);
                super::trace::count(2, 1);
            }
            return Ok(false);
        }
        if added == 0 && self.slot.iter().all(|&s| s == DEAD) {
            // Every slot of `P` died and nothing was added: the level holds
            // no node, and no pair names a dead slot, so none is built.
            super::trace::count(1, self.width as u64);
            return Ok(true);
        }
        let node = |j: usize| -> &[(Ref, Ref)] {
            match self.slot.get(j).copied().unwrap_or(KEEP) {
                KEEP => src.node(j as u32),
                DEAD => &[],
                k => self.rew.node(k),
            }
        };
        let w = self.width as usize;
        let total = (0..w).map(|j| node(j).len()).sum::<usize>() + self.app.pairs.len();
        let mut lvl = SLevel::default();
        lim.reserve_exact(&mut lvl.start, w + added + 1)?;
        lim.reserve_exact(&mut lvl.pairs, total)?;
        lvl.start.push(0);
        let mut kept = 0;
        for j in 0..w {
            lvl.pairs.extend_from_slice(node(j));
            lvl.start.push(lvl.pairs.len() as u32);
            if self.slot.get(j).is_none_or(|&s| s == KEEP) {
                kept += src.node(j as u32).len();
            }
        }
        if added > 0 {
            let base = lvl.pairs.len() as u32;
            lvl.pairs.extend_from_slice(&self.app.pairs);
            lvl.start.extend(self.app.start[1..].iter().map(|&s| s + base));
        }
        super::trace::count(0, kept as u64);
        let dead = self.slot.iter().filter(|&&s| s == DEAD).count();
        super::trace::count(1, dead as u64);
        super::trace::count(3, self.rew.width() as u64);
        out.levels.set(t, lvl);
        out.active.push(self.t);
        Ok(dead > 0)
    }

    fn discard(self, lim: &Limits) {
        lim.discard(self.slot);
        for l in [self.rew, self.app] {
            lim.discard(l.start);
            lim.discard(l.pairs);
        }
    }
}

/// Buffers the overlay reuses from level to level.
#[derive(Default)]
struct Scratch {
    /// Bumped per level that stamps references.
    stamp: u32,
    /// `S`'s stored pairs at a level with one leaf child by their inner side:
    /// `(inner reference, literal cells, node)`, sorted.
    sidx: Vec<(Ref, u8, u32)>,
    /// `S`'s stored pairs at a level with two internal children by their
    /// left side: `(left, right, node)`, sorted.
    lidx: Vec<(Ref, Ref, u32)>,
    /// The skips of `P` through the level whose pieces are rebuilt there.
    touched: Vec<Ref>,
    /// The inner sides `P`'s pairs at the level name.
    names: Vec<Ref>,
    /// One class's cells as `(class of S, pair)`.
    tagged: Vec<(Ref, Ref, Ref)>,
    /// A level's pieces of `P`'s classes at each child by the class of `S`
    /// they lie in: `(class of S, piece, class of P)`, sorted.
    left: Vec<(Ref, Ref, Ref)>,
    right: Vec<(Ref, Ref, Ref)>,
    /// Cells of `S`'s residues against a leaf child by the class they go to.
    adds: Vec<(Ref, (Ref, Ref))>,
    pairs: Vec<(Ref, Ref)>,
    items: Vec<(Ref, Ref)>,
    /// Each operand's classes at a level with two leaf children, with their
    /// cells ([`grid`]).
    pcls: Vec<(Ref, u8)>,
    scls: Vec<(Ref, u8)>,
    owner: FxHashMap<(Ref, Ref), u32>,
    held: FxHashSet<(Ref, Ref)>,
}

impl Scratch {
    fn discard(self, lim: &Limits) {
        lim.discard(self.sidx);
        lim.discard(self.lidx);
        lim.discard(self.touched);
        lim.discard(self.names);
        lim.discard(self.tagged);
        lim.discard(self.left);
        lim.discard(self.right);
        lim.discard(self.adds);
        lim.discard(self.pairs);
        lim.discard(self.items);
        lim.discard(self.pcls);
        lim.discard(self.scls);
        lim.discard(self.owner);
        lim.discard(self.held);
    }
}

/// The walk's state.
struct Walk<'a> {
    sh: &'a Shape,
    big: Big<'a>,
    small: Small<'a>,
    /// Whether `P`'s pieces outside every class of `S` are kept (not for
    /// `S ∧ ¬P`, which never reads them).
    rest: bool,
    /// Whether `S`'s residues are built (not for `P ∧ ¬S`).
    residues: bool,
    build: Build,
    sc: Scratch,
    lim: &'a Limits,
}

/// `f ∨ g` or `f ∧ ¬g` for two canonical skip forms on one shape, nothing
/// summed out: canonical.
pub(crate) fn overlay(op: Combine, f: &SkipForm, g: &SkipForm, lim: &Limits) -> Result<SkipForm, OperationError> {
    let sh = &*f.shape;
    match (f.is_zero(), g.is_zero(), op) {
        (true, _, Combine::Or) => return Ok(g.clone()),
        (true, _, Combine::AndNot) => return Ok(SkipForm::zero(f.shape.clone())),
        (false, true, _) => return Ok(f.clone()),
        _ => {}
    }
    if sh.is_leaf(sh.root) {
        // One variable: the outputs are literal cells.
        let (a, b) = (literal_cells(ix(f.out)), literal_cells(ix(g.out)));
        let m = match op {
            Combine::Or => a | b,
            Combine::AndNot => a & !b & 3,
        };
        let mut out = SkipForm::zero(f.shape.clone());
        out.out = if m == 0 { ZREF } else { mk(sh.root, cells_label(m)) };
        return Ok(out);
    }
    let raw = overlay_raw(op, f, g, lim)?;
    if !super::trace::on() {
        return contract_in(raw, None, lim);
    }
    traced(op, f, g, raw, lim)
}

/// The contraction after an overlay under `TIDIDI_SKIP_TRACE` (experiment):
/// one line per call on standard error with the operands' sizes, the pairs
/// of the operand with more active levels inside the other's cone, the raw
/// result's pairs kept from the larger operand, its dead slots, shared
/// levels and rewritten slots, and the time of each phase (0 to 3 the raw
/// overlay, 7 the contraction).
fn traced(op: Combine, f: &SkipForm, g: &SkipForm, raw: SkipForm, lim: &Limits) -> Result<SkipForm, OperationError> {
    let sh = &*f.shape;
    let pairs = |s: &SkipForm, cone: Option<&[bool]>| -> usize {
        s.active.iter().filter(|&&t| cone.is_none_or(|c| c[t as usize])).map(|&t| s.levels[t as usize].pairs.len()).sum()
    };
    let (big, small) = if f.active.len() < g.active.len() { (g, f) } else { (f, g) };
    let cone = super::conjoin::cone_of(sh, small, lim)?;
    let (fp, gp, bc, rp) = (pairs(f, None), pairs(g, None), pairs(big, Some(&cone)), pairs(&raw, None));
    let lv = cone.iter().filter(|&&c| c).count();
    lim.discard(cone);
    let mut clock = super::trace::Clock::start();
    let res = contract_in(raw, None, lim);
    clock.lap(7);
    let ph = super::trace::take();
    let cn = super::trace::take_counts();
    if let Ok(r) = &res {
        let (same, _) = copied(r, big);
        eprintln!(
            "SKIPTRACE {} f {fp} g {gp} big {} bigcone {bc} raw {rp} out {} copied {same} conelv {lv} kept {} dead {} shared {} rewritten {} us {}",
            if op == Combine::Or { "or" } else { "andnot" },
            if std::ptr::eq(big, f) { "f" } else { "g" },
            pairs(r, None),
            cn[0],
            cn[1],
            cn[2],
            cn[3],
            ph.iter().map(|n| (n / 1000).to_string()).collect::<Vec<_>>().join(" ")
        );
    }
    res
}

/// Experiment (`TIDIDI_SKIP_TRACE`): the pairs of `r` in nodes equal to
/// nodes of `big`, their sides equal too, and all of `r`'s pairs.
pub(super) fn copied(r: &SkipForm, big: &SkipForm) -> (usize, usize) {
    let sh = &*r.shape;
    let mut lv: Vec<u32> = r.active.clone();
    lv.sort_unstable_by_key(|&t| Reverse(sh.pre[t as usize]));
    let mut copy: FxHashMap<Ref, Ref> = FxHashMap::default();
    let map = |copy: &FxHashMap<Ref, Ref>, x: Ref| if sh.is_leaf(vt(x)) { Some(x) } else { copy.get(&x).copied() };
    let (mut same, mut all) = (0, 0);
    let mut buf = Vec::new();
    for &t in &lv {
        let bl = &big.levels[t as usize];
        let mut index: FxHashMap<Vec<(Ref, Ref)>, u32> = FxHashMap::default();
        for j in 0..bl.width() {
            let mut v = bl.node(j as u32).to_vec();
            v.sort_unstable();
            index.insert(v, j as u32);
        }
        let rl = &r.levels[t as usize];
        for i in 0..rl.width() {
            let node = rl.node(i as u32);
            all += node.len();
            buf.clear();
            let mapped = node.iter().all(|&(a, b)| match (map(&copy, a), map(&copy, b)) {
                (Some(a), Some(b)) => {
                    buf.push((a, b));
                    true
                }
                _ => false,
            });
            if !mapped {
                continue;
            }
            buf.sort_unstable();
            if let Some(&j) = index.get(&buf) {
                copy.insert(mk(t, i as u32), mk(t, j));
                same += node.len();
            }
        }
    }
    (same, all)
}

/// The pairs a skip form holds.
fn pair_count(s: &SkipForm) -> usize {
    s.active.iter().map(|&t| s.levels[t as usize].pairs.len()).sum()
}

/// The overlay of two nonzero canonical skip forms on one shape whose root
/// is internal, pruned and not contracted. It keeps the numbering of the
/// operand with more pairs, sharing that operand's levels where they stand.
pub(crate) fn overlay_raw(op: Combine, f: &SkipForm, g: &SkipForm, lim: &Limits) -> Result<SkipForm, OperationError> {
    debug_assert_eq!(f.shape.kids, g.shape.kids);
    debug_assert!(!f.is_zero() && !g.is_zero() && !f.shape.is_leaf(f.shape.root));
    let shape = f.shape.clone();
    let sh = &*shape;
    let mut clock = super::trace::Clock::start();
    let big_is_f = pair_count(f) >= pair_count(g);
    let (p, s) = if big_is_f { (f, g) } else { (g, f) };
    let rest = op == Combine::Or || big_is_f;
    let mut w = Walk {
        sh,
        big: Big::new(p, None, lim)?,
        small: Small::new(s, lim)?,
        rest,
        residues: op == Combine::Or || !big_is_f,
        build: Build::default(),
        sc: Scratch::default(),
        lim,
    };
    let levels = w.levels(None)?;
    let mut out = SkipForm::zero(shape.clone());
    let mut dead: Vec<u32> = Vec::new();
    clock.lap(0);
    w.run(&levels, &mut out, &mut dead)?;
    lim.discard(levels);
    clock.lap(1);
    out.out = match (op, big_is_f) {
        (Combine::AndNot, true) => w.big.pieces(p.out).iter().find(|x| x.1 == ABSENT).map_or(ZREF, |x| x.0),
        (Combine::AndNot, false) => w.small.res(s.out),
        (Combine::Or, _) => {
            let mut tops: Vec<Ref> = Vec::new();
            let rest = w.small.res(s.out);
            lim.reserve(&mut tops, w.big.pieces(p.out).len() + 1)?;
            tops.extend(w.big.pieces(p.out).iter().map(|x| x.0));
            if rest != ZREF {
                tops.push(rest);
            }
            let top = match tops[..] {
                [] => ZREF,
                [x] => x,
                _ => {
                    // Every piece is read at the higher of the two levels
                    // where the outputs were last split: their union is one
                    // node there.
                    let mut u = w.big.out_at;
                    if rest != ZREF {
                        let b = w.small.out_at;
                        debug_assert!(u != NONE && b != NONE && (sh.within(u, b) || sh.within(b, u)));
                        if sh.within(u, b) {
                            u = b;
                        }
                    }
                    let pairs = &mut w.sc.pairs;
                    pairs.clear();
                    for &x in &tops {
                        let mut buf = [(ZREF, ZREF)];
                        let ps = out.pairs_at(u, x, &mut buf);
                        lim.reserve(pairs, ps.len())?;
                        pairs.extend_from_slice(ps);
                    }
                    // The union takes the pairs of the tops stored at `u`,
                    // which nothing names then.
                    lim.try_push(&mut dead, u)?;
                    match normalize(sh, u, pairs) {
                        Err(x) => x,
                        Ok(()) => push_after(&mut out, u, pairs, lim)?,
                    }
                }
            };
            lim.discard(tops);
            top
        }
    };
    w.big.discard(lim);
    w.small.discard(lim);
    w.sc.discard(lim);
    w.build.discard(lim);
    clock.lap(2);
    // A disjunction reaches every piece and residue but the tops its union
    // took in, so only dead slots and those tops go, and the levels whose
    // nodes name them or a node they renumber: those above. A difference
    // drops the products of its output, and their parts below may be unread
    // at any level.
    let cand = match op {
        Combine::Or => {
            let mut up = filled(lim, sh.num_nodes(), false)?;
            for &t in &dead {
                let mut u = t;
                while u != NONE && !up[u as usize] {
                    up[u as usize] = true;
                    u = sh.parent[u as usize];
                }
            }
            let mut cand = Vec::new();
            if !dead.is_empty() {
                lim.reserve(&mut cand, out.active.len())?;
                cand.extend(out.active.iter().copied().filter(|&t| up[t as usize]));
            }
            lim.discard(up);
            cand
        }
        Combine::AndNot => out.active.clone(),
    };
    lim.discard(dead);
    out.collect_in(cand, lim)?;
    clock.lap(3);
    Ok(out)
}

impl Walk<'_> {
    /// The levels where an operand stores a node or a skip ends on a leaf
    /// label, those in `cone` where there is one, children first.
    fn levels(&self, cone: Option<&[bool]>) -> Result<Vec<u32>, OperationError> {
        let (p, s) = (self.big.s, self.small.s);
        let mut levels: Vec<u32> = Vec::new();
        self.lim.reserve(&mut levels, p.active.len() + s.active.len() + self.big.skips.len() + self.small.skips.len())?;
        levels.extend(p.active.iter().chain(&s.active).chain(self.big.skips.keys()).chain(self.small.skips.keys()).copied().filter(|&t| inside(cone, t)));
        levels.sort_unstable_by_key(|&t| Reverse(self.sh.pre[t as usize]));
        levels.dedup();
        Ok(levels)
    }

    /// Build `levels`, children first, into `out`, listing in `dead` those
    /// where a slot of `P` died.
    fn run(&mut self, levels: &[u32], out: &mut SkipForm, dead: &mut Vec<u32>) -> Result<(), OperationError> {
        let (sh, p, lim) = (self.sh, self.big.s, self.lim);
        for &t in levels {
            self.build.begin(t, p.levels[t as usize].width() as u32);
            let (l, r) = sh.kids[t as usize];
            match (sh.is_leaf(l), sh.is_leaf(r)) {
                (false, false) => inner(self, t)?,
                (true, true) => leaves(self, t)?,
                _ => lopsided(self, t)?,
            }
            if self.build.finish(out, p, lim)? {
                lim.try_push(dead, t)?;
            }
        }
        lim.check_stop()
    }

    /// Where `x`, a reference of `P` read at level `c`, is a class at no
    /// level of `cone` (it or the leaf's parent it skips to is beyond it), its
    /// pieces: itself in `S`'s `true` over its level, none where `S` reads
    /// nothing there; noted under that class where it is read at a level of
    /// the cone above its own. A leaf label read at its own leaf is a
    /// literal cell, not a class. `x` is marked once, 1 for a node, listed in
    /// `boundary`, and 3 for a leaf label.
    #[inline]
    fn seed(&mut self, x: Ref, c: u32, cone: &[bool], mark: &mut [u8], boundary: &mut Vec<Ref>) -> Result<(), OperationError> {
        let sh = self.sh;
        let v = vt(x);
        let leaf = sh.is_leaf(v);
        if leaf && v == c {
            return Ok(());
        }
        let q = if leaf { sh.parent[v as usize] } else { v };
        if cone[q as usize] {
            return Ok(());
        }
        self.seed_beyond(x, q, leaf, cone, mark, boundary)
    }

    /// [`Walk::seed`] for `x` beyond the cone, its level or its leaf's
    /// parent `q`.
    #[inline(never)]
    fn seed_beyond(&mut self, x: Ref, q: u32, leaf: bool, cone: &[bool], mark: &mut [u8], boundary: &mut Vec<Ref>) -> Result<(), OperationError> {
        let (sh, lim) = (self.sh, self.lim);
        let id = self.big.ids.id(x);
        if mark[id] != 0 {
            return Ok(());
        }
        if leaf {
            mark[id] = 3;
        } else {
            mark[id] = 1;
            lim.try_push(boundary, x)?;
        }
        // Beyond the cone `S` reads only its `true`, the same skip over every
        // level on `x`'s way down from `c`.
        let top = sh.one[q as usize];
        if top == ZREF {
            return Ok(());
        }
        self.big.put(id, &[(x, top)], lim)?;
        let from = self.big.rec[id].from;
        if from != NONE && cone[from as usize] {
            self.small.note(top, x, lim)?;
        }
        Ok(())
    }
}

/// `f ∧ g` for two canonical skip forms on one shape whose root is internal,
/// nothing summed out, where the product of the outputs is built at a level
/// ([`super::conjoin::land_pair`]): pruned, not contracted, with the
/// upward-closed cone of the levels where it may not be canonical, the
/// smaller operand's cone and the levels beyond it that lost a node.
///
/// The overlay's walk without `S`'s residues and the pieces outside `S`: the
/// piece of `y` in `S`'s class `s` is `y ∧ s`, and where `y` implies `s` it
/// is `y`, which keeps its slot. `P` is the operand with more active levels
/// and `S`'s cone bounds the walk: beyond it `S` is `true`, `P`'s references
/// there are their own products, and `P`'s levels there are the result's,
/// shared where it reaches them. The output is the piece of `P`'s in `S`'s.
pub(crate) fn conjoin(f: &SkipForm, g: &SkipForm, lim: &Limits) -> Result<(SkipForm, Vec<bool>), OperationError> {
    debug_assert_eq!(f.shape.kids, g.shape.kids);
    debug_assert!(!f.is_zero() && !g.is_zero() && !f.shape.is_leaf(f.shape.root));
    let shape = f.shape.clone();
    let sh = &*shape;
    let mut clock = super::trace::Clock::start();
    let (p, s) = if g.active.len() > f.active.len() { (g, f) } else { (f, g) };
    let mut out = SkipForm::zero(shape.clone());
    let mut cone = super::conjoin::cone_of(sh, s, lim)?;
    if super::conjoin::is_one(sh, s.out) {
        for &t in &p.active {
            out.levels.share(t as usize, &p.levels, lim)?;
        }
        out.active.clone_from(&p.active);
        out.out = p.out;
        return Ok((out, cone));
    }
    let mut w = Walk {
        sh,
        big: Big::new(p, Some(&cone), lim)?,
        small: Small::new(s, lim)?,
        rest: false,
        residues: false,
        build: Build::default(),
        sc: Scratch::default(),
        lim,
    };
    clock.lap(0);
    let mut mark = filled(lim, w.big.ids.len, 0u8)?;
    let mut boundary: Vec<Ref> = Vec::new();
    w.seed(p.out, sh.root, &cone, &mut mark, &mut boundary)?;
    for &u in &p.active {
        if cone[u as usize] {
            let (l, r) = sh.kids[u as usize];
            for &(a, b) in &p.levels[u as usize].pairs {
                w.seed(a, l, &cone, &mut mark, &mut boundary)?;
                w.seed(b, r, &cone, &mut mark, &mut boundary)?;
            }
        }
    }
    clock.lap(1);
    let levels = w.levels(Some(&cone))?;
    let mut dead: Vec<u32> = Vec::new();
    clock.lap(2);
    w.run(&levels, &mut out, &mut dead)?;
    lim.discard(levels);
    lim.discard(dead);
    out.out = w.big.pieces(p.out).iter().find(|x| x.1 == s.out).map_or(ZREF, |x| x.0);
    let Walk { big, small, sc, build, .. } = w;
    let ids = big.into_ids(lim);
    small.discard(lim);
    sc.discard(lim);
    build.discard(lim);
    clock.lap(3);
    // A conjunction drops cells and keeps one piece of the output: unread
    // nodes may be at any level of the cone.
    let cand = out.active.clone();
    out.collect_in(cand, lim)?;
    clock.lap(4);
    if out.out != ZREF {
        super::conjoin::share_outside(sh, p, &mut out, &mut cone, &boundary, &mut mark, |x| ids.id(x), &mut clock, lim)?;
    }
    lim.discard(ids.base);
    lim.discard(boundary);
    lim.discard(mark);
    clock.lap(5);
    Ok((out, cone))
}

/// Append a node at level `u` of the finished result, which may share the
/// level with `P`: a shared level is copied first, its charge moved to the
/// copy.
fn push_after(out: &mut SkipForm, u: u32, pairs: &[(Ref, Ref)], lim: &Limits) -> Result<Ref, OperationError> {
    let before = out.levels[u as usize].bytes();
    let mut lvl = out.levels.take(u as usize);
    lim.release_bytes(before);
    lim.charge_bytes(lvl.bytes())?;
    if lvl.width() == 0 {
        out.active.push(u);
    }
    let i = lvl.push(pairs, lim)?;
    out.levels.set(u as usize, lvl);
    Ok(mk(u, i))
}

/// The pieces of `P`'s class at `slot` (`None` for a skip) from its tagged
/// cells: a piece per class of `S` the cells lie in, the rest last where
/// `rest` keeps it. `items` gets each piece with that class, [`ABSENT`] for
/// the rest.
#[allow(clippy::too_many_arguments)]
fn emit(
    sh: &Shape,
    b: &mut Build,
    slot: Option<u32>,
    rest: bool,
    tagged: &mut [(Ref, Ref, Ref)],
    pairs: &mut Vec<(Ref, Ref)>,
    items: &mut Vec<(Ref, Ref)>,
    lim: &Limits,
) -> Result<(), OperationError> {
    items.clear();
    // By class, then by the side that is not a leaf, which `normalize`
    // orders a class's pairs by before it merges them: it then finds them in
    // order and does not sort them again.
    if sh.is_leaf(sh.kids[b.t as usize].0) {
        tagged.sort_unstable_by_key(|x| (x.0, x.2));
    } else {
        tagged.sort_unstable();
    }
    let mut taken = false;
    for run in tagged.chunk_by(|a, b| a.0 == b.0) {
        if run[0].0 == ABSENT && !rest {
            continue;
        }
        pairs.clear();
        lim.reserve(pairs, run.len())?;
        pairs.extend(run.iter().map(|&(_, a, b)| (a, b)));
        let o = b.place(sh, slot, &mut taken, pairs, lim)?;
        lim.try_push(items, (o, run[0].0))?;
    }
    if let Some(j) = slot
        && !taken
    {
        b.kill(j, lim)?;
    }
    Ok(())
}

/// `P`'s class `y` at slot `j` stands, one piece in `S`'s class `tag`: its
/// node is `P`'s own, or where `tag` is [`ABSENT`] and the rest is not kept,
/// it has no piece.
fn stand(w: &mut Walk, y: Ref, j: u32, tag: Ref, t: u32) -> Result<(), OperationError> {
    if tag == ABSENT && !w.rest {
        w.build.kill(j, w.lim)?;
        return w.big.set(y, &[], t, &mut w.small, w.lim);
    }
    w.big.set(y, &[(y, tag)], t, &mut w.small, w.lim)
}

/// A level whose children are both internal: no reference skips through it,
/// so each operand's classes are its stored nodes there, and a cell's class
/// in `S` is the node holding the pair of its sides' classes.
fn inner(w: &mut Walk, t: u32) -> Result<(), OperationError> {
    let lim = w.lim;
    let (pl, sl) = (&w.big.s.levels[t as usize], &w.small.s.levels[t as usize]);
    w.sc.owner.clear();
    lim.reserve_map(&mut w.sc.owner, sl.pairs.len())?;
    for j in 0..sl.width() {
        for &p in sl.node(j as u32) {
            w.sc.owner.insert(p, j as u32);
        }
    }
    let owner = std::mem::take(&mut w.sc.owner);
    let cell = |ya: Ref, yb: Ref| if ya == ABSENT || yb == ABSENT { ABSENT } else { owner.get(&(ya, yb)).map_or(ABSENT, |&j| mk(t, j)) };
    // The stamp of `S`'s pairs here by left side, once indexed.
    let mut lstamp = 0;
    let mut gate = lim.gate();
    for j in 0..pl.width() as u32 {
        let node = pl.node(j);
        let y = mk(t, j);
        // The node stands where every side is itself, in one class of `S`,
        // and every cell lies in one class of `S`.
        let mut tag = None;
        for &(a, b) in node {
            match (w.big.whole(a), w.big.whole(b)) {
                (Some(ya), Some(yb)) if tag.is_none_or(|x| x == cell(ya, yb)) => tag = Some(cell(ya, yb)),
                _ => {
                    tag = None;
                    break;
                }
            }
        }
        gate.poll(node.len() as u64 + 1)?;
        if let Some(tag) = tag {
            stand(w, y, j, tag, t)?;
            continue;
        }
        w.sc.tagged.clear();
        for &(a, b) in node {
            let (pa, pb) = (w.big.pieces(a), w.big.pieces(b));
            // Without the rest only the cells `S`'s pairs hold are kept: from
            // the pairs naming a left piece's class where they are fewer than
            // the cross of the pieces.
            if !w.rest && pa.len() * pb.len() > 1 {
                if lstamp == 0 {
                    w.sc.stamp += 1;
                    lstamp = w.sc.stamp;
                    index_left(&mut w.small, t, &mut w.sc.lidx, lstamp, lim)?;
                }
                let held: usize = pa.iter().map(|&(_, ya)| w.small.named(&w.sc.lidx, ya, lstamp).len()).sum();
                if held + pb.len() < pa.len() * pb.len() {
                    for &(q, yb) in pb {
                        if w.small.ids.has(yb) {
                            let k = w.small.ids.id(yb);
                            w.small.rec[k].seen = q;
                        }
                    }
                    for &(p, ya) in pa {
                        for &(_, yb, j) in w.small.named(&w.sc.lidx, ya, lstamp) {
                            let q = w.small.rec[w.small.ids.id(yb)].seen;
                            if q != ZREF {
                                lim.try_push(&mut w.sc.tagged, (mk(t, j), p, q))?;
                            }
                        }
                    }
                    for &(_, yb) in pb {
                        if w.small.ids.has(yb) {
                            let k = w.small.ids.id(yb);
                            w.small.rec[k].seen = ZREF;
                        }
                    }
                    gate.poll((held + 2 * pb.len()) as u64)?;
                    continue;
                }
            }
            for &(p, ya) in pa {
                for &(q, yb) in pb {
                    let c = cell(ya, yb);
                    if c != ABSENT || w.rest {
                        lim.try_push(&mut w.sc.tagged, (c, p, q))?;
                    }
                }
            }
            gate.poll((pa.len() * pb.len()) as u64)?;
        }
        emit(w.sh, &mut w.build, Some(j), w.rest, &mut w.sc.tagged, &mut w.sc.pairs, &mut w.sc.items, lim)?;
        w.big.set(y, &w.sc.items, t, &mut w.small, lim)?;
    }
    w.sc.owner = owner;
    if !w.residues {
        return gate.flush();
    }
    // `S`'s residues: the cells of its pairs no pair of `P` holds, from the
    // pieces of `P`'s classes at each child by the class of `S` they lie
    // in, and the residues of `S`'s classes there.
    w.sc.stamp += 1;
    let stamp = w.sc.stamp;
    let (held, left, right) = (&mut w.sc.held, &mut w.sc.left, &mut w.sc.right);
    held.clear();
    left.clear();
    right.clear();
    lim.reserve_set(held, pl.pairs.len())?;
    for &(a, b) in &pl.pairs {
        held.insert((a, b));
        for (x, list) in [(a, &mut *left), (b, &mut *right)] {
            if w.big.name(x, 0, stamp) {
                for &(p, z) in w.big.pieces(x) {
                    if z != ABSENT {
                        lim.try_push(list, (z, p, x))?;
                    }
                }
            }
        }
    }
    gate.poll(pl.pairs.len() as u64)?;
    left.sort_unstable();
    right.sort_unstable();
    let side = |list: &[(Ref, Ref, Ref)], z: Ref| -> std::ops::Range<usize> {
        let lo = list.partition_point(|e| e.0 < z);
        lo..lo + list[lo..].partition_point(|e| e.0 == z)
    };
    for j in 0..sl.width() as u32 {
        w.sc.pairs.clear();
        for &(za, zb) in sl.node(j) {
            let (ra, rb) = (w.small.res(za), w.small.res(zb));
            let (la, lb) = (side(left, za), side(right, zb));
            let a_cells = left[la].iter().map(|e| (e.1, e.2)).chain((ra != ZREF).then_some((ra, ABSENT)));
            for (p, ya) in a_cells {
                let b_cells = right[lb.clone()].iter().map(|e| (e.1, e.2)).chain((rb != ZREF).then_some((rb, ABSENT)));
                for (q, yb) in b_cells {
                    if ya == ABSENT || yb == ABSENT || !held.contains(&(ya, yb)) {
                        lim.try_push(&mut w.sc.pairs, (p, q))?;
                    }
                }
            }
            gate.poll(w.sc.pairs.len() as u64 + 1)?;
        }
        let r = if w.sc.pairs.is_empty() { ZREF } else { w.build.piece(w.sh, &mut w.sc.pairs, lim)? };
        w.small.set_res(mk(t, j), r, t);
    }
    gate.flush()
}

/// `S`'s stored pairs at level `t`, whose leaf child is the left one where
/// `leaf_left`, by their inner side, grouped ([`index_by`]) as
/// `(inner reference, literal cells, node)`; each inner reference's run
/// noted in its record under `stamp`.
fn index_inner(side: &mut Small, t: u32, leaf_left: bool, idx: &mut Vec<(Ref, u8, u32)>, stamp: u32, lim: &Limits) -> Result<(), OperationError> {
    let entry = |(a, b): (Ref, Ref), i: u32| {
        let (lab, e) = if leaf_left { (a, b) } else { (b, a) };
        (e, literal_cells(ix(lab)), i)
    };
    index_by(side, t, idx, entry, |x| x.0, stamp, lim)
}

/// `S`'s stored pairs at level `t`, whose children are both internal, by
/// their left side: `(left, right, node)`, grouped ([`index_by`]); each left
/// side's run noted in its record under `stamp`.
fn index_left(side: &mut Small, t: u32, idx: &mut Vec<(Ref, Ref, u32)>, stamp: u32, lim: &Limits) -> Result<(), OperationError> {
    index_by(side, t, idx, |(a, b), i| (a, b, i), |x| x.0, stamp, lim)
}

/// `S`'s stored pairs at level `t` as `entry(pair, node)` in `idx`, grouped
/// by `first` by counting, no comparison sort: the groups in the order of
/// their first pair, each in the level's order, and each group's run noted
/// in its record under `stamp` as `(stamp, start, count)`.
fn index_by<T: Copy + Default>(
    side: &mut Small,
    t: u32,
    idx: &mut Vec<T>,
    entry: impl Fn((Ref, Ref), u32) -> T,
    first: impl Fn(&T) -> Ref,
    stamp: u32,
    lim: &Limits,
) -> Result<(), OperationError> {
    let s = side.s;
    let lvl = &s.levels[t as usize];
    idx.clear();
    lim.try_resize_amortized(idx, lvl.pairs.len(), T::default())?;
    let nodes = || lvl.start.windows(2).enumerate().flat_map(|(i, w)| lvl.pairs[w[0] as usize..w[1] as usize].iter().map(move |&p| (p, i as u32)));
    // Each group's size, its start unset.
    for (p, i) in nodes() {
        let k = side.ids.id(first(&entry(p, i)));
        let r = &mut side.rec[k].run;
        if r.0 != stamp {
            *r = (stamp, NONE, 0);
        }
        r.2 += 1;
    }
    // Each group's start where its first entry is met, its count rebuilt
    // as the entries are placed.
    let mut next = 0u32;
    for (p, i) in nodes() {
        let x = entry(p, i);
        let r = &mut side.rec[side.ids.id(first(&x))].run;
        if r.1 == NONE {
            r.1 = next;
            next += r.2;
            r.2 = 0;
        }
        idx[(r.1 + r.2) as usize] = x;
        r.2 += 1;
    }
    Ok(())
}

/// The class of `S` at a level with one leaf child that the literals `m`
/// against a piece in `S`'s class `y` at the inner child lie in, given the
/// pairs of `S` there naming `y` (`owners`), or `None` where they split the
/// literals. A class no pair there names skips through the level.
#[inline]
fn one_class(m: u8, y: Ref, owners: &[(Ref, u8, u32)], t: u32) -> Option<Ref> {
    if y == ABSENT || owners.is_empty() {
        return Some(y);
    }
    for &(_, mb, j) in owners {
        let k = m & mb;
        if k != 0 {
            // The classes of `S` are disjoint: no other owner holds these.
            return (k == m).then_some(mk(t, j));
        }
    }
    Some(ABSENT)
}

/// The cells of one pair of a class at level `t` with one leaf child: each
/// piece `p` of the inner side, with the literals `m`, split by `S`'s pairs
/// that name the class of `p` there (`sidx`), each part tagged with the
/// class it lies in. A class of `S` that no pair there names skips through
/// `t` and holds the whole column.
#[allow(clippy::too_many_arguments)]
fn split(
    m: u8,
    pieces: &[(Ref, Ref)],
    small: &Small,
    sidx: &[(Ref, u8, u32)],
    stamp: u32,
    t: u32,
    orient: &impl Fn(u8, Ref) -> (Ref, Ref),
    tagged: &mut Vec<(Ref, Ref, Ref)>,
    lim: &Limits,
) -> Result<(), OperationError> {
    for &(p, y) in pieces {
        let mut rem = m;
        if y != ABSENT {
            let owners = small.named(sidx, y, stamp);
            if owners.is_empty() {
                let (a, b) = orient(m, p);
                lim.try_push(tagged, (y, a, b))?;
                continue;
            }
            for &(_, mb, j) in owners {
                let k = m & mb;
                if k != 0 {
                    let (a, b) = orient(k, p);
                    lim.try_push(tagged, (mk(t, j), a, b))?;
                }
                rem &= !mb;
            }
        }
        if rem != 0 {
            let (a, b) = orient(rem, p);
            lim.try_push(tagged, (ABSENT, a, b))?;
        }
    }
    Ok(())
}

/// A level with one leaf child. Each operand's classes are its stored nodes
/// and the references that skip through it. A skip of `P` is rebuilt where
/// `S`'s stored pairs there name the class of one of its pieces, and its
/// pieces are its child's otherwise; a skip of `S` has a new residue only
/// where a class `P`'s pairs name partly holds a piece in it.
fn lopsided(w: &mut Walk, t: u32) -> Result<(), OperationError> {
    let (sh, lim) = (w.sh, w.lim);
    let (l, r) = sh.kids[t as usize];
    let leaf_left = sh.is_leaf(l);
    let v = if leaf_left { l } else { r };
    let orient = |m: u8, e: Ref| {
        let lab = mk(v, cells_label(m));
        if leaf_left { (lab, e) } else { (e, lab) }
    };
    let inner_side = |p: (Ref, Ref)| if leaf_left { p } else { (p.1, p.0) };
    w.sc.stamp += 1;
    let stamp = w.sc.stamp;
    index_inner(&mut w.small, t, leaf_left, &mut w.sc.sidx, stamp, lim)?;
    // The skips of `P` through `t` with a piece in a class `S`'s pairs here
    // name, from that class's list.
    w.sc.touched.clear();
    let mut prev = ZREF;
    for &(y, _, _) in &w.sc.sidx {
        if y == prev {
            continue;
        }
        prev = y;
        let h = &mut w.small.rec[w.small.ids.id(y)].head;
        let mut k = std::mem::replace(h, WALKED);
        debug_assert!(k != WALKED, "a class read at two levels");
        if k == WALKED {
            k = NONE;
        }
        while k != NONE {
            w.small.walked += 1;
            let (x, next) = w.small.inv[k as usize];
            let r = &mut w.big.rec[w.big.ids.id(x)];
            if r.mark != stamp && sh.within(t, r.from) && r.pieces(&w.big.arena).iter().any(|p| p.1 == y) {
                debug_assert!(sh.within(vt(x), t) && vt(x) != t, "a skip through the level");
                r.mark = stamp;
                lim.try_push(&mut w.sc.touched, x)?;
            }
            k = next;
        }
    }
    let pl = &w.big.s.levels[t as usize];
    let mut gate = lim.gate();
    w.sc.names.clear();
    for j in 0..pl.width() as u32 {
        let node = pl.node(j);
        let y = mk(t, j);
        let mut tag = None;
        for &pair in node {
            let (lab, e) = inner_side(pair);
            let m = literal_cells(ix(lab));
            if w.residues && w.big.name(e, m, stamp) {
                lim.try_push(&mut w.sc.names, e)?;
            }
            if tag == Some(None) {
                continue;
            }
            let one = w.big.whole(e).and_then(|z| {
                let owners = if z == ABSENT { &[][..] } else { w.small.named(&w.sc.sidx, z, stamp) };
                one_class(m, z, owners, t)
            });
            tag = match (tag, one) {
                (None, Some(x)) => Some(Some(x)),
                (Some(Some(q)), Some(x)) if q == x => Some(Some(x)),
                _ => Some(None),
            };
        }
        gate.poll(node.len() as u64 + 1)?;
        if let Some(Some(tag)) = tag {
            stand(w, y, j, tag, t)?;
            continue;
        }
        w.sc.tagged.clear();
        for &pair in node {
            let (lab, e) = inner_side(pair);
            split(literal_cells(ix(lab)), w.big.pieces(e), &w.small, &w.sc.sidx, stamp, t, &orient, &mut w.sc.tagged, lim)?;
        }
        gate.poll(w.sc.tagged.len() as u64)?;
        emit(sh, &mut w.build, Some(j), w.rest, &mut w.sc.tagged, &mut w.sc.pairs, &mut w.sc.items, lim)?;
        w.big.set(y, &w.sc.items, t, &mut w.small, lim)?;
    }
    for i in 0..w.sc.touched.len() {
        let x = w.sc.touched[i];
        w.sc.tagged.clear();
        split(3, w.big.pieces(x), &w.small, &w.sc.sidx, stamp, t, &orient, &mut w.sc.tagged, lim)?;
        gate.poll(w.sc.tagged.len() as u64 + 1)?;
        emit(sh, &mut w.build, None, w.rest, &mut w.sc.tagged, &mut w.sc.pairs, &mut w.sc.items, lim)?;
        w.big.set(x, &w.sc.items, t, &mut w.small, lim)?;
    }
    if !w.residues {
        return gate.flush();
    }
    // `S`'s residues: the residues of its classes' inner sides, and the
    // literals the pairs of `P` naming a class leave out, against each piece
    // of that class, sent to the class of `S` the piece lies in. A class of
    // `P` no pair here names skips through `t` and leaves none out.
    let adds = &mut w.sc.adds;
    adds.clear();
    for &e in &w.sc.names {
        let cov = w.big.cells(e);
        if cov == 3 {
            continue;
        }
        let gap = 3 & !cov;
        for &(q, z) in w.big.pieces(e) {
            if z == ABSENT {
                continue;
            }
            let owners = w.small.named(&w.sc.sidx, z, stamp);
            if owners.is_empty() {
                lim.try_push(adds, (z, orient(gap, q)))?;
            }
            for &(_, mb, j) in owners {
                if gap & mb != 0 {
                    lim.try_push(adds, (mk(t, j), orient(gap & mb, q)))?;
                }
            }
        }
    }
    adds.sort_unstable_by_key(|x| x.0);
    let sl = &w.small.s.levels[t as usize];
    let mut at = 0;
    for j in 0..sl.width() as u32 {
        let class = mk(t, j);
        w.sc.pairs.clear();
        for &pair in sl.node(j) {
            let (lab, e) = inner_side(pair);
            let r = w.small.res(e);
            if r != ZREF {
                lim.try_push(&mut w.sc.pairs, orient(literal_cells(ix(lab)), r))?;
            }
        }
        // The added cells of the level's stored nodes come in node order.
        at += adds[at..].partition_point(|x| x.0 < class);
        while at < adds.len() && adds[at].0 == class {
            lim.try_push(&mut w.sc.pairs, adds[at].1)?;
            at += 1;
        }
        gate.poll(w.sc.pairs.len() as u64 + 1)?;
        let r = if w.sc.pairs.is_empty() { ZREF } else { w.build.piece(sh, &mut w.sc.pairs, lim)? };
        w.small.set_res(class, r, t);
    }
    // A skip of `S` with cells added: its residue, `true` beside its child's,
    // and those cells.
    for run in adds.chunk_by(|a, b| a.0 == b.0) {
        let z = run[0].0;
        if vt(z) == t {
            continue;
        }
        w.sc.pairs.clear();
        let r = w.small.res(z);
        if r != ZREF {
            lim.try_push(&mut w.sc.pairs, orient(3, r))?;
        }
        lim.reserve(&mut w.sc.pairs, run.len())?;
        w.sc.pairs.extend(run.iter().map(|x| x.1));
        gate.poll(w.sc.pairs.len() as u64)?;
        let r = w.build.piece(sh, &mut w.sc.pairs, lim)?;
        w.small.set_res(z, r, t);
    }
    gate.flush()
}

/// The cells of `a x b` for literal cells `a` of the left leaf and `b` of
/// the right one: bit `2i + j` for left literal `i` and right literal `j`,
/// 0 positive and 1 negative.
#[inline]
fn grid(a: u8, b: u8) -> u8 {
    let mut m = 0;
    for i in 0..2 {
        if a >> i & 1 != 0 {
            m |= (b & 3) << (2 * i);
        }
    }
    m
}

/// The classes of `s` at level `t`, whose children are the leaves `x` and
/// another, with their [`grid`] cells: its stored nodes, and the leaf
/// references it reads there through a skip (`skips`).
fn grid_classes(s: &SkipForm, skips: &FxHashMap<u32, Vec<Ref>>, t: u32, x: u32, cls: &mut Vec<(Ref, u8)>, lim: &Limits) -> Result<(), OperationError> {
    cls.clear();
    let lvl = &s.levels[t as usize];
    for i in 0..lvl.width() {
        let m = lvl.node(i as u32).iter().fold(0, |m, &(a, b)| m | grid(literal_cells(ix(a)), literal_cells(ix(b))));
        lim.try_push(cls, (mk(t, i as u32), m))?;
    }
    if let Some(skips) = skips.get(&t) {
        for &r in skips {
            let c = literal_cells(ix(r));
            lim.try_push(cls, (r, if vt(r) == x { grid(c, 3) } else { grid(3, c) }))?;
        }
    }
    Ok(())
}

/// The cells `m` at `t` (leaves `x` and `y`) as pairs.
fn grid_pairs(x: u32, y: u32, m: u8, pairs: &mut Vec<(Ref, Ref)>, lim: &Limits) -> Result<(), OperationError> {
    pairs.clear();
    for bit in 0..4u32 {
        if m >> bit & 1 != 0 {
            lim.try_push(pairs, (mk(x, POS + bit / 2), mk(y, POS + bit % 2)))?;
        }
    }
    Ok(())
}

/// A level whose children are both leaves: each operand has at most four
/// classes there, sets of the four cells, and every class is split.
fn leaves(w: &mut Walk, t: u32) -> Result<(), OperationError> {
    let (sh, lim) = (w.sh, w.lim);
    let (x, y) = sh.kids[t as usize];
    grid_classes(w.big.s, &w.big.skips, t, x, &mut w.sc.pcls, lim)?;
    grid_classes(w.small.s, &w.small.skips, t, x, &mut w.sc.scls, lim)?;
    let p_all = w.sc.pcls.iter().fold(0, |a, c| a | c.1);
    let s_all = w.sc.scls.iter().fold(0, |a, c| a | c.1);
    for k in 0..w.sc.pcls.len() {
        let (a, ma) = w.sc.pcls[k];
        let slot = (vt(a) == t).then_some(ix(a));
        // The class's parts by class of `S`, the rest last.
        let mut parts = [(ABSENT, 0u8); 5];
        let mut n = 0;
        for &(b, mb) in &w.sc.scls {
            if ma & mb != 0 {
                parts[n] = (b, ma & mb);
                n += 1;
            }
        }
        if ma & !s_all != 0 {
            parts[n] = (ABSENT, ma & !s_all);
            n += 1;
        }
        if n == 1 {
            match slot {
                Some(j) => stand(w, a, j, parts[0].0, t)?,
                None if parts[0].0 == ABSENT && !w.rest => w.big.set(a, &[], t, &mut w.small, lim)?,
                None => w.big.set(a, &[(a, parts[0].0)], t, &mut w.small, lim)?,
            }
            continue;
        }
        w.sc.items.clear();
        let mut taken = false;
        for &(b, m) in &parts[..n] {
            if b == ABSENT && !w.rest {
                continue;
            }
            grid_pairs(x, y, m, &mut w.sc.pairs, lim)?;
            let o = w.build.place(sh, slot, &mut taken, &mut w.sc.pairs, lim)?;
            lim.try_push(&mut w.sc.items, (o, b))?;
        }
        if let Some(j) = slot
            && !taken
        {
            w.build.kill(j, lim)?;
        }
        w.big.set(a, &w.sc.items, t, &mut w.small, lim)?;
    }
    if w.residues {
        for k in 0..w.sc.scls.len() {
            let (b, mb) = w.sc.scls[k];
            let rest = mb & !p_all;
            let r = if rest == 0 {
                ZREF
            } else {
                grid_pairs(x, y, rest, &mut w.sc.pairs, lim)?;
                w.build.piece(sh, &mut w.sc.pairs, lim)?
            };
            w.small.set_res(b, r, t);
        }
    }
    lim.charge_work(16);
    Ok(())
}
