//! Conditioning and quantification in skip form.
//!
//! `f` conditioned on `x = b` is `f ∧ [x = b]` with `x` made free. Once the
//! conjunction has fixed `x`, every node at the level `t` whose leaf child is
//! `x` holds one pair, on a literal of `x`; freeing `x` makes it `true` on `x`
//! beside the other side, a node the skip form does not store, so each
//! reference to it names the other side instead. A reference that skipped
//! through `t` to `x`'s leaf names `true` over `t`, which normalization
//! writes on `t`'s right leaf ([`super::normalize`] resolves the left leaf
//! first). Forgetting `x` maps the functions that fix it one to one onto
//! the functions that ignore it, so no other node changes, merges or becomes
//! one a skip passes: the result is canonical without a contraction.
//!
//! `∃X.f` is one bottom-up regroup of the levels above `X`'s leaves for the
//! whole set, as the stored form's ([`crate::apply::project`]). The classes
//! of a level are disjoint, so the image of one is the union of its pairs'
//! products of side images, and the images are cut into cells by the classes
//! that hold them: a cell per set of classes. Each vtree node is whole (every
//! variable below it quantified: its classes become `true`), split
//! (regrouped) or free (untouched, its references copied unread).
//!
//! In skip form the classes of a level are its stored nodes and the
//! references that skip through it from above. A skip is not visited at the
//! levels it passes: a cell all of whose classes skip through a level stays
//! one cell there, read through it. Where a stored node's pairs meet a cell
//! that a skip through the level also holds, the skip's part is regrouped
//! with them as a class of its own, `true` on the skipped leaf beside that
//! cell; the cell is lifted to the cells that part fell into, and a skip read
//! further up follows the lifts of its cells. Each cell knows the highest
//! level a class holding it is read at, which tells a level whether a skip
//! through it holds the cell. A skip to a node of a free level is never met:
//! its function ignores the quantified variables, so no other class's image
//! meets it.
//!
//! The raw result is deterministic, each level's cells pairwise disjoint, but
//! cells that every class above pairs alike are still apart; the contraction
//! merges them, from the levels with a quantified variable below them.
//!
//! For `∀`, which projects a negation, the pass first finds nodes whose
//! image is `true` by a sufficient rule, in one linear pass up the split
//! levels: a node is `true` when one of its pairs has both sides `true`, or,
//! against a leaf child that is not quantified, when its pairs on both
//! literals of the leaf have the other side `true`. Those pairs are the
//! node's witness: their image alone is `true`, so it is the node's whole
//! image. A node with `true` sides in a whole subtree, a quantified literal,
//! `true` in a free subtree, and a skip to such a node are `true` too. Where
//! the output is `true` the result is, with nothing regrouped.
//!
//! The rule needs one node to hold the witness, which a region of `true`
//! spread over several classes of a level escapes: under a right-linear
//! vtree a literal's cofactor is a union of classes of the level below.
//! A second flag, down from the output, credits the contexts instead. A
//! class wins where, whatever the free variables outside it, one choice of
//! the quantified ones puts all of it in the output: the output wins, so
//! does a side of a winning node's pair whose other side is `true`, and,
//! against a free leaf, a class that winning pairs of one level hold on both
//! of its literals (the choice outside may follow the leaf's value). The
//! result is `true` where a winning class is `true`, or where both literals
//! of a free leaf win. Neither flag reads a class through its siblings
//! when the sibling level is inner, so a balanced vtree can still need the
//! regroup to find `true`.
//!
//! Otherwise a marking down from the output keeps the nodes the regroup
//! reads: every pair of a node that is not `true`, and only the witness of
//! one that is, so that what only the other pairs of a `true` node reach is
//! never built.

use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use super::contract::contract_in;
use super::{build_top, discard_nested, filled, ix, literal_cells, merge_cells, mk, normalize, vt, Ref, SLevel, Shape, SkipForm, NEG, NONE, ONE, POS, ZREF};
use crate::limits::{Limits, PollGate};
use crate::OperationError;

/// `f` with the variable of leaf `leaf` made free, where `f` fixes it: every
/// node at its parent level holds one pair, on a literal of it.
pub(crate) fn free(mut f: SkipForm, leaf: u32, lim: &Limits) -> Result<SkipForm, OperationError> {
    if f.out == ZREF {
        return Ok(f);
    }
    let sh = Arc::clone(&f.shape);
    let t = sh.parent[leaf as usize];
    if t == NONE {
        // A vtree of one variable.
        f.out = mk(leaf, ONE);
        return Ok(f);
    }
    let (l, r) = sh.kids[t as usize];
    let level = f.levels.take(t as usize);
    let mut other = Vec::new();
    lim.reserve_exact(&mut other, level.width())?;
    for i in 0..level.width() {
        let pairs = level.node(i as u32);
        debug_assert!(pairs.len() == 1 && ix(if l == leaf { pairs[0].0 } else { pairs[0].1 }) != ONE, "the variable is fixed");
        let (a, b) = pairs[0];
        other.push(if l == leaf { b } else { a });
    }
    lim.release_bytes(level.bytes());
    // `true` over `t`, which a skip through it reaches where both its
    // children are leaves.
    let true_t = mk(r, ONE);
    let fix = |x: Ref| -> Ref {
        if x == ZREF {
            x
        } else if vt(x) == t {
            other[ix(x) as usize]
        } else if vt(x) == leaf && ix(x) != ONE {
            true_t
        } else {
            x
        }
    };
    f.active.retain(|&u| u != t);
    let mut buf = Vec::new();
    for &u in &f.active {
        let lvl: &mut SLevel = &mut f.levels[u as usize];
        for i in 0..lvl.width() {
            let node = &mut lvl.pairs[lvl.start[i] as usize..lvl.start[i + 1] as usize];
            let mut renamed = false;
            for p in node.iter_mut() {
                let q = (fix(p.0), fix(p.1));
                renamed |= q != *p;
                *p = q;
            }
            // A renamed side moves its pair in the node's stored order. The
            // renaming is one to one, so no pairs merge.
            if renamed && node.len() > 1 {
                buf.clear();
                lim.reserve(&mut buf, node.len())?;
                buf.extend_from_slice(node);
                merge_cells(&sh, u, &mut buf);
                debug_assert_eq!(buf.len(), node.len(), "freeing a variable merges no pairs");
                node.copy_from_slice(&buf);
            }
        }
    }
    lim.discard(buf);
    f.out = fix(f.out);
    lim.discard(other);
    Ok(f)
}

/// How a vtree node relates to the quantified variables.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Role {
    /// No quantified variable below it.
    Free,
    /// Every variable below it is quantified.
    Whole,
    /// Some, but not all, of its variables are quantified.
    Split,
}

/// The depth of a level nothing reads.
const UNREAD: u32 = u32::MAX;

/// `∃ X. f` for the variables of the distinct leaves `leaves`, canonical.
/// With `seek_true` the flags look for a `true` result and for nodes whose
/// image is `true` before the regroup: `∀X.g` is `¬∃X.¬g`, false where that
/// projection is `true`, which is where it is cheapest to find. A projection
/// of the diagram itself is rarely `true`, and the flags' two passes cost
/// about a twentieth of the regroup, so `∃` alone does not look.
pub(crate) fn exists(f: &SkipForm, leaves: &[u32], seek_true: bool, lim: &Limits) -> Result<SkipForm, OperationError> {
    let sh = &*f.shape;
    if f.out == ZREF || leaves.is_empty() {
        return Ok(f.clone());
    }
    let mut role = filled(lim, sh.num_nodes(), Role::Free)?;
    for &x in leaves {
        role[x as usize] = Role::Whole;
    }
    for &t in &sh.bottomup {
        let (l, r) = sh.kids[t as usize];
        role[t as usize] = match (role[l as usize], role[r as usize]) {
            (Role::Free, Role::Free) => Role::Free,
            (Role::Whole, Role::Whole) => Role::Whole,
            _ => Role::Split,
        };
    }
    match role[sh.root as usize] {
        Role::Free => {
            lim.discard(role);
            return Ok(f.clone());
        }
        Role::Whole => {
            // `f` is satisfiable.
            lim.discard(role);
            return all_true(f, lim);
        }
        Role::Split => {}
    }
    let mut q = Regroup::new(f, role, lim)?;
    if seek_true {
        q.mark_true()?;
        if q.is_true(f.out) || q.wins()? {
            let (raw, role) = q.into_raw();
            raw.discard(lim);
            lim.discard(role);
            return all_true(f, lim);
        }
    }
    q.note_readers();
    for &t in &sh.bottomup {
        if q.role[t as usize] == Role::Split {
            q.level(t)?;
        }
    }
    q.finish()
}

/// `true` over `f`'s vtree.
fn all_true(f: &SkipForm, lim: &Limits) -> Result<SkipForm, OperationError> {
    let mut out = SkipForm::zero(Arc::clone(&f.shape));
    out.out = top_of(&mut out, f.shape.root, lim)?;
    out.collect_pushed(lim)?;
    Ok(out)
}

/// The pairs of a node whose image is `true` by themselves, by index.
#[derive(Copy, Clone)]
enum Witness {
    None,
    One(u32),
    /// Both literals of a leaf child that is not quantified.
    Two(u32, u32),
}

/// The context's flag while it is played ([`Regroup::wins`]): per stored
/// node of a split level, and per other class in a map, the level whose
/// leaf's literals it was last credited with plus one (bits 2 and up) and
/// those literals (bits 0 and 1, both where it wins); per leaf that is not
/// quantified, its literals that win.
struct Game {
    split: Vec<Vec<u32>>,
    other: FxHashMap<Ref, u32>,
    leaf: Vec<u8>,
}

/// The pairs of `node` a regroup reads: all of them, or the witness.
fn read_pairs<'b>(node: &'b [(Ref, Ref)], w: Witness, buf: &'b mut [(Ref, Ref); 2]) -> &'b [(Ref, Ref)] {
    match w {
        Witness::None => node,
        Witness::One(k) => &node[k as usize..=k as usize],
        Witness::Two(j, k) => {
            *buf = [node[j as usize], node[k as usize]];
            &buf[..]
        }
    }
}

/// `true` over `u` as normalization writes it, its nodes pushed to `out`.
fn top_of(out: &mut SkipForm, u: u32, lim: &Limits) -> Result<Ref, OperationError> {
    let sh = Arc::clone(&out.shape);
    build_top(&sh, u, &mut |t, pair| Ok(mk(t, out.push_node(t, &[pair], lim)?)))
}

/// Each stored node's cells at a regrouped level: node `i`'s are
/// `cells[start[i]..start[i + 1]]`.
#[derive(Default)]
struct Img {
    start: Vec<u32>,
    cells: Vec<Ref>,
}

impl Img {
    fn get(&self, i: u32) -> &[Ref] {
        &self.cells[self.start[i as usize] as usize..self.start[i as usize + 1] as usize]
    }
}

/// The regroup's state while it climbs.
struct Regroup<'a> {
    f: &'a SkipForm,
    sh: Arc<Shape>,
    role: Vec<Role>,
    /// Per whole node, the highest whole node at or above it, whose parent
    /// is split: its classes there are all `true`.
    whole_top: Vec<u32>,
    /// Per stored node of a split level, the depth of the level its readers
    /// read it at; [`UNREAD`] where the regroup does not read it.
    reader: Vec<Vec<u32>>,
    /// Per stored node of a split level, whether its image is `true` by the
    /// sufficient rule.
    truth: Vec<Vec<bool>>,
    /// How many of those are `true`: none unless the flags ran.
    trues: u64,
    /// Per internal node of a free subtree, whether its level holds `true`.
    free_true: Vec<bool>,
    /// Per whole top, the least depth of a level a reference into its
    /// subtree is read at.
    hi_in: Vec<u32>,
    /// Per split level, a bound below the depths [`Regroup::hi`] holds for
    /// the cells it hands up: where it is deeper than the parent level, no
    /// skip through the parent holds one of them.
    low: Vec<u32>,
    /// The raw result's nodes.
    out: SkipForm,
    /// `true` over each whole top, once built.
    tops: Vec<Ref>,
    /// Per split level, its stored nodes' cells.
    img: Vec<Img>,
    /// Per cell, the least depth of a level one of the classes holding it is
    /// read at.
    hi: FxHashMap<Ref, u32>,
    /// Per cell that a skip through the level above held and that was
    /// regrouped there, the cells it became: a run of `arena`.
    lift: FxHashMap<Ref, (u32, u32)>,
    arena: Vec<Ref>,
    stack: Vec<Ref>,
    lim: &'a Limits,
    gate: PollGate<'a>,
    // Scratch of one level.
    sa: Vec<Ref>,
    sb: Vec<Ref>,
    /// The atoms, with the class holding them: stored nodes by index, then
    /// the lifted cells' parts.
    trip: Vec<(Ref, Ref, u32)>,
    /// The cells lifted here, and on which side.
    lam: Vec<(Ref, bool)>,
    lam_seen: FxHashSet<Ref>,
    pairs: Vec<(Ref, Ref)>,
}

impl<'a> Regroup<'a> {
    fn new(f: &'a SkipForm, role: Vec<Role>, lim: &'a Limits) -> Result<Regroup<'a>, OperationError> {
        let sh = Arc::clone(&f.shape);
        let n = sh.num_nodes();
        let mut whole_top = filled(lim, n, NONE)?;
        for &t in &sh.topdown {
            let (l, r) = sh.kids[t as usize];
            for c in [l, r] {
                if role[c as usize] == Role::Whole {
                    whole_top[c as usize] = if role[t as usize] == Role::Whole { whole_top[t as usize] } else { c };
                }
            }
        }
        let mut reader: Vec<Vec<u32>> = Vec::new();
        lim.try_resize(&mut reader, n, Vec::new())?;
        let mut truth: Vec<Vec<bool>> = Vec::new();
        lim.try_resize(&mut truth, n, Vec::new())?;
        for &t in &f.active {
            if role[t as usize] == Role::Split {
                reader[t as usize] = filled(lim, f.levels[t as usize].width(), UNREAD)?;
                truth[t as usize] = filled(lim, f.levels[t as usize].width(), false)?;
            }
        }
        let mut img: Vec<Img> = Vec::new();
        lim.reserve_exact(&mut img, n)?;
        img.resize_with(n, Img::default);
        Ok(Regroup {
            f,
            role,
            whole_top,
            reader,
            truth,
            trues: 0,
            free_true: filled(lim, n, false)?,
            hi_in: filled(lim, n, UNREAD)?,
            low: filled(lim, n, UNREAD)?,
            out: SkipForm::zero(Arc::clone(&sh)),
            tops: filled(lim, n, ZREF)?,
            img,
            hi: FxHashMap::default(),
            lift: FxHashMap::default(),
            arena: Vec::new(),
            stack: Vec::new(),
            lim,
            gate: lim.gate(),
            sa: Vec::new(),
            sb: Vec::new(),
            trip: Vec::new(),
            lam: Vec::new(),
            lam_seen: FxHashSet::default(),
            pairs: Vec::new(),
            sh,
        })
    }

    /// Whether `∃X` of what `x` names is `true` by the sufficient rule, read
    /// at any level at or above its node: the levels a skip passes are `true`
    /// on their skipped leaf.
    fn is_true(&self, x: Ref) -> bool {
        if x == ZREF {
            return false;
        }
        let v = vt(x);
        match self.role[v as usize] {
            Role::Whole => true,
            Role::Free if self.sh.is_leaf(v) => ix(x) == ONE,
            Role::Free => self.free_true[v as usize],
            Role::Split => self.truth[v as usize][ix(x) as usize],
        }
    }

    /// The witness of a node with pairs `node` at split level `w`, from the
    /// truth of the levels below.
    fn witness(&self, w: u32, node: &[(Ref, Ref)]) -> Witness {
        let (l, r) = self.sh.kids[w as usize];
        // A leaf child that is not quantified, and on which side.
        let free_leaf = |c: u32| self.sh.is_leaf(c) && self.role[c as usize] == Role::Free;
        let side = if free_leaf(l) { Some(true) } else if free_leaf(r) { Some(false) } else { None };
        let (mut pos, mut neg) = (NONE, NONE);
        for (k, &(a, b)) in node.iter().enumerate() {
            let (ta, tb) = (self.is_true(a), self.is_true(b));
            if ta && tb {
                return Witness::One(k as u32);
            }
            let lit = match side {
                Some(true) if tb => ix(a),
                Some(false) if ta => ix(b),
                _ => continue,
            };
            if lit == POS {
                pos = k as u32;
            } else if lit == NEG {
                neg = k as u32;
            }
        }
        if pos != NONE && neg != NONE { Witness::Two(pos, neg) } else { Witness::None }
    }

    /// Whether the result is `true` by the context's flag, top down from
    /// the output. A class wins where, whatever the variables outside it
    /// that are not quantified, one choice of the quantified ones puts all
    /// of it in the output. The output wins; so does a side of a winning
    /// node's pair whose other side is `true`, and, against a leaf that is
    /// not quantified, a side that winning pairs of one level hold on both
    /// literals of the leaf. A skip wins where its target does: the levels
    /// it passes are `true` on their leaf. The result is `true` where a class
    /// that wins is `true`, or where both literals of a leaf win.
    fn wins(&mut self) -> Result<bool, OperationError> {
        let sh = Arc::clone(&self.sh);
        let lim = self.lim;
        let f: &'a SkipForm = self.f;
        let n = sh.num_nodes();
        let mut game = Game { split: Vec::new(), other: FxHashMap::default(), leaf: filled(lim, n, 0u8)? };
        lim.try_resize(&mut game.split, n, Vec::new())?;
        for &t in &f.active {
            if self.role[t as usize] == Role::Split {
                game.split[t as usize] = filled(lim, f.levels[t as usize].width(), 0u32)?;
            }
        }
        let mut won = self.play(&mut game, f.out, sh.root, 3, lim)?;
        // Parents first: every level that reads a class is above it.
        for &t in &f.active {
            if won {
                break;
            }
            if self.role[t as usize] != Role::Split {
                continue;
            }
            let (l, r) = sh.kids[t as usize];
            let free_leaf = |c: u32| sh.is_leaf(c) && self.role[c as usize] == Role::Free;
            let (lf, rf) = (free_leaf(l), free_leaf(r));
            let lvl = &f.levels[t as usize];
            for i in 0..lvl.width() {
                if game.split[t as usize][i] & 3 != 3 {
                    continue;
                }
                let node = lvl.node(i as u32);
                for &(a, b) in node {
                    let (ta, tb) = (self.is_true(a), self.is_true(b));
                    let bits_a = if tb { 3 } else if rf { literal_cells(ix(b)) } else { 0 };
                    let bits_b = if ta { 3 } else if lf { literal_cells(ix(a)) } else { 0 };
                    if bits_a != 0 {
                        won |= self.play(&mut game, a, t, bits_a, lim)?;
                    }
                    if bits_b != 0 {
                        won |= self.play(&mut game, b, t, bits_b, lim)?;
                    }
                }
                self.gate.poll(node.len() as u64)?;
            }
        }
        discard_nested(lim, game.split);
        lim.discard(game.other);
        lim.discard(game.leaf);
        Ok(won)
    }

    /// Credit `x`, read at level `w`, with the literals `bits` of `w`'s leaf
    /// child (both where it wins outright); whether the result is `true`.
    fn play(&self, game: &mut Game, x: Ref, w: u32, bits: u8, lim: &Limits) -> Result<bool, OperationError> {
        if x == ZREF {
            return Ok(false);
        }
        let v = vt(x);
        let slot = match self.role[v as usize] {
            Role::Split => &mut game.split[v as usize][ix(x) as usize],
            _ => {
                lim.reserve_map(&mut game.other, 1)?;
                game.other.entry(x).or_insert(0)
            }
        };
        // A game played at another level starts over.
        if *slot & 3 != 3 && *slot >> 2 != w + 1 {
            *slot = (w + 1) << 2;
        }
        *slot |= bits as u32;
        if *slot & 3 != 3 {
            return Ok(false);
        }
        Ok(match self.role[v as usize] {
            Role::Free if self.sh.is_leaf(v) => {
                game.leaf[v as usize] |= literal_cells(ix(x));
                game.leaf[v as usize] == 3
            }
            _ => self.is_true(x),
        })
    }

    /// The truth of every stored node of the split levels and of every free
    /// level, children first.
    fn mark_true(&mut self) -> Result<(), OperationError> {
        let sh = Arc::clone(&self.sh);
        let f: &'a SkipForm = self.f;
        for &t in &sh.bottomup {
            if self.role[t as usize] != Role::Free {
                continue;
            }
            // `true` over a free subtree is the one node of its level.
            let lvl = &f.levels[t as usize];
            self.free_true[t as usize] = lvl.width() == 1 && matches!(*lvl.node(0), [(a, b)] if self.is_true(a) && self.is_true(b));
        }
        for &t in f.active.iter().rev() {
            if self.role[t as usize] != Role::Split {
                continue;
            }
            let lvl = &f.levels[t as usize];
            let mut flags = std::mem::take(&mut self.truth[t as usize]);
            for (i, flag) in flags.iter_mut().enumerate() {
                *flag = !matches!(self.witness(t, lvl.node(i as u32)), Witness::None);
                self.trues += *flag as u64;
            }
            self.gate.poll(lvl.pairs.len() as u64)?;
            self.truth[t as usize] = flags;
        }
        Ok(())
    }

    /// Note that `x` is read at child level `c`.
    fn note(&mut self, x: Ref, c: u32) {
        let v = vt(x) as usize;
        let d = self.sh.depth[c as usize];
        match self.role[v] {
            Role::Free => {}
            Role::Whole => {
                let u = self.whole_top[v] as usize;
                self.hi_in[u] = self.hi_in[u].min(d);
            }
            Role::Split => {
                let s = &mut self.reader[v][ix(x) as usize];
                *s = (*s).min(d);
            }
        }
    }

    /// The nodes the regroup reads, down from the output, and the depth
    /// each class of a split level and each whole top is read at: a stored
    /// node by its parent level or through a skip, the levels above all
    /// split. A node is read where a node read above reads it: through any
    /// pair of one that is not `true`, through the witness of one that is.
    fn note_readers(&mut self) {
        let sh = Arc::clone(&self.sh);
        let f: &'a SkipForm = self.f;
        self.note(f.out, sh.root);
        let mut buf = [(ZREF, ZREF); 2];
        // Parents first: every reader of a level is above it.
        for &t in &f.active {
            if self.role[t as usize] != Role::Split {
                continue;
            }
            let (l, r) = sh.kids[t as usize];
            let lvl = &f.levels[t as usize];
            if self.trues == 0 {
                // No node is `true`: every node is read, through all its pairs.
                for &(a, b) in &lvl.pairs {
                    self.note(a, l);
                    self.note(b, r);
                }
                continue;
            }
            for i in 0..lvl.width() as u32 {
                if self.reader[t as usize][i as usize] == UNREAD {
                    continue;
                }
                let node = lvl.node(i);
                let w = if self.truth[t as usize][i as usize] { self.witness(t, node) } else { Witness::None };
                for &(a, b) in read_pairs(node, w, &mut buf) {
                    self.note(a, l);
                    self.note(b, r);
                }
            }
        }
    }

    /// `true` over whole top `u`, built once.
    fn top(&mut self, u: u32) -> Result<Ref, OperationError> {
        if self.tops[u as usize] == ZREF {
            let t = top_of(&mut self.out, u, self.lim)?;
            self.tops[u as usize] = t;
            self.lim.reserve_map(&mut self.hi, 1)?;
            self.hi.insert(t, self.hi_in[u as usize]);
        }
        Ok(self.tops[u as usize])
    }

    /// [`Regroup::low`] of a child of a split level.
    fn low_of(&self, c: u32) -> u32 {
        match self.role[c as usize] {
            Role::Free => UNREAD,
            Role::Whole => self.hi_in[c as usize],
            Role::Split => self.low[c as usize],
        }
    }

    /// The cells of leaf `c`'s label `label`: `true` where `c` is
    /// quantified, else its literals.
    fn leaf_cells(&mut self, c: u32, label: u32, buf: &mut Vec<Ref>) -> Result<(), OperationError> {
        if self.role[c as usize] == Role::Whole {
            let t = self.top(c)?;
            return self.lim.try_push(buf, t);
        }
        if label == ONE {
            self.lim.reserve(buf, 2)?;
            buf.push(mk(c, POS));
            buf.push(mk(c, NEG));
            return Ok(());
        }
        self.lim.try_push(buf, mk(c, label))
    }

    /// The cells of `x` read at child level `c` of the level being
    /// regrouped, appended to `buf`.
    fn side(&mut self, c: u32, x: Ref, buf: &mut Vec<Ref>) -> Result<(), OperationError> {
        let v = vt(x);
        if v == c && self.sh.is_leaf(c) {
            return self.leaf_cells(c, ix(x), buf);
        }
        match self.role[v as usize] {
            Role::Free => self.lim.try_push(buf, x),
            Role::Whole => {
                let t = self.top(self.whole_top[v as usize])?;
                if v == c { self.lim.try_push(buf, t) } else { self.resolve(t, buf) }
            }
            Role::Split => {
                let img = &self.img[v as usize];
                let (lo, hi) = (img.start[ix(x) as usize] as usize, img.start[ix(x) as usize + 1] as usize);
                if v == c {
                    // Read at its own level: no cell of it was lifted yet.
                    self.lim.reserve(buf, hi - lo)?;
                    buf.extend_from_slice(&img.cells[lo..hi]);
                    return Ok(());
                }
                for k in lo..hi {
                    let d = self.img[v as usize].cells[k];
                    self.resolve(d, buf)?;
                }
                Ok(())
            }
        }
    }

    /// The cells `d` became through the lifts above it, appended to `buf`;
    /// a lift several levels deep is replaced by what it resolved to.
    fn resolve(&mut self, d: Ref, buf: &mut Vec<Ref>) -> Result<(), OperationError> {
        let Some(&(s, n)) = self.lift.get(&d) else {
            return self.lim.try_push(buf, d);
        };
        let mark = buf.len();
        let mut deep = false;
        self.stack.clear();
        self.lim.reserve(&mut self.stack, n as usize)?;
        self.stack.extend_from_slice(&self.arena[s as usize..(s + n) as usize]);
        while let Some(x) = self.stack.pop() {
            match self.lift.get(&x) {
                None => self.lim.try_push(buf, x)?,
                Some(&(s2, n2)) => {
                    deep = true;
                    self.lim.reserve(&mut self.stack, n2 as usize)?;
                    self.stack.extend_from_slice(&self.arena[s2 as usize..(s2 + n2) as usize]);
                }
            }
        }
        self.gate.poll((buf.len() - mark) as u64)?;
        if deep {
            let at = self.arena.len() as u32;
            self.lim.reserve(&mut self.arena, buf.len() - mark)?;
            self.arena.extend_from_slice(&buf[mark..]);
            self.lift.insert(d, (at, (buf.len() - mark) as u32));
        }
        Ok(())
    }

    /// Regroup split level `w`.
    fn level(&mut self, w: u32) -> Result<(), OperationError> {
        let sh = Arc::clone(&self.sh);
        let lim = self.lim;
        let (l, r) = sh.kids[w as usize];
        let (ll, rl) = (sh.is_leaf(l), sh.is_leaf(r));
        let f: &'a SkipForm = self.f;
        let lvl = &f.levels[w as usize];
        let width = lvl.width() as u32;
        let dw = sh.depth[w as usize];
        self.lam.clear();
        self.lam_seen.clear();
        // Where skips pass `w` into an internal child, the child whose cells
        // a skip may hold, and whether it is the left one.
        let mut watch = None;
        if ll && rl {
            // Skips to a quantified leaf: `true` over the leaf, beside the
            // other, is written on the right leaf.
            for (c, left) in [(l, true), (r, false)] {
                if self.role[c as usize] == Role::Whole && self.hi_in[c as usize] <= dw {
                    let t = self.top(c)?;
                    lim.try_push(&mut self.lam, (t, left))?;
                }
            }
        } else if ll != rl {
            let (c, left) = if ll { (r, false) } else { (l, true) };
            if self.low_of(c) <= dw {
                watch = Some(left);
            }
        }
        self.trip.clear();
        let (mut sa, mut sb) = (std::mem::take(&mut self.sa), std::mem::take(&mut self.sb));
        let mut buf = [(ZREF, ZREF); 2];
        for i in 0..width {
            if self.reader[w as usize][i as usize] == UNREAD {
                continue;
            }
            let node = lvl.node(i);
            let wit = if self.truth[w as usize][i as usize] { self.witness(w, node) } else { Witness::None };
            let node = read_pairs(node, wit, &mut buf);
            self.gate.poll(node.len() as u64)?;
            for &(a, b) in node {
                sa.clear();
                self.side(l, a, &mut sa)?;
                sb.clear();
                self.side(r, b, &mut sb)?;
                lim.reserve(&mut self.trip, sa.len() * sb.len())?;
                for &x in &sa {
                    for &y in &sb {
                        self.trip.push((x, y, i));
                    }
                }
                if let Some(left) = watch {
                    for &d in if left { &sa } else { &sb } {
                        if self.hi.get(&d).is_some_and(|&h| h <= dw) && !self.lam_seen.contains(&d) {
                            lim.reserve_set(&mut self.lam_seen, 1)?;
                            self.lam_seen.insert(d);
                            lim.try_push(&mut self.lam, (d, left))?;
                        }
                    }
                }
            }
        }
        // Each lifted cell's part: `true` on the skipped leaf beside it.
        for k in 0..self.lam.len() {
            let (d, left) = self.lam[k];
            let skipped = if left { r } else { l };
            sa.clear();
            self.leaf_cells(skipped, ONE, &mut sa)?;
            lim.reserve(&mut self.trip, sa.len())?;
            for &y in &sa {
                self.trip.push(if left { (d, y, width + k as u32) } else { (y, d, width + k as u32) });
            }
        }
        self.sa = sa;
        self.sb = sb;
        let owners = width as usize + self.lam.len();
        // The cells of the child skips pass into that no stored node met
        // pass up as they are.
        let into_low = if ll != rl { self.low_of(if ll { r } else { l }) } else { UNREAD };
        if self.trip.is_empty() {
            // No class here is read.
            self.low[w as usize] = into_low;
            return Ok(());
        }
        let cells = if owners == 1 { self.one_cell(w)? } else { self.cells(w, owners)? };
        // Per cell its reference and the classes holding it, by owner.
        let Cells { refs, owner_start, owner_items } = cells;
        let mut low = into_low;
        let mut hi_of = Vec::new();
        lim.reserve_exact(&mut hi_of, refs.len())?;
        for c in 0..refs.len() {
            let mut h = UNREAD;
            for &o in &owner_items[owner_start[c] as usize..owner_start[c + 1] as usize] {
                h = h.min(if o < width { self.reader[w as usize][o as usize] } else { self.lam_hi(o - width) });
            }
            hi_of.push(h);
            low = low.min(h);
        }
        self.low[w as usize] = low;
        // The stored nodes' cells, and the lifted cells' lifts.
        let mut count = filled(lim, owners + 1, 0u32)?;
        for &o in &owner_items {
            count[o as usize + 1] += 1;
        }
        for o in 0..owners {
            count[o + 1] += count[o];
        }
        let mut by_owner = filled(lim, owner_items.len(), ZREF)?;
        let mut at = filled(lim, owners, 0u32)?;
        at.copy_from_slice(&count[..owners]);
        for c in 0..refs.len() {
            for &o in &owner_items[owner_start[c] as usize..owner_start[c + 1] as usize] {
                by_owner[at[o as usize] as usize] = refs[c];
                at[o as usize] += 1;
            }
        }
        lim.discard(at);
        if width > 0 {
            let mut start = Vec::new();
            lim.reserve_exact(&mut start, width as usize + 1)?;
            start.extend_from_slice(&count[..=width as usize]);
            let mut cells = Vec::new();
            let n = count[width as usize] as usize;
            lim.reserve_exact(&mut cells, n)?;
            cells.extend_from_slice(&by_owner[..n]);
            self.img[w as usize] = Img { start, cells };
        }
        for k in 0..self.lam.len() {
            let (d, _) = self.lam[k];
            let o = width as usize + k;
            let run = &by_owner[count[o] as usize..count[o + 1] as usize];
            if run != [d] {
                let at = self.arena.len() as u32;
                lim.reserve(&mut self.arena, run.len())?;
                self.arena.extend_from_slice(run);
                lim.reserve_map(&mut self.lift, 1)?;
                self.lift.insert(d, (at, run.len() as u32));
            }
        }
        lim.reserve_map(&mut self.hi, refs.len())?;
        for (c, &x) in refs.iter().enumerate() {
            self.hi.insert(x, hi_of[c]);
        }
        lim.discard(by_owner);
        lim.discard(count);
        lim.discard(hi_of);
        lim.discard(refs);
        lim.discard(owner_start);
        lim.discard(owner_items);
        Ok(())
    }

    /// The depth of the highest reader of lifted cell `k`'s classes.
    fn lam_hi(&self, k: u32) -> u32 {
        let d = self.lam[k as usize].0;
        self.hi.get(&d).copied().unwrap_or(UNREAD)
    }

    /// The node of `pairs` at `w`, or the reference it skips to.
    fn node(&mut self, w: u32, pairs: &mut Vec<(Ref, Ref)>) -> Result<Ref, OperationError> {
        match normalize(&self.sh, w, pairs) {
            Err(x) => Ok(x),
            Ok(()) => Ok(mk(w, self.out.push_node(w, pairs, self.lim)?)),
        }
    }

    /// The one cell of a level with one class.
    fn one_cell(&mut self, w: u32) -> Result<Cells, OperationError> {
        let lim = self.lim;
        let mut pairs = std::mem::take(&mut self.pairs);
        pairs.clear();
        lim.reserve(&mut pairs, self.trip.len())?;
        pairs.extend(self.trip.iter().map(|&(x, y, _)| (x, y)));
        let x = self.node(w, &mut pairs);
        self.pairs = pairs;
        let mut refs = Vec::new();
        lim.try_push(&mut refs, x?)?;
        let mut owner_start = Vec::new();
        lim.reserve_exact(&mut owner_start, 2)?;
        owner_start.extend([0, 1]);
        let mut owner_items = Vec::new();
        lim.try_push(&mut owner_items, 0u32)?;
        Ok(Cells { refs, owner_start, owner_items })
    }

    /// The cells of level `w` from its atoms: one per set of classes holding
    /// an atom, numbered as the scan first reaches one of their atoms.
    fn cells(&mut self, w: u32, owners: usize) -> Result<Cells, OperationError> {
        let lim = self.lim;
        // Atoms by number, each with its classes in increasing order: the
        // atoms were listed by class.
        let mut atom_of: FxHashMap<(Ref, Ref), u32> = FxHashMap::default();
        lim.reserve_map(&mut atom_of, self.trip.len())?;
        let mut atoms: Vec<(Ref, Ref)> = Vec::new();
        let mut last: Vec<u32> = Vec::new();
        let mut entries: Vec<(u32, u32)> = Vec::new();
        lim.reserve(&mut entries, self.trip.len())?;
        for &(x, y, o) in &self.trip {
            let a = match atom_of.get(&(x, y)) {
                Some(&a) => a,
                None => {
                    let a = atoms.len() as u32;
                    atom_of.insert((x, y), a);
                    lim.try_push(&mut atoms, (x, y))?;
                    lim.try_push(&mut last, NONE)?;
                    a
                }
            };
            if last[a as usize] != o {
                last[a as usize] = o;
                entries.push((a, o));
            }
        }
        self.gate.poll(self.trip.len() as u64)?;
        lim.discard(atom_of);
        lim.discard(last);
        let (own_start, own) = pack(lim, atoms.len(), entries.iter().copied())?;
        lim.discard(entries);
        let owners_of = |a: usize| &own[own_start[a] as usize..own_start[a + 1] as usize];
        // Atoms by cell: a class alone finds its cell by number, a set by
        // its hash, confirmed against the cell's first atom.
        let mut by_one = filled(lim, owners, NONE)?;
        let mut by_hash: FxHashMap<u64, u32> = FxHashMap::default();
        let mut alike: Vec<u32> = Vec::new();
        let mut first: Vec<u32> = Vec::new();
        let mut cell_of = Vec::new();
        lim.reserve_exact(&mut cell_of, atoms.len())?;
        for a in 0..atoms.len() {
            let mine = owners_of(a);
            let cell = if let [o] = *mine {
                let c = by_one[o as usize];
                if c != NONE {
                    c
                } else {
                    let c = first.len() as u32;
                    by_one[o as usize] = c;
                    lim.try_push(&mut first, a as u32)?;
                    lim.try_push(&mut alike, NONE)?;
                    c
                }
            } else {
                let h = set_hash(mine);
                let mut c = by_hash.get(&h).copied().unwrap_or(NONE);
                while c != NONE && owners_of(first[c as usize] as usize) != mine {
                    c = alike[c as usize];
                }
                if c == NONE {
                    c = first.len() as u32;
                    lim.reserve_map(&mut by_hash, 1)?;
                    let prev = by_hash.insert(h, c).unwrap_or(NONE);
                    lim.try_push(&mut alike, prev)?;
                    lim.try_push(&mut first, a as u32)?;
                }
                c
            };
            cell_of.push(cell);
        }
        lim.discard(by_one);
        lim.discard(by_hash);
        lim.discard(alike);
        let n = first.len();
        let (cstart, members) = pack(lim, n, cell_of.iter().enumerate().map(|(a, &c)| (c, a as u32)))?;
        lim.discard(cell_of);
        let mut refs = Vec::new();
        lim.reserve_exact(&mut refs, n)?;
        let mut owner_start = Vec::new();
        lim.reserve_exact(&mut owner_start, n + 1)?;
        owner_start.push(0u32);
        let mut owner_items = Vec::new();
        let mut pairs = std::mem::take(&mut self.pairs);
        for c in 0..n {
            let ms = &members[cstart[c] as usize..cstart[c + 1] as usize];
            pairs.clear();
            lim.reserve(&mut pairs, ms.len())?;
            pairs.extend(ms.iter().map(|&a| atoms[a as usize]));
            self.gate.poll(ms.len() as u64)?;
            let x = match self.node(w, &mut pairs) {
                Ok(x) => x,
                Err(e) => {
                    self.pairs = pairs;
                    return Err(e);
                }
            };
            refs.push(x);
            let mine = owners_of(first[c] as usize);
            lim.reserve(&mut owner_items, mine.len())?;
            owner_items.extend_from_slice(mine);
            owner_start.push(owner_items.len() as u32);
        }
        self.pairs = pairs;
        lim.discard(cstart);
        lim.discard(members);
        lim.discard(first);
        lim.discard(own_start);
        lim.discard(own);
        lim.discard(atoms);
        Ok(Cells { refs, owner_start, owner_items })
    }

    /// The raw result, contracted to the canonical form.
    fn finish(mut self) -> Result<SkipForm, OperationError> {
        let lim = self.lim;
        let sh = Arc::clone(&self.sh);
        let f = self.f;
        let mut cells = std::mem::take(&mut self.sa);
        cells.clear();
        // The output is the one class of every level from the root down to
        // its node, which no other stored node is above, so it is one cell.
        self.side(sh.root, f.out, &mut cells)?;
        debug_assert_eq!(cells.len(), 1, "the output is one cell");
        let out = cells[0];
        lim.discard(cells);
        self.gate.flush()?;
        let (mut raw, role) = self.into_raw();
        raw.out = out;
        // The free levels as they are: every node there is still read.
        let mut cand: Vec<u32> = Vec::new();
        lim.reserve_exact(&mut cand, raw.active.len())?;
        cand.extend_from_slice(&raw.active);
        for &t in &f.active {
            if role[t as usize] == Role::Free {
                raw.levels.share(t as usize, &f.levels, lim)?;
                lim.try_push(&mut raw.active, t)?;
            }
        }
        raw.collect_in(cand, lim)?;
        let mut cone = filled(lim, sh.num_nodes(), false)?;
        for (c, &r) in cone.iter_mut().zip(&role) {
            *c = r != Role::Free;
        }
        lim.discard(role);
        let res = contract_in(raw, Some(&cone), lim);
        lim.discard(cone);
        res
    }

    /// The raw result's nodes and the roles, every other charge handed back.
    fn into_raw(self) -> (SkipForm, Vec<Role>) {
        let Regroup { role, whole_top, reader, truth, free_true, hi_in, low, out, tops, img, hi, lift, arena, stack, sa, sb, trip, lam, lam_seen, pairs, lim, .. } = self;
        lim.discard(whole_top);
        discard_nested(lim, reader);
        discard_nested(lim, truth);
        lim.discard(free_true);
        lim.discard(hi_in);
        lim.discard(low);
        lim.discard(tops);
        for i in img {
            lim.discard(i.start);
            lim.discard(i.cells);
        }
        lim.discard(hi);
        lim.discard(lift);
        lim.discard(arena);
        lim.discard(stack);
        lim.discard(sa);
        lim.discard(sb);
        lim.discard(trip);
        lim.discard(lam);
        lim.discard(lam_seen);
        lim.discard(pairs);
        (out, role)
    }
}

/// A level's cells: each one's reference, and the classes holding it as
/// runs of `owner_items`.
struct Cells {
    refs: Vec<Ref>,
    owner_start: Vec<u32>,
    owner_items: Vec<u32>,
}

/// `entries` grouped by their key below `keys`, stably: key `k`'s items are
/// `items[start[k]..start[k + 1]]`.
fn pack(lim: &Limits, keys: usize, entries: impl Iterator<Item = (u32, u32)> + Clone) -> Result<(Vec<u32>, Vec<u32>), OperationError> {
    let mut start = filled(lim, keys + 1, 0u32)?;
    let mut n = 0usize;
    for (k, _) in entries.clone() {
        start[k as usize + 1] += 1;
        n += 1;
    }
    for k in 0..keys {
        start[k + 1] += start[k];
    }
    let mut items = filled(lim, n, 0u32)?;
    let mut at = filled(lim, keys, 0u32)?;
    at.copy_from_slice(&start[..keys]);
    for (k, x) in entries {
        items[at[k as usize] as usize] = x;
        at[k as usize] += 1;
    }
    lim.discard(at);
    Ok((start, items))
}

/// A hash of a sorted set of classes.
fn set_hash(set: &[u32]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    set.hash(&mut h);
    h.finish()
}
