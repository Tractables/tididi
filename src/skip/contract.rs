//! Twin contraction in skip form.
//!
//! The stored-form contraction merges, level by level from the root, the
//! classes of a level that every node of the parent level pairs with the
//! same partners. In skip form a level's classes are its stored nodes and
//! the references that skip through it. A reference that skips a level
//! from above has a single owner there (the class it skips through one
//! level up), so it is never a twin; only the stored nodes and the skips
//! the parent level's pairs start can be. Merging a skip into a class makes
//! the skipped level's node explicit: `true` on the skipped variable paired
//! with the rest of the skip, whose own continuation below is then a class
//! the next level compares. Skips that merge stay twins down to the first
//! level where one of them is stored, is named by another pair, or where
//! they part, so their merged class is made just above it; the levels
//! between resolve to it.
//!
//! Twins are found by a hash of each entity's partners, summed so that
//! order does not matter, and confirmed by comparing the partner lists of
//! entities whose hashes agree. The diagram is then patched from the
//! leaves up: a stored node keeps its number, a merged class is appended,
//! and only nodes whose sides changed are rewritten.
//!
//! Above a summed-out subtree a node's pairs are a sum: twins merge into
//! the sum of their pairs, which keeps a repeated pair, a parent keeps the
//! pairs of one member for the group's, and `One` stands for both literals
//! only against a variable. A group stands for a stored member only beside
//! the classes it was found among: another level's skip to the member keeps
//! it, since that skip's class was not compared.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rustc_hash::FxHashMap;

use super::{filled, ix, literal_cells, mk, normalize, normalize_sum, vt, Ref, SLevel, Shape, SkipForm, NEG, ONE, POS, ZREF};

use crate::limits::Limits;
use crate::OperationError;

const NONE: u32 = u32::MAX;
/// A node's result where it is unchanged.
const KEEP: Ref = ZREF;

/// A visited level while the contraction runs.
struct CLv {
    level: u32,
    /// The input level's width: class `k < width` is stored node `k`, class
    /// `width + g` is merged group `g`.
    width: u32,
    /// Per stored node, the group it merged into (empty where none did).
    group_of: Vec<u32>,
    /// Group `g`'s pairs, as the input's references: its stored members'
    /// pairs, and `true` beside each skip it took in. Empty until the first
    /// group, then one more entry than groups.
    gstart: Vec<u32>,
    gpairs: Vec<(Ref, Ref)>,
    /// Whether a skip read here joined a group (here, or sent down from
    /// here): its join is listed by [`Ids::join`].
    joins: bool,
    /// Per group, the member whose pairs in the parent level's classes stand
    /// for the group's there, where a sum would count every member's.
    rep: Vec<Ref>,
    /// Per group, its result.
    gres: Vec<Ref>,
    /// Per stored node, its result where it no longer names itself (empty
    /// where none changed).
    rm: Vec<Ref>,
}

impl CLv {
    fn new(level: u32, width: u32) -> CLv {
        CLv {
            level,
            width,
            group_of: Vec::new(),
            gstart: Vec::new(),
            gpairs: Vec::new(),
            joins: false,
            rep: Vec::new(),
            gres: Vec::new(),
            rm: Vec::new(),
        }
    }
    fn groups(&self) -> usize {
        self.gstart.len().saturating_sub(1)
    }
    /// The number of the group about to be added, its start recorded.
    fn open(&mut self, lim: &Limits) -> Result<u32, OperationError> {
        if self.gstart.is_empty() {
            lim.try_push(&mut self.gstart, 0)?;
        }
        Ok(self.groups() as u32)
    }
    fn group(&self, g: usize) -> &[(Ref, Ref)] {
        &self.gpairs[self.gstart[g] as usize..self.gstart[g + 1] as usize]
    }
    /// Add a group of the stored nodes and skips `members` (`sh`'s level
    /// `c`, whose children are `l` and `r`), returning its number.
    fn push_group(&mut self, s: &SkipForm, members: &[Ref], l: u32, r: u32, lim: &Limits) -> Result<u32, OperationError> {
        let g = self.open(lim)?;
        for &e in members {
            if vt(e) == self.level {
                if self.group_of.is_empty() {
                    self.group_of = filled(lim, self.width as usize, NONE)?;
                }
                self.group_of[ix(e) as usize] = g;
                let node = s.levels[self.level as usize].node(ix(e));
                lim.reserve(&mut self.gpairs, node.len())?;
                self.gpairs.extend_from_slice(node);
            } else {
                lim.try_push(&mut self.gpairs, if s.shape.within(vt(e), l) { (e, mk(r, ONE)) } else { (mk(l, ONE), e) })?;
            }
        }
        lim.try_push(&mut self.gstart, self.gpairs.len() as u32)?;
        lim.try_push(&mut self.rep, members[0])?;
        Ok(g)
    }

    /// Hand the charge for the level's buffers back to `lim`.
    fn discard(self, lim: &Limits) {
        lim.discard(self.group_of);
        lim.discard(self.gstart);
        lim.discard(self.gpairs);
        lim.discard(self.rep);
        lim.discard(self.gres);
        lim.discard(self.rm);
    }
}

/// Skips that merged into one class beside a stored member, carried down
/// as one entity: a pair `(true, tail)` of the group stands for a pair
/// with each member. Listing the members at every level they pass would
/// cost a level per member, the members of a chain (one stored at each
/// level of a spine, the rest skipping through it) the square of their
/// number. A tail is `arenas[arena][lo..hi]`, sorted by preorder, less the
/// positions `gone`: members a pair names besides, split off as entities
/// of their own where the tail is read. `res` is what the tail resolves
/// to there: the group its live members joined and each split member.
struct Tail {
    arena: u32,
    lo: u32,
    hi: u32,
    gone: Vec<u32>,
    res: Vec<Res>,
}

/// A class a tail's members joined where it was read.
#[derive(Clone, Copy)]
enum Res {
    /// Group `g` of the visited level `to`.
    Group(u32, u32),
    /// The class a member resolves to where the tail is read.
    Via(Ref),
}

#[derive(Default)]
struct Tails {
    arenas: Vec<Vec<Ref>>,
    /// Per member's id ([`Ids`]), the arena and position it was last listed
    /// at (`NONE` where never).
    of: Vec<(u32, u32)>,
    list: Vec<Tail>,
}

impl Tails {
    fn push(&mut self, arena: u32, lo: u32, hi: u32, gone: Vec<u32>, lim: &Limits) -> Result<usize, OperationError> {
        lim.try_push(&mut self.list, Tail { arena, lo, hi, gone, res: Vec::new() })?;
        Ok(self.list.len() - 1)
    }

    /// The arena and position member `id` was last listed at.
    #[inline]
    fn of(&self, id: usize) -> Option<(u32, u32)> {
        self.of.get(id).copied().filter(|p| p.0 != NONE)
    }

    /// The live members of tail `t` at `lo..hi`.
    fn live(&self, (t, lo, hi): (u32, u32, u32)) -> usize {
        (hi - lo) as usize - self.list[t as usize].gone.iter().filter(|p| (lo..hi).contains(p)).count()
    }

    fn discard(self, lim: &Limits) {
        for a in self.arenas {
            lim.discard(a);
        }
        lim.discard(self.of);
        for t in self.list {
            lim.discard(t.gone);
            lim.discard(t.res);
        }
    }
}

/// A tail's reference in a group's pairs: the top bit set, which no node's
/// reference has.
const TTAG: Ref = 1 << 63;

#[inline]
fn is_tail(x: Ref) -> bool {
    x != ZREF && x & TTAG != 0
}

#[inline]
fn tail_ref(t: usize) -> Ref {
    TTAG | t as Ref
}

#[inline]
fn tail_id(x: Ref) -> usize {
    (x & 0xffff_ffff) as usize
}

/// An order-free hash of one (class, partner) occurrence.
#[inline]
fn mix(k: u32, p: Ref) -> u64 {
    let mut x = p.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (k as u64 + 1).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 29;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 32)
}

/// A node's pairs at `c` normalized: as a sum where a summed-out subtree
/// is below `c`.
fn made(sh: &Shape, c: u32, buf: &mut Vec<(Ref, Ref)>) -> Result<(), Ref> {
    if sh.weighted[c as usize] { normalize_sum(sh, c, buf) } else { normalize(sh, c, buf) }
}

/// `s` contracted to the canonical skip form of its function. `s` holds no
/// false side and no unreachable node.
pub(crate) fn contract(s: SkipForm, lim: &Limits) -> Result<SkipForm, OperationError> {
    contract_in(s, None, lim)
}

/// [`contract`] scoped to the upward-closed `cone` of changed levels when
/// `scope` is `Some(cone)`: levels outside `cone` whose parent had no merge
/// and lost no node are already canonical and are not walked.
pub(crate) fn contract_in(
    mut s: SkipForm,
    scope: Option<&[bool]>,
    lim: &Limits,
) -> Result<SkipForm, OperationError> {
    let shape = s.shape.clone();
    let sh = &*shape;
    let n = sh.num_nodes();
    if s.out == ZREF {
        let z = SkipForm::zero(s.shape.clone());
        s.discard(lim);
        return Ok(z);
    }
    let mut ids = Ids::new(n);
    // The levels visited, parents first: those holding nodes, and the
    // inner children of a level with classes, where skips it names may merge.
    let mut at = vec![NONE; n];
    let mut lvs: Vec<CLv> = Vec::new();
    let mut heap: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
    let mut queued = vec![false; n];
    for &t in &s.active {
        if scope.is_none_or(|cone| cone[t as usize]) {
            queued[t as usize] = true;
            heap.push(Reverse((sh.pre[t as usize], t)));
        }
    }
    // Per entity of the current level (a stored node or a skip the parent
    // names): its reference, partner hash and partner count.
    let mut epoch = 0u32;
    let mut ent: Vec<(Ref, u64, u32)> = Vec::new();
    let mut order: Vec<u32> = Vec::new();
    let mut table: Vec<u32> = Vec::new();
    let mut cand: Vec<bool> = Vec::new();
    let mut link: Vec<(u32, u32)> = Vec::new();
    let mut heads: Vec<u32> = Vec::new();
    let mut occ: Vec<(u32, u32, Ref)> = Vec::new();
    let mut ctx: Vec<(u32, u32)> = Vec::new();
    let mut plist: Vec<(u32, Ref)> = Vec::new();
    let mut firsts: Vec<u32> = Vec::new();
    let mut members: Vec<Ref> = Vec::new();
    // Merged skips sent down to the level where they part: per level, the
    // groups (entry they merged at, members).
    let mut pending: FxHashMap<u32, Vec<(u32, Vec<Ref>)>> = FxHashMap::default();
    let mut names = Names::default();
    let mut merged = false;
    // Tails (`Tails`): those read at the current level, members a pair
    // there names besides (entity, arena, position) and those split off
    // (tail, entity); per tail a stamp, entity and mark of the level.
    let mut tails = Tails::default();
    let mut here_tails: Vec<u32> = Vec::new();
    let mut maybe_split: Vec<(u32, u32, u32)> = Vec::new();
    let mut splits: Vec<(u32, u32)> = Vec::new();
    let (mut tstamp, mut tslot, mut tmark): (Vec<u32>, Vec<u32>, Vec<u32>) = Default::default();
    // Each level's groups to form, as runs of `forms`, and one's members.
    let (mut forms, mut fstart): (Vec<Ref>, Vec<u32>) = Default::default();
    let (mut stored, mut skips): (Vec<Ref>, Vec<Ref>) = Default::default();
    let mut tparts: Vec<(u32, u32, u32)> = Vec::new();
    let mut qbuf: Vec<Ref> = Vec::new();
    let mut mbuf: Vec<Ref> = Vec::new();
    let mut clock = super::trace::Clock::start();
    while let Some(Reverse((_, c))) = heap.pop() {
        let width = s.levels[c as usize].width() as u32;
        let mut cl = CLv::new(c, width);
        let here = lvs.len() as u32;
        let (l, r) = sh.kids[c as usize];
        let p = sh.parent[c as usize];
        let pe = if p == NONE { NONE } else { at[p as usize] };
        let twins = pe != NONE && may_twin(&s, &lvs[pe as usize], p, c, &mut qbuf, lim)?;
        // Debug builds search a screened level too, to check the screen.
        if twins || (cfg!(debug_assertions) && pe != NONE) {
            let pcl = &lvs[pe as usize];
            let plvl = &s.levels[p as usize];
            let (pl, pr) = sh.kids[p as usize];
            let left = pl == c;
            let sib = if left { pr } else { pl };
            let sib_literal = sh.literal_leaf(sib);
            // Every class of the parent with its pairs.
            let classes = || {
                (0..pcl.width)
                    .filter(|&i| pcl.group_of.is_empty() || pcl.group_of[i as usize] == NONE)
                    .map(|i| (i, plvl.node(i)))
                    .chain((0..pcl.groups()).map(|g| (pcl.width + g as u32, pcl.group(g))))
            };
            // Partners (One as both literals against a leaf sibling).
            let partners = |b: Ref| -> [Ref; 2] {
                if sib_literal && !is_tail(b) && ix(b) == ONE { [mk(sib, POS), mk(sib, NEG)] } else { [b, ZREF] }
            };
            epoch += 1;
            ent.clear();
            occ.clear();
            here_tails.clear();
            maybe_split.clear();
            splits.clear();
            // The stored nodes of `c` hold slots `0..width`, an entity once
            // a pair names it (a partner count above 0); the skips and tails
            // the pairs name follow.
            lim.reserve(&mut ent, width as usize)?;
            ent.extend((0..width).map(|i| (mk(c, i), 0, 0)));
            if width > 0 {
                ids.id_alloc(&s, mk(c, 0), lim)?;
            }
            let mut named = 0u64;
            for (k, pairs) in classes() {
                for &(a, b) in pairs {
                    let (e, partner) = if left { (a, b) } else { (b, a) };
                    let slot = if is_tail(e) {
                        let t = tail_id(e);
                        if tstamp[t] != epoch {
                            tstamp[t] = epoch;
                            tslot[t] = ent.len() as u32;
                            lim.try_push(&mut ent, (e, 0, 0))?;
                            lim.try_push(&mut here_tails, t as u32)?;
                            named += 1;
                        }
                        tslot[t]
                    } else if vt(e) == c {
                        ix(e)
                    } else {
                        let id = ids.id_alloc(&s, e, lim)?;
                        if ids.stamp[id].0 != epoch {
                            ids.stamp[id] = (epoch, ent.len() as u32);
                            lim.try_push(&mut ent, (e, 0, 0))?;
                            named += 1;
                        }
                        ids.stamp[id].1
                    };
                    let x = &mut ent[slot as usize];
                    named += (x.2 == 0 && vt(e) == c && !is_tail(e)) as u64;
                    for q in partners(partner) {
                        if q != ZREF {
                            x.1 = x.1.wrapping_add(mix(k, q));
                            x.2 += 1;
                        }
                    }
                }
            }
            // A tail's member that a pair names besides is split off: its
            // partners are its own and the tail's.
            if !here_tails.is_empty() {
                for (slot, x) in ent.iter().enumerate() {
                    if x.2 > 0 && !is_tail(x.0) && let Some((ar, pos)) = tails.of(ids.id(x.0)) {
                        lim.try_push(&mut maybe_split, (slot as u32, ar, pos))?;
                    }
                }
            }
            for &(slot, ar, pos) in &maybe_split {
                for &t in &here_tails {
                    let tl = &mut tails.list[t as usize];
                    if tl.arena == ar && (tl.lo..tl.hi).contains(&pos) && !tl.gone.contains(&pos) {
                        let (h, n) = (ent[tslot[t as usize] as usize].1, ent[tslot[t as usize] as usize].2);
                        let x = &mut ent[slot as usize];
                        x.1 = x.1.wrapping_add(h);
                        x.2 += n;
                        lim.try_push(&mut tl.gone, pos)?;
                        lim.try_push(&mut splits, (t, slot))?;
                        break;
                    }
                }
            }
            // Entities whose hashes agree, confirmed by their partner lists.
            cand.clear();
            lim.try_resize_amortized(&mut cand, ent.len(), false)?;
            let any = collide(&ent, named as usize, &mut cand, &mut link, &mut heads, &mut table, lim)?;
            // The twin groups, then each tail in none, as lists of entities.
            forms.clear();
            fstart.clear();
            fstart.push(0);
            if any {
                for (k, pairs) in classes() {
                    for &(a, b) in pairs {
                        let (e, partner) = if left { (a, b) } else { (b, a) };
                        let i = if is_tail(e) {
                            tslot[tail_id(e)]
                        } else if vt(e) == c {
                            ix(e)
                        } else {
                            ids.stamp[ids.id(e)].1
                        };
                        for q in partners(partner) {
                            if q == ZREF {
                                continue;
                            }
                            if cand[i as usize] {
                                lim.try_push(&mut occ, (i, k, q))?;
                            }
                            if is_tail(e) {
                                for &(t, m) in &splits {
                                    if t as usize == tail_id(e) && cand[m as usize] {
                                        lim.try_push(&mut occ, (m, k, q))?;
                                    }
                                }
                            }
                        }
                    }
                }
                // Each candidate's partner list, sorted, at `ctx[i]` in
                // `plist`: placed by counting, each list then sorted alone.
                ctx.clear();
                lim.try_resize_amortized(&mut ctx, ent.len(), (0, 0))?;
                for x in &occ {
                    ctx[x.0 as usize].1 += 1;
                }
                let mut o = 0u32;
                for c in ctx.iter_mut() {
                    let n = c.1;
                    *c = (o, o);
                    o += n;
                }
                plist.clear();
                lim.try_resize_amortized(&mut plist, occ.len(), (0, 0))?;
                for &(i, k, q) in &occ {
                    let c = &mut ctx[i as usize];
                    plist[c.1 as usize] = (k, q);
                    c.1 += 1;
                }
                for &(a, b) in &ctx {
                    plist[a as usize..b as usize].sort_unstable();
                }
                let list = |i: u32| {
                    let (a, b) = ctx[i as usize];
                    &plist[a as usize..b as usize]
                };
                for &h in &heads {
                    // The run of entities whose hash and count are `h`'s.
                    order.clear();
                    let mut i = h;
                    while i != NONE {
                        lim.try_push(&mut order, i)?;
                        i = link[i as usize].0;
                    }
                    let run = &order[..];
                    // Split the run into groups of equal lists.
                    firsts.clear();
                    let mut of: Vec<u32> = Vec::with_capacity(run.len());
                    for &i in run {
                        let g = firsts.iter().position(|&f| list(f) == list(i));
                        of.push(match g {
                            Some(g) => g as u32,
                            None => {
                                lim.try_push(&mut firsts, i)?;
                                (firsts.len() - 1) as u32
                            }
                        });
                    }
                    for g in 0..firsts.len() as u32 {
                        let before = forms.len();
                        lim.reserve(&mut forms, run.len())?;
                        forms.extend(run.iter().zip(&of).filter(|x| *x.1 == g).map(|x| ent[*x.0 as usize].0));
                        if forms.len() - before < 2 {
                            forms.truncate(before);
                            continue;
                        }
                        lim.try_push(&mut fstart, forms.len() as u32)?;
                    }
                }
            }
            debug_assert!(twins || (forms.is_empty() && here_tails.is_empty()), "a level the screen passed has twins");
            for &e in &forms {
                if is_tail(e) {
                    tmark[tail_id(e)] = epoch;
                }
            }
            for &t in &here_tails {
                if tmark[t as usize] != epoch {
                    lim.try_push(&mut forms, tail_ref(t as usize))?;
                    lim.try_push(&mut fstart, forms.len() as u32)?;
                }
            }
            for f in 0..fstart.len() - 1 {
                let fm = &forms[fstart[f] as usize..fstart[f + 1] as usize];
                // The members: stored here, skips, and the tails' live
                // members by side (`tparts`), those stored here first.
                stored.clear();
                skips.clear();
                tparts.clear();
                let first = if sh.pre[l as usize] < sh.pre[r as usize] { l } else { r };
                for &e in fm {
                    if is_tail(e) {
                        let t = tail_id(e);
                        let tl = &tails.list[t];
                        let arena = &tails.arenas[tl.arena as usize];
                        let mut i = tl.lo;
                        while i < tl.hi && vt(arena[i as usize]) == c {
                            if !tl.gone.contains(&i) {
                                lim.try_push(&mut stored, arena[i as usize])?;
                            }
                            i += 1;
                        }
                        let mid = i + arena[i as usize..tl.hi as usize].partition_point(|&m| sh.within(vt(m), first)) as u32;
                        for (lo, hi) in [(i, mid), (mid, tl.hi)] {
                            if lo < hi {
                                lim.try_push(&mut tparts, (t as u32, lo, hi))?;
                            }
                        }
                    } else if vt(e) == c {
                        lim.try_push(&mut stored, e)?;
                    } else {
                        lim.try_push(&mut skips, e)?;
                    }
                }
                let total = stored.len() + skips.len() + tparts.iter().map(|&x| tails.live(x)).sum::<usize>();
                if total < 2 {
                    // One member is the class: a tail is it.
                    let one = stored.first().or(skips.first()).copied().or_else(|| {
                        tparts.iter().find_map(|&(t, lo, hi)| {
                            let tl = &tails.list[t as usize];
                            (lo..hi).find(|i| !tl.gone.contains(i)).map(|i| tails.arenas[tl.arena as usize][i as usize])
                        })
                    });
                    if let Some(m) = one {
                        for &e in fm {
                            if is_tail(e) {
                                lim.try_push(&mut tails.list[tail_id(e)].res, Res::Via(m))?;
                            }
                        }
                    }
                    continue;
                }
                merged = true;
                if stored.is_empty() {
                    // Skips alone stay twins down to where they part.
                    members.clear();
                    lim.reserve(&mut members, total)?;
                    members.extend_from_slice(&skips);
                    for &(t, lo, hi) in &tparts {
                        let tl = &tails.list[t as usize];
                        let arena = &tails.arenas[tl.arena as usize];
                        members.extend((lo..hi).filter(|i| !tl.gone.contains(i)).map(|i| arena[i as usize]));
                    }
                    for &e in fm {
                        if is_tail(e) {
                            lim.try_push(&mut tails.list[tail_id(e)].res, Res::Via(members[0]))?;
                        }
                    }
                    let v = parts(&s, &mut names, c, members.iter().copied(), lim)?;
                    if v != c {
                        let mut group = Vec::new();
                        lim.reserve_exact(&mut group, members.len())?;
                        group.extend_from_slice(&members);
                        pending.entry(v).or_default().push((here, group));
                        let mut u = v;
                        while u != c && !queued[u as usize] {
                            queued[u as usize] = true;
                            heap.push(Reverse((sh.pre[u as usize], u)));
                            u = sh.parent[u as usize];
                        }
                        continue;
                    }
                    let g = cl.push_group(&s, &members, l, r, lim)?;
                    cl.joins = true;
                    for &e in &members {
                        ids.join(&s, e, here, here, g, lim)?;
                    }
                    continue;
                }
                // Stored members with the skips beside them: two or more
                // skips down one inner side go as a tail.
                let g = cl.open(lim)?;
                for &e in &stored {
                    if cl.group_of.is_empty() {
                        cl.group_of = filled(lim, cl.width as usize, NONE)?;
                    }
                    cl.group_of[ix(e) as usize] = g;
                    let node = s.levels[c as usize].node(ix(e));
                    lim.reserve(&mut cl.gpairs, node.len())?;
                    cl.gpairs.extend_from_slice(node);
                }
                for side in [l, r] {
                    let pair = |x: Ref| if side == l { (x, mk(r, ONE)) } else { (mk(l, ONE), x) };
                    let sk = skips.iter().filter(|&&e| sh.within(vt(e), side)).count();
                    let sp = tparts
                        .iter()
                        .filter(|&&(t, lo, _)| sh.within(vt(tails.arenas[tails.list[t as usize].arena as usize][lo as usize]), side));
                    let count = sk + sp.clone().map(|&x| tails.live(x)).sum::<usize>();
                    if count == 0 {
                        continue;
                    }
                    if count < 2 || sh.is_leaf(side) || sh.weighted[c as usize] {
                        for &e in skips.iter().filter(|&&e| sh.within(vt(e), side)) {
                            lim.try_push(&mut cl.gpairs, pair(e))?;
                        }
                        for &(t, lo, hi) in sp {
                            let tl = &tails.list[t as usize];
                            for i in (lo..hi).filter(|i| !tl.gone.contains(i)) {
                                lim.try_push(&mut cl.gpairs, pair(tails.arenas[tl.arena as usize][i as usize]))?;
                            }
                        }
                        continue;
                    }
                    let sp: Vec<(u32, u32, u32)> = sp.copied().collect();
                    let nt = if sk == 0 && sp.len() == 1 {
                        let (t, lo, hi) = sp[0];
                        let tl = &tails.list[t as usize];
                        let ar = tl.arena;
                        let gone: Vec<u32> = tl.gone.iter().copied().filter(|p| (lo..hi).contains(p)).collect();
                        tails.push(ar, lo, hi, gone, lim)?
                    } else {
                        // By preorder: the skips sorted, and each tail's
                        // part, sorted already, merged in.
                        let key = |m: Ref| ((sh.pre[vt(m) as usize] as u64) << 32) | ix(m) as u64;
                        let mut a: Vec<Ref> = Vec::new();
                        lim.reserve_exact(&mut a, count)?;
                        a.extend(skips.iter().copied().filter(|&e| sh.within(vt(e), side)));
                        a.sort_unstable_by_key(|&m| key(m));
                        for &(t, lo, hi) in &sp {
                            let tl = &tails.list[t as usize];
                            let arena = &tails.arenas[tl.arena as usize];
                            let mid = a.len();
                            a.extend((lo..hi).filter(|i| !tl.gone.contains(i)).map(|i| arena[i as usize]));
                            merge_sorted(&mut a, mid, key, &mut mbuf, lim)?;
                        }
                        let ar = tails.arenas.len() as u32;
                        lim.try_resize_amortized(&mut tails.of, ids.n as usize, (NONE, NONE))?;
                        for (i, &m) in a.iter().enumerate() {
                            tails.of[ids.id(m)] = (ar, i as u32);
                        }
                        let n = a.len() as u32;
                        tails.arenas.push(a);
                        tails.push(ar, 0, n, Vec::new(), lim)?
                    };
                    lim.try_resize_amortized(&mut tstamp, tails.list.len(), 0)?;
                    lim.try_resize_amortized(&mut tslot, tails.list.len(), 0)?;
                    lim.try_resize_amortized(&mut tmark, tails.list.len(), 0)?;
                    lim.try_push(&mut cl.gpairs, pair(tail_ref(nt)))?;
                }
                lim.try_push(&mut cl.gstart, cl.gpairs.len() as u32)?;
                lim.try_push(&mut cl.rep, stored[0])?;
                cl.joins |= !skips.is_empty();
                for &e in &skips {
                    ids.join(&s, e, here, here, g, lim)?;
                }
                for &e in fm {
                    if is_tail(e) {
                        lim.try_push(&mut tails.list[tail_id(e)].res, Res::Group(here, g))?;
                    }
                }
            }
            for &(t, slot) in &splits {
                lim.try_push(&mut tails.list[t as usize].res, Res::Via(ent[slot as usize].0))?;
            }
            lim.charge_work(named + occ.len() as u64);
        }
        // Merged skips sent down here: each group is a class of its own,
        // since only the levels above name its members.
        if let Some(groups) = pending.remove(&c) {
            for (from, group) in groups {
                let g = cl.push_group(&s, &group, l, r, lim)?;
                lvs[from as usize].joins = true;
                for &e in &group {
                    ids.join(&s, e, from, here, g, lim)?;
                }
                lim.discard(group);
            }
        }
        lim.charge_work(s.levels[c as usize].pairs.len() as u64);
        if width > 0 || cl.groups() > 0 {
            let descend = scope.is_none_or(|cone| cone[c as usize] || cl.groups() > 0);
            if descend {
                for ch in [l, r] {
                    if !sh.is_leaf(ch) && !queued[ch as usize] {
                        queued[ch as usize] = true;
                        heap.push(Reverse((sh.pre[ch as usize], ch)));
                    }
                }
            }
        }
        at[c as usize] = here;
        lvs.push(cl);
    }
    lim.check_stop()?;
    clock.lap(8);
    // Patch from the leaves up. A side read at `d` resolves through a
    // group a skip joined there, else through its node's result, a group's
    // only where its member is stored at `d`. `changed[v]` is the visited
    // level of `v` once a node there resolves to another reference, and
    // `dirty[v]` says whether one at or below `v` does.
    let mut changed = vec![NONE; n];
    let mut dirty = vec![false; n];
    let mut buf: Vec<(Ref, Ref)> = Vec::new();
    for e in (0..lvs.len()).rev() {
        let c = lvs[e].level;
        let (l, r) = sh.kids[c as usize];
        let width = lvs[e].width as usize;
        // A side changes only through a join where it is read or a node
        // that changed below: on a side where neither can, nothing is
        // resolved.
        let still = |d: u32| !dirty[d as usize] && (sh.is_leaf(d) || at[d as usize] == NONE || !lvs[at[d as usize] as usize].joins);
        let (ml, mr) = (!still(l), !still(r));
        if lvs[e].groups() == 0 && !ml && !mr {
            continue;
        }
        let (sl, sr) = (Side::new(sh, &at, &lvs, l), Side::new(sh, &at, &lvs, r));
        let lvl = &s.levels[c as usize];
        let sum = sh.weighted[c as usize];
        let grouped = |i: usize| !lvs[e].group_of.is_empty() && lvs[e].group_of[i] != NONE;
        let changes = |lvs: &[CLv], changed: &[u32], i: usize| {
            lvl.node(i as u32).iter().any(|&(a, b)| (ml && resolve(lvs, &ids, changed, sl, a) != a) || (mr && resolve(lvs, &ids, changed, sr, b) != b))
        };
        let rewrite = |i: usize| !grouped(i) && changes(&lvs, &changed, i);
        let first = (0..width).find(|&i| rewrite(i));
        if lvs[e].groups() == 0 && first.is_none() {
            continue;
        }
        // Stored nodes keep their numbers; those merged into a group, or
        // that now resolve to another reference, stay behind unreachable.
        // Runs of nodes kept as they were are copied whole.
        let mut out = SLevel::default();
        lim.reserve(&mut out.start, width + 1 + lvs[e].groups())?;
        lim.reserve(&mut out.pairs, lvl.pairs.len() + lvs[e].gpairs.len())?;
        out.start.push(0);
        let mut rm = filled(lim, width, KEEP)?;
        let mut run = 0;
        let mut i = first.unwrap_or(width);
        while i < width {
            if Some(i) == first || rewrite(i) {
                let node = lvl.node(i as u32);
                buf.clear();
                lim.reserve(&mut buf, node.len())?;
                buf.extend(
                    node.iter()
                        .filter(|&&(a, b)| !sum || !(absorbed(sh, &lvs, &ids, &at, sl, a) || absorbed(sh, &lvs, &ids, &at, sr, b)))
                        .map(|&(a, b)| (resolve(&lvs, &ids, &changed, sl, a), resolve(&lvs, &ids, &changed, sr, b))),
                );
                match made(sh, c, &mut buf) {
                    Err(t) => {
                        rm[i] = t;
                        merged = true;
                    }
                    Ok(()) => {
                        out.copy_run(lvl, run, i, lim)?;
                        out.push(&buf, lim)?;
                        run = i + 1;
                    }
                }
            }
            i += 1;
        }
        out.copy_run(lvl, run, width, lim)?;
        let mut gres = Vec::new();
        lim.reserve_exact(&mut gres, lvs[e].groups())?;
        for g in 0..lvs[e].groups() {
            buf.clear();
            let group = lvs[e].group(g);
            lim.reserve(&mut buf, 2 * group.len())?;
            for &(a, b) in group {
                if sum && (absorbed(sh, &lvs, &ids, &at, sl, a) || absorbed(sh, &lvs, &ids, &at, sr, b)) {
                    continue;
                }
                if !(is_tail(a) || is_tail(b)) {
                    lim.try_push(&mut buf, (resolve(&lvs, &ids, &changed, sl, a), resolve(&lvs, &ids, &changed, sr, b)))?;
                    continue;
                }
                // A tail is the classes its members joined where it was read.
                let (t, side) = if is_tail(a) { (tail_id(a), sl) } else { (tail_id(b), sr) };
                for item in &tails.list[t].res {
                    let x = match *item {
                        Res::Group(to, g) => lvs[to as usize].gres[g as usize],
                        Res::Via(m) => resolve(&lvs, &ids, &changed, side, m),
                    };
                    lim.try_push(&mut buf, if is_tail(a) { (x, resolve(&lvs, &ids, &changed, sr, b)) } else { (resolve(&lvs, &ids, &changed, sl, a), x) })?;
                }
            }
            // Merged members can repeat a pair, which a sum keeps.
            buf.sort_unstable();
            if !sum {
                buf.dedup();
            }
            gres.push(match made(sh, c, &mut buf) {
                Err(t) => t,
                Ok(()) => mk(c, out.push(&buf, lim)?),
            });
        }
        for (x, &g) in rm.iter_mut().zip(&lvs[e].group_of) {
            if g != NONE {
                *x = gres[g as usize];
            }
        }
        if rm.iter().any(|&x| x != KEEP) {
            changed[c as usize] = e as u32;
            let mut u = c;
            while u != NONE && !dirty[u as usize] {
                dirty[u as usize] = true;
                u = sh.parent[u as usize];
            }
        }
        if out.pairs.is_empty() && out.start.len() == 1 {
            out.start.clear();
        }
        lvs[e].rm = rm;
        lvs[e].gres = gres;
        lim.release_bytes(s.levels[c as usize].bytes());
        s.levels.set(c as usize, out);
    }
    s.out = resolve(&lvs, &ids, &changed, Side::new(sh, &at, &lvs, sh.root), s.out);
    clock.lap(9);
    lim.discard(qbuf);
    lim.discard(mbuf);
    for v in [order, firsts, table] {
        lim.discard(v);
    }
    lim.discard(cand);
    lim.discard(link);
    lim.discard(heads);
    lim.discard(ent);
    lim.discard(occ);
    lim.discard(ctx);
    lim.discard(plist);
    lim.discard(members);
    lim.discard(buf);
    tails.discard(lim);
    ids.discard(lim);
    for v in [tstamp, tslot, tmark, here_tails, fstart] {
        lim.discard(v);
    }
    lim.discard(maybe_split);
    lim.discard(splits);
    lim.discard(tparts);
    for v in [forms, stored, skips] {
        lim.discard(v);
    }
    names.discard(lim);
    let visited: Vec<u32> = lvs.iter().map(|l| l.level).collect();
    for cl in lvs {
        cl.discard(lim);
    }
    if merged {
        s.collect_in(visited, lim)?;
    } else {
        s.active.retain(|&t| s.levels[t as usize].width() > 0);
    }
    clock.lap(10);
    Ok(s)
}

/// `a[..mid]` and `a[mid..]`, each sorted by `key`, merged in place
/// through `buf`, which takes a copy of the first.
pub(super) fn merge_sorted(a: &mut [Ref], mid: usize, key: impl Fn(Ref) -> u64, buf: &mut Vec<Ref>, lim: &Limits) -> Result<(), OperationError> {
    if mid == 0 || mid == a.len() || key(a[mid - 1]) <= key(a[mid]) {
        return Ok(());
    }
    buf.clear();
    lim.reserve(buf, mid)?;
    buf.extend_from_slice(&a[..mid]);
    // The write never passes the next element of the second run.
    let (mut i, mut j, mut w) = (0, mid, 0);
    while i < buf.len() && j < a.len() {
        if key(a[j]) < key(buf[i]) {
            a[w] = a[j];
            j += 1;
        } else {
            a[w] = buf[i];
            i += 1;
        }
        w += 1;
    }
    a[w..w + buf.len() - i].copy_from_slice(&buf[i..]);
    Ok(())
}

/// A child `d` of a level being patched, as its sides read there resolve:
/// its visited level (`NONE` for a leaf or a level not visited) and the
/// visited level whose joins hold the skips read at `d` (`NONE` where none
/// joined), each read once per level.
#[derive(Clone, Copy)]
struct Side {
    d: u32,
    vis: u32,
    jat: u32,
}

impl Side {
    fn new(sh: &Shape, at: &[u32], lvs: &[CLv], d: u32) -> Side {
        let vis = if sh.is_leaf(d) { NONE } else { at[d as usize] };
        let jat = if vis != NONE && lvs[vis as usize].joins { vis } else { NONE };
        Side { d, vis, jat }
    }
}

/// The group a skip read at `side` joined there.
#[inline]
fn joined(ids: &Ids, side: Side, x: Ref) -> Option<(u32, u32)> {
    if side.jat == NONE || vt(x) == side.d || is_tail(x) {
        return None;
    }
    ids.joined(side.jat, x)
}

/// What `x`, read at `side`, resolves to: through a group a skip joined
/// there, else through its node's result, a group's only where its member
/// is stored at `side`'s level.
#[inline]
fn resolve(lvs: &[CLv], ids: &Ids, changed: &[u32], side: Side, x: Ref) -> Ref {
    if let Some((to, g)) = joined(ids, side, x) {
        return lvs[to as usize].gres[g as usize];
    }
    let v = vt(x);
    let e = changed[v as usize];
    if e != NONE {
        let cl = &lvs[e as usize];
        let foreign = v != side.d && !cl.group_of.is_empty() && cl.group_of[ix(x) as usize] != NONE;
        if !foreign && cl.rm[ix(x) as usize] != KEEP {
            return cl.rm[ix(x) as usize];
        }
    }
    x
}

/// Whether `x`, read at `side`, joined a group there whose pairs another
/// member stands for.
#[inline]
fn absorbed(sh: &Shape, lvs: &[CLv], ids: &Ids, at: &[u32], side: Side, x: Ref) -> bool {
    if let Some((to, g)) = joined(ids, side, x) {
        return lvs[to as usize].rep[g as usize] != x;
    }
    let d = side.d;
    if is_tail(x) || sh.is_leaf(d) || vt(x) != d || at[d as usize] == NONE {
        return false;
    }
    let cl = &lvs[at[d as usize] as usize];
    !cl.group_of.is_empty() && cl.group_of[ix(x) as usize] != NONE && cl.rep[cl.group_of[ix(x) as usize] as usize] != x
}

/// Whether an entity of `c`, the child of `p` whose classes `pcl` holds, can
/// have a twin: two entities' partner lists agree only where a class pairs
/// both with one partner (`One` meets either literal against a literal
/// sibling), and a tail read on `c`'s side brings its members' partners.
fn may_twin(s: &SkipForm, pcl: &CLv, p: u32, c: u32, buf: &mut Vec<Ref>, lim: &Limits) -> Result<bool, OperationError> {
    let sh = &*s.shape;
    let plvl = &s.levels[p as usize];
    let (pl, pr) = sh.kids[p as usize];
    let left = pl == c;
    let sib = if left { pr } else { pl };
    let literal = sh.literal_leaf(sib);
    let mut class = |pairs: &[(Ref, Ref)]| -> Result<bool, OperationError> {
        if literal {
            let mut m = 0u8;
            for &(a, b) in pairs {
                let (e, q) = if left { (a, b) } else { (b, a) };
                let cells = literal_cells(ix(q));
                if is_tail(e) || m & cells != 0 {
                    return Ok(true);
                }
                m |= cells;
            }
            return Ok(false);
        }
        if pairs.iter().any(|&(a, b)| is_tail(if left { a } else { b })) {
            return Ok(true);
        }
        let q = |&(a, b): &(Ref, Ref)| if left { b } else { a };
        if pairs.len() <= 8 {
            return Ok(pairs.iter().enumerate().any(|(i, x)| pairs[..i].iter().any(|y| q(x) == q(y))));
        }
        buf.clear();
        lim.reserve(buf, pairs.len())?;
        buf.extend(pairs.iter().map(q));
        buf.sort_unstable();
        Ok(buf.windows(2).any(|w| w[0] == w[1]))
    };
    for i in 0..pcl.width {
        if (pcl.group_of.is_empty() || pcl.group_of[i as usize] == NONE) && class(plvl.node(i))? {
            return Ok(true);
        }
    }
    for g in 0..pcl.groups() {
        if class(pcl.group(g))? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Entities whose partner hash and count agree with another's, marked in
/// `cand` and chained by hash and count, no sort: `heads` lists each
/// chain's first entity, and `link[i].0` is the entity after `i` in its
/// chain (`NONE` at the end), in index order; `link[h].1` is the last of
/// head `h`'s. An open-addressing table of slots by the hash's top bits,
/// sized to the `named` entities, holds each chain's head. An entry with no
/// partner is no entity. Whether there is any.
fn collide(
    ent: &[(Ref, u64, u32)],
    named: usize,
    cand: &mut [bool],
    link: &mut Vec<(u32, u32)>,
    heads: &mut Vec<u32>,
    table: &mut Vec<u32>,
    lim: &Limits,
) -> Result<bool, OperationError> {
    let bits = (2 * named).next_power_of_two().trailing_zeros().max(4);
    let mask = (1usize << bits) - 1;
    table.clear();
    lim.try_resize_amortized(table, mask + 1, NONE)?;
    link.clear();
    lim.try_resize_amortized(link, ent.len(), (NONE, NONE))?;
    heads.clear();
    for (i, x) in ent.iter().enumerate() {
        if x.2 == 0 {
            continue;
        }
        let mut p = (x.1 >> (64 - bits)) as usize;
        loop {
            let j = table[p];
            if j == NONE {
                table[p] = i as u32;
                break;
            }
            let y = &ent[j as usize];
            if (y.1, y.2) == (x.1, x.2) {
                if !cand[j as usize] {
                    cand[j as usize] = true;
                    link[j as usize].1 = j;
                    lim.try_push(heads, j)?;
                }
                cand[i] = true;
                let last = link[j as usize].1;
                link[last as usize].0 = i as u32;
                link[j as usize].1 = i as u32;
                break;
            }
            p = (p + 1) & mask;
        }
    }
    Ok(!heads.is_empty())
}

/// Dense numbers for a diagram's references: stored node `i` of level `t`
/// is `base[t] + i`, leaf label `k` of leaf `c` is `base[c] + k`, numbered
/// the first time a reference of `t` is an entity. Per number: the epoch it
/// was last an entity in with its slot there, and the skips' joins.
struct Ids {
    base: Vec<u32>,
    n: u32,
    stamp: Vec<(u32, u32)>,
    /// Per number, its latest join, read with one lookup: the visited level
    /// the skip was read at (`NONE` where it joined none), the visited level
    /// and group it joined, and its join before that in `joins` (`NONE`
    /// where none).
    jhead: Vec<[u32; 4]>,
    /// The joins before each number's latest, in the same form.
    joins: Vec<[u32; 4]>,
}

impl Ids {
    fn new(num_nodes: usize) -> Ids {
        Ids { base: vec![NONE; num_nodes], n: 0, stamp: Vec::new(), jhead: Vec::new(), joins: Vec::new() }
    }
    #[inline]
    fn id_alloc(&mut self, s: &SkipForm, x: Ref, lim: &Limits) -> Result<usize, OperationError> {
        let v = vt(x) as usize;
        let mut b = self.base[v];
        if b == NONE {
            b = self.n;
            self.base[v] = b;
            let add = if s.shape.is_leaf(v as u32) { s.shape.labels[v] } else { s.levels[v].width() as u32 };
            self.n += add;
        }
        let id = (b + ix(x)) as usize;
        if id >= self.stamp.len() {
            lim.try_resize_amortized(&mut self.stamp, self.n as usize, (0, 0))?;
        }
        Ok(id)
    }
    #[inline]
    fn id(&self, x: Ref) -> usize {
        (self.base[vt(x) as usize] + ix(x)) as usize
    }

    /// Note that skip `x`, read at visited level `at`, joined group `g` of
    /// visited level `to`.
    fn join(&mut self, s: &SkipForm, x: Ref, at: u32, to: u32, g: u32, lim: &Limits) -> Result<(), OperationError> {
        let id = self.id_alloc(s, x, lim)?;
        if id >= self.jhead.len() {
            lim.try_resize_amortized(&mut self.jhead, self.n as usize, [NONE; 4])?;
        }
        let mut next = NONE;
        if self.jhead[id][0] != NONE {
            lim.try_push(&mut self.joins, self.jhead[id])?;
            next = (self.joins.len() - 1) as u32;
        }
        self.jhead[id] = [at, to, g, next];
        Ok(())
    }

    /// The group skip `x` joined where it was read at visited level `at`.
    #[inline]
    fn joined(&self, at: u32, x: Ref) -> Option<(u32, u32)> {
        let b = self.base[vt(x) as usize];
        if b == NONE {
            return None;
        }
        let mut j = *self.jhead.get((b + ix(x)) as usize)?;
        loop {
            let [a, to, g, next] = j;
            if a == at {
                return Some((to, g));
            }
            if next == NONE {
                return None;
            }
            j = self.joins[next as usize];
        }
    }

    fn discard(self, lim: &Limits) {
        lim.discard(self.stamp);
        lim.discard(self.jhead);
        lim.discard(self.joins);
    }
}

/// Which levels name a reference, read from the diagram as it was before
/// the patch. Lookups first walk down from the merge level, reading each
/// level's sorted references, built the first time a walk reaches it; once
/// the walks' steps and the sorting of the lists they built (two steps a
/// pair) pass a quarter of the diagram's pairs, the levels naming each
/// reference are listed for every reference at once and looked up
/// instead: by counting, the levels whose skips name it, above its parent
/// level, and whether its parent level names it. Nothing is built before
/// the first lookup.
#[derive(Default)]
struct Names {
    at: Vec<u32>,
    lists: Vec<Vec<Ref>>,
    /// The levels that hold nodes, by preorder position.
    held: Vec<(u32, u32)>,
    steps: usize,
    /// The held levels' pairs, and a quarter of them.
    pairs: usize,
    budget: usize,
    /// Once the budget is spent: every reference numbered densely (a
    /// level's or leaf's first number), per number the levels whose skips
    /// name it, by preorder (`start` delimits them), and whether its
    /// parent level names it.
    base: Vec<u32>,
    start: Vec<u32>,
    levels: Vec<u32>,
    direct: Vec<bool>,
}

impl Names {
    fn ready(&mut self, s: &SkipForm) {
        if !self.at.is_empty() {
            return;
        }
        let sh = &*s.shape;
        self.held = s.active.iter().filter(|&&t| s.levels[t as usize].width() > 0).map(|&t| (sh.pre[t as usize], t)).collect();
        self.held.sort_unstable();
        self.pairs = self.held.iter().map(|&(_, t)| s.levels[t as usize].pairs.len()).sum();
        self.budget = self.pairs / 4;
        self.at = vec![NONE; sh.num_nodes()];
    }

    /// The highest level at or below `c` whose pairs name `e`, from the
    /// lists of every reference.
    fn indexed(&mut self, s: &SkipForm, c: u32, e: Ref, lim: &Limits) -> Result<Option<u32>, OperationError> {
        let sh = &*s.shape;
        if self.base.is_empty() {
            let mut base = filled(lim, sh.num_nodes(), 0u32)?;
            let mut n = 0u32;
            for (v, b) in base.iter_mut().enumerate() {
                *b = n;
                n += if sh.is_leaf(v as u32) { sh.labels[v] } else { s.levels[v].width() as u32 };
            }
            let id = |x: Ref| (base[vt(x) as usize] + ix(x)) as usize;
            let mut start = filled(lim, n as usize + 1, 0u32)?;
            let mut direct = filled(lim, n as usize, false)?;
            for &(_, t) in &self.held {
                let (l, r) = sh.kids[t as usize];
                for &(a, b) in &s.levels[t as usize].pairs {
                    for (d, x) in [(l, a), (r, b)] {
                        if vt(x) == d {
                            direct[id(x)] = true;
                        } else {
                            start[id(x) + 1] += 1;
                        }
                    }
                }
            }
            for i in 0..n as usize {
                start[i + 1] += start[i];
            }
            let mut fill = filled(lim, n as usize, 0u32)?;
            fill.copy_from_slice(&start[..n as usize]);
            let mut levels = filled(lim, start[n as usize] as usize, 0u32)?;
            // `held` is in preorder, so each list is too.
            for &(_, t) in &self.held {
                let (l, r) = sh.kids[t as usize];
                for &(a, b) in &s.levels[t as usize].pairs {
                    for (d, x) in [(l, a), (r, b)] {
                        if vt(x) != d {
                            let i = id(x);
                            levels[fill[i] as usize] = t;
                            fill[i] += 1;
                        }
                    }
                }
            }
            lim.discard(fill);
            (self.base, self.start, self.levels, self.direct) = (base, start, levels, direct);
        }
        // Every level naming `e` is on the path to it, the skips' above its
        // parent level: the first at or below `c` by preorder.
        let i = (self.base[vt(e) as usize] + ix(e)) as usize;
        let ls = &self.levels[self.start[i] as usize..self.start[i + 1] as usize];
        let k = ls.partition_point(|&u| sh.pre[u as usize] < sh.pre[c as usize]);
        let p = sh.parent[vt(e) as usize];
        Ok(ls.get(k).copied().or_else(|| (self.direct[i] && sh.pre[p as usize] >= sh.pre[c as usize]).then_some(p)))
    }

    /// The highest level from `c` down to above `e`'s level whose pairs
    /// name `e`, among those whose child toward `e` is above `w` (by
    /// preorder): walked down the vtree, or over the levels that hold nodes
    /// in between, whichever is shorter.
    fn highest(&mut self, s: &SkipForm, c: u32, e: Ref, w: u32, lim: &Limits) -> Result<Option<u32>, OperationError> {
        self.ready(s);
        let sh = &*s.shape;
        let v = vt(e);
        if self.steps > self.budget {
            // On the path to `e` (an ancestor of its level at or below `c`),
            // so its child toward `e` is compared with `w` as a walk's is.
            return Ok(self.indexed(s, c, e, lim)?.filter(|&u| sh.pre[sh.toward(u, v) as usize] < sh.pre[w as usize]));
        }
        let lo = self.held.partition_point(|&(q, _)| q < sh.pre[c as usize]);
        let hi = self.held.partition_point(|&(q, _)| q < sh.pre[v as usize]);
        let path = (sh.depth[v as usize] - sh.depth[c as usize]) as usize;
        if path <= hi - lo {
            let mut u = c;
            while u != v {
                self.steps += 1;
                let x = sh.toward(u, v);
                if sh.pre[x as usize] >= sh.pre[w as usize] {
                    break;
                }
                if s.levels[u as usize].width() > 0 && self.names(s, u, e, lim)? {
                    return Ok(Some(u));
                }
                u = x;
            }
        } else {
            for i in lo..hi {
                self.steps += 1;
                let u = self.held[i].1;
                if !sh.within(v, u) {
                    continue;
                }
                if sh.pre[sh.toward(u, v) as usize] >= sh.pre[w as usize] {
                    break;
                }
                if self.names(s, u, e, lim)? {
                    return Ok(Some(u));
                }
            }
        }
        Ok(None)
    }

    /// Whether a pair of level `u` of `s` names `e`.
    fn names(&mut self, s: &SkipForm, u: u32, e: Ref, lim: &Limits) -> Result<bool, OperationError> {
        if self.at[u as usize] == NONE {
            let pairs = &s.levels[u as usize].pairs;
            self.steps += 2 * pairs.len();
            let mut v: Vec<Ref> = Vec::new();
            lim.reserve_exact(&mut v, 2 * pairs.len())?;
            v.extend(pairs.iter().flat_map(|&(a, b)| [a, b]));
            v.sort_unstable();
            v.dedup();
            self.at[u as usize] = self.lists.len() as u32;
            lim.try_push(&mut self.lists, v)?;
        }
        Ok(self.lists[self.at[u as usize] as usize].binary_search(&e).is_ok())
    }

    fn discard(self, lim: &Limits) {
        for v in self.lists {
            lim.discard(v);
        }
        lim.discard(self.base);
        lim.discard(self.start);
        lim.discard(self.levels);
        lim.discard(self.direct);
    }
}

/// The level whose class a group of skips that merge at `c` is: the parent
/// of the highest level below `c` where the members stop being twins (one
/// is stored there, a pair of the level above names one, or they part).
/// `c` itself when that level is `c`'s child.
fn parts(s: &SkipForm, nm: &mut Names, c: u32, members: impl Iterator<Item = Ref> + Clone, lim: &Limits) -> Result<u32, OperationError> {
    let sh = &*s.shape;
    // The highest member, and where the members part.
    let mut h = NONE;
    for e in members.clone() {
        if h == NONE || sh.pre[vt(e) as usize] < sh.pre[h as usize] {
            h = vt(e);
        }
    }
    let mut w = if members.clone().all(|e| sh.within(vt(e), h)) { h } else { sh.parent[h as usize] };
    for e in members {
        // The highest level at or below `c` naming `e`, as far as it could
        // still raise `w`.
        if let Some(u) = nm.highest(s, c, e, w, lim)? {
            w = sh.toward(u, vt(e));
        }
    }
    // Members on both leaves under `c` part at `c` itself.
    if w == c {
        return Ok(c);
    }
    debug_assert!(sh.within(w, c));
    Ok(sh.parent[w as usize])
}
