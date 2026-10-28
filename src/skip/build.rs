//! Breadth-first construction of a skip-form result.
//!
//! A [`Rule`] names each result node by a key. The construction finds every
//! key the output needs, level by level from the root (a key's children land
//! strictly below it, so preorder meets each level once all its keys are
//! in), then builds the nodes from the leaves up: a pair with a false side is
//! dropped, a node with no pairs is false, and a node that is `true` on a
//! single-variable side resolves to the reference it skips to.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::hash::Hash;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::{mk, normalize, Ref, Shape, SkipForm, ZREF};
use crate::limits::Limits;
use crate::OperationError;

/// Where a key's node is.
pub(crate) enum Land {
    /// Known without building: a reference into the result (a leaf label or
    /// false).
    Done(Ref),
    /// Built at this internal level.
    At(u32),
}

pub(crate) trait Rule {
    type Key: Copy + Eq + Hash;
    fn land(&mut self, key: Self::Key) -> Land;
    /// The pairs of the node of `key`, landed at `u`, as keys of their sides.
    fn expand(&mut self, u: u32, key: Self::Key, out: &mut Vec<(Self::Key, Self::Key)>);
}

/// A side's handle while the construction runs: a finished reference (top
/// bit clear, or [`ZREF`]) or a pending key, `PENDING | level << 32 | slot`.
const PENDING: u64 = 1 << 63;

const NONE: u32 = u32::MAX;

/// One level's keys while the construction runs.
pub(crate) struct Lv<K> {
    pub(crate) level: u32,
    map: FxHashMap<K, u32>,
    pub(crate) keys: Vec<K>,
    /// Key `s`'s pairs are `combos[cstart[s]..cstart[s + 1]]`.
    pub(crate) cstart: Vec<u32>,
    pub(crate) combos: Vec<(u64, u64)>,
    pub(crate) res: Vec<Ref>,
}

/// The levels keys land on, visited parents first: a level is taken in
/// preorder once every level above it is done, which is when all its keys
/// are in (a key's pairs land strictly below it).
pub(crate) struct Levels<K> {
    /// Per vtree node, its entry in `lv`.
    pub(crate) at: Vec<u32>,
    pub(crate) lv: Vec<Lv<K>>,
    pub(crate) heap: BinaryHeap<Reverse<(u32, u32)>>,
}

impl<K: Copy + Eq + Hash> Levels<K> {
    pub(crate) fn new(n: usize) -> Self {
        Levels { at: vec![NONE; n], lv: Vec::new(), heap: BinaryHeap::new() }
    }

    /// The handle of key `k` at level `u`.
    #[inline]
    pub(crate) fn insert(&mut self, sh: &Shape, u: u32, k: K) -> u64 {
        PENDING | (u as u64) << 32 | self.slot(sh, u, k) as u64
    }

    /// The slot of key `k` at level `u`.
    #[inline]
    pub(crate) fn slot(&mut self, sh: &Shape, u: u32, k: K) -> u32 {
        let mut e = self.at[u as usize];
        if e == NONE {
            e = self.lv.len() as u32;
            self.at[u as usize] = e;
            self.lv.push(Lv { level: u, map: FxHashMap::default(), keys: Vec::new(), cstart: Vec::new(), combos: Vec::new(), res: Vec::new() });
            self.heap.push(Reverse((sh.pre[u as usize], u)));
        }
        let lv = &mut self.lv[e as usize];
        let next = lv.keys.len() as u32;
        *lv.map.entry(k).or_insert_with(|| {
            lv.keys.push(k);
            next
        })
    }

    /// The finished reference of key slot `slot` at level `u`.
    #[inline]
    pub(crate) fn res(&self, u: u32, slot: u32) -> Ref {
        self.lv[self.at[u as usize] as usize].res[slot as usize]
    }

    /// The finished reference of handle `h`.
    #[inline]
    pub(crate) fn get(&self, h: u64) -> Ref {
        if h != ZREF && h & PENDING != 0 {
            self.lv[self.at[((h >> 32) & 0x7fff_ffff) as usize] as usize].res[h as u32 as usize]
        } else {
            h
        }
    }
}

pub(crate) fn construct<R: Rule>(
    shape: &Arc<Shape>,
    rule: &mut R,
    root: R::Key,
    lim: &Limits,
) -> Result<SkipForm, OperationError> {
    let sh = &**shape;
    let mut form = SkipForm::zero(Arc::clone(shape));
    let mut lv: Levels<R::Key> = Levels::new(sh.num_nodes());
    let root_h = match rule.land(root) {
        Land::Done(r) => {
            form.out = r;
            form.collect_in(Vec::new());
            return Ok(form);
        }
        Land::At(u) => lv.insert(sh, u, root),
    };
    let mut buf = Vec::new();
    let mut order: Vec<u32> = Vec::new();
    while let Some(Reverse((_, u))) = lv.heap.pop() {
        let e = lv.at[u as usize] as usize;
        order.push(e as u32);
        let mut cs = Vec::with_capacity(lv.lv[e].keys.len() + 1);
        let mut out = Vec::new();
        cs.push(0u32);
        let mut slot = 0;
        while slot < lv.lv[e].keys.len() {
            let key = lv.lv[e].keys[slot];
            buf.clear();
            rule.expand(u, key, &mut buf);
            for &(kl, kr) in &buf {
                let ll = rule.land(kl);
                if matches!(ll, Land::Done(ZREF)) {
                    continue;
                }
                let lr = rule.land(kr);
                if matches!(lr, Land::Done(ZREF)) {
                    continue;
                }
                let hl = match ll {
                    Land::Done(r) => r,
                    Land::At(w) => lv.insert(sh, w, kl),
                };
                let hr = match lr {
                    Land::Done(r) => r,
                    Land::At(w) => lv.insert(sh, w, kr),
                };
                out.push((hl, hr));
            }
            cs.push(out.len() as u32);
            slot += 1;
        }
        lim.charge_work(out.len() as u64 + lv.lv[e].keys.len() as u64);
        lim.check_stop()?;
        let l = &mut lv.lv[e];
        l.map = FxHashMap::default();
        l.cstart = cs;
        l.combos = out;
    }
    let mut pairs = Vec::new();
    for &e in order.iter().rev() {
        let e = e as usize;
        let u = lv.lv[e].level;
        let cs = std::mem::take(&mut lv.lv[e].cstart);
        let out = std::mem::take(&mut lv.lv[e].combos);
        let mut r = Vec::with_capacity(cs.len() - 1);
        for s in 0..cs.len() - 1 {
            pairs.clear();
            for &(hl, hr) in &out[cs[s] as usize..cs[s + 1] as usize] {
                let (a, b) = (lv.get(hl), lv.get(hr));
                if a != ZREF && b != ZREF {
                    pairs.push((a, b));
                }
            }
            r.push(if pairs.is_empty() {
                ZREF
            } else {
                match normalize(sh, u, &mut pairs) {
                    Err(target) => target,
                    Ok(()) => mk(u, form.levels[u as usize].push(&pairs)),
                }
            });
        }
        lv.lv[e].res = r;
        lv.lv[e].keys = Vec::new();
    }
    form.out = lv.get(root_h);
    let cand = order.iter().map(|&e| lv.lv[e as usize].level).collect();
    form.collect_in(cand);
    Ok(form)
}
