//! Breadth-first construction of a skip-form result.
//!
//! A [`Rule`] names each result node by a key. The construction finds every
//! key the output needs, level by level from the root (a key's children land
//! strictly below it, so preorder meets each level once all its keys are
//! in), then builds the nodes from the leaves up: a pair with a false side is
//! dropped, a node with no pairs is false, and a node that is `true` on a
//! single-variable side resolves to the reference it skips to.

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

pub(crate) fn construct<R: Rule>(
    shape: &Arc<Shape>,
    rule: &mut R,
    root: R::Key,
    lim: &Limits,
) -> Result<SkipForm, OperationError> {
    let sh = &**shape;
    let n = sh.num_nodes();
    let mut form = SkipForm::zero(Arc::clone(shape));
    let mut maps: Vec<FxHashMap<R::Key, u32>> = (0..n).map(|_| FxHashMap::default()).collect();
    let mut keys: Vec<Vec<R::Key>> = vec![Vec::new(); n];
    let mut cstart: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut combos: Vec<Vec<(u64, u64)>> = vec![Vec::new(); n];
    let insert = |maps: &mut Vec<FxHashMap<R::Key, u32>>, keys: &mut Vec<Vec<R::Key>>, u: u32, k: R::Key| -> u64 {
        let m = &mut maps[u as usize];
        let next = keys[u as usize].len() as u32;
        let slot = *m.entry(k).or_insert_with(|| {
            keys[u as usize].push(k);
            next
        });
        PENDING | (u as u64) << 32 | slot as u64
    };
    let root_h = match rule.land(root) {
        Land::Done(r) => {
            form.out = r;
            return Ok(form);
        }
        Land::At(u) => insert(&mut maps, &mut keys, u, root),
    };
    let mut buf = Vec::new();
    for &u in &sh.topdown {
        if keys[u as usize].is_empty() {
            continue;
        }
        let mut cs = Vec::with_capacity(keys[u as usize].len() + 1);
        let mut out = Vec::new();
        cs.push(0u32);
        let mut slot = 0;
        while slot < keys[u as usize].len() {
            let key = keys[u as usize][slot];
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
                    Land::At(w) => insert(&mut maps, &mut keys, w, kl),
                };
                let hr = match lr {
                    Land::Done(r) => r,
                    Land::At(w) => insert(&mut maps, &mut keys, w, kr),
                };
                out.push((hl, hr));
            }
            cs.push(out.len() as u32);
            slot += 1;
        }
        lim.charge_work(out.len() as u64 + keys[u as usize].len() as u64);
        lim.check_stop()?;
        maps[u as usize] = FxHashMap::default();
        cstart[u as usize] = cs;
        combos[u as usize] = out;
    }
    drop(maps);
    let mut res: Vec<Vec<Ref>> = vec![Vec::new(); n];
    let get = |res: &Vec<Vec<Ref>>, h: u64| -> Ref {
        if h != ZREF && h & PENDING != 0 {
            res[((h >> 32) & 0x7fff_ffff) as usize][h as u32 as usize]
        } else {
            h
        }
    };
    let mut pairs = Vec::new();
    for &u in &sh.bottomup {
        let cs = std::mem::take(&mut cstart[u as usize]);
        if cs.is_empty() {
            continue;
        }
        let out = std::mem::take(&mut combos[u as usize]);
        let mut r = Vec::with_capacity(cs.len() - 1);
        for s in 0..cs.len() - 1 {
            pairs.clear();
            for &(hl, hr) in &out[cs[s] as usize..cs[s + 1] as usize] {
                let (a, b) = (get(&res, hl), get(&res, hr));
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
        res[u as usize] = r;
        keys[u as usize] = Vec::new();
    }
    form.out = get(&res, root_h);
    form.collect();
    Ok(form)
}
