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
//! the next level compares. After the merges the levels are rebuilt from
//! the leaves up, so that a class that became `true` on one side resolves
//! to the reference it skips to.

use rustc_hash::FxHashMap;

use super::{ix, mk, normalize, vt, Ref, SkipForm, NEG, ONE, POS, ZREF};

use crate::limits::Limits;
use crate::OperationError;

const NONE: u32 = u32::MAX;

/// `s` contracted to the canonical skip form of its function. `s` holds no
/// false side and no unreachable node.
pub(crate) fn contract(s: &SkipForm, lim: &Limits) -> Result<SkipForm, OperationError> {
    let sh = &*s.shape;
    let n = sh.num_nodes();
    if s.out == ZREF {
        return Ok(SkipForm::zero(s.shape.clone()));
    }
    // Per level: the merged classes' pairs (sides as the input's
    // references), the class of each stored node, and the class of each
    // skip a merge took in.
    let mut classes: Vec<Vec<Vec<(Ref, Ref)>>> = vec![Vec::new(); n];
    let mut classmap: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut chainmap: Vec<FxHashMap<Ref, u32>> = (0..n).map(|_| FxHashMap::default()).collect();
    let mut occ: Vec<(Ref, u32, Ref)> = Vec::new();
    for &c in &sh.topdown {
        let lvl = &s.levels[c as usize];
        let mut cmap = vec![NONE; lvl.width()];
        let mut cls: Vec<Vec<(Ref, Ref)>> = Vec::new();
        let p = sh.parent[c as usize];
        if p != NONE {
            let (pl, pr) = sh.kids[p as usize];
            let left = pl == c;
            let sib = if left { pr } else { pl };
            let sib_leaf = sh.is_leaf(sib);
            // Each class of the parent that names an entity of this level,
            // with the partner it names it with (One as both literals).
            occ.clear();
            for (k, pairs) in classes[p as usize].iter().enumerate() {
                for &(a, b) in pairs {
                    let (e, partner) = if left { (a, b) } else { (b, a) };
                    if sib_leaf && ix(partner) == ONE {
                        occ.push((e, k as u32, mk(sib, POS)));
                        occ.push((e, k as u32, mk(sib, NEG)));
                    } else {
                        occ.push((e, k as u32, partner));
                    }
                }
            }
            occ.sort_unstable();
            // Group entities by their sorted context list.
            let mut sigs: Vec<(Ref, usize, usize)> = Vec::new();
            let mut i = 0;
            while i < occ.len() {
                let mut j = i;
                while j < occ.len() && occ[j].0 == occ[i].0 {
                    j += 1;
                }
                sigs.push((occ[i].0, i, j));
                i = j;
            }
            // The context list without the entity itself, as a key.
            let mut keyed: FxHashMap<Vec<(u32, Ref)>, u32> = FxHashMap::default();
            let mut members: Vec<Vec<Ref>> = Vec::new();
            for &(e, i, j) in &sigs {
                let key: Vec<(u32, Ref)> = occ[i..j].iter().map(|o| (o.1, o.2)).collect();
                let g = *keyed.entry(key).or_insert_with(|| {
                    members.push(Vec::new());
                    (members.len() - 1) as u32
                });
                members[g as usize].push(e);
            }
            let (l, r) = sh.kids[c as usize];
            for m in members {
                if m.len() == 1 && vt(m[0]) != c {
                    // A lone skip stays a skip.
                    continue;
                }
                let k = cls.len() as u32;
                let mut pairs = Vec::new();
                for &e in &m {
                    if vt(e) == c {
                        cmap[ix(e) as usize] = k;
                        pairs.extend_from_slice(lvl.node(ix(e)));
                    } else {
                        chainmap[c as usize].insert(e, k);
                        pairs.push(if sh.within(vt(e), l) { (e, mk(r, ONE)) } else { (mk(l, ONE), e) });
                    }
                }
                cls.push(pairs);
            }
        }
        // Stored nodes no class of the parent names directly: the output,
        // and nodes skips reach. Each is its own class.
        for i in 0..lvl.width() {
            if cmap[i] == NONE {
                cmap[i] = cls.len() as u32;
                cls.push(lvl.node(i as u32).to_vec());
            }
        }
        lim.charge_work(lvl.pairs.len() as u64);
        classes[c as usize] = cls;
        classmap[c as usize] = cmap;
    }
    lim.check_stop()?;
    // Rebuild from the leaves up.
    let mut out = SkipForm::zero(s.shape.clone());
    let mut res: Vec<Vec<Ref>> = vec![Vec::new(); n];
    let target = |res: &Vec<Vec<Ref>>, classmap: &Vec<Vec<u32>>, h: Ref| -> Ref {
        if sh.is_leaf(vt(h)) {
            h
        } else {
            res[vt(h) as usize][classmap[vt(h) as usize][ix(h) as usize] as usize]
        }
    };
    let side = |res: &Vec<Vec<Ref>>, classmap: &Vec<Vec<u32>>, chainmap: &Vec<FxHashMap<Ref, u32>>, d: u32, h: Ref| -> Ref {
        if sh.is_leaf(d) || vt(h) == d {
            target(res, classmap, h)
        } else if let Some(&k) = chainmap[d as usize].get(&h) {
            res[d as usize][k as usize]
        } else {
            target(res, classmap, h)
        }
    };
    let mut buf = Vec::new();
    for &c in &sh.bottomup {
        let (l, r) = sh.kids[c as usize];
        let cls = std::mem::take(&mut classes[c as usize]);
        let mut rc = Vec::with_capacity(cls.len());
        for pairs in &cls {
            buf.clear();
            for &(a, b) in pairs {
                buf.push((side(&res, &classmap, &chainmap, l, a), side(&res, &classmap, &chainmap, r, b)));
            }
            // Merged children can repeat a pair.
            buf.sort_unstable();
            buf.dedup();
            rc.push(match normalize(sh, c, &mut buf) {
                Err(t) => t,
                Ok(()) => mk(c, out.levels[c as usize].push(&buf)),
            });
        }
        res[c as usize] = rc;
    }
    out.out = side(&res, &classmap, &chainmap, sh.root, s.out);
    out.collect();
    Ok(out)
}
