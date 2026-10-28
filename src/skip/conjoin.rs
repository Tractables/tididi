//! Conjunction in skip form.
//!
//! The product of two references, read from the same child level, is built
//! where the first of them is a stored node: at that node's level when one
//! names a node above the other's, and where they part when neither does
//! (two leaves under one level). Above it both skip the same variables, so
//! the product skips them too. A reference that skips through the level
//! where the product is built acts as the one pair `true` on the skipped
//! variable makes.

use std::sync::Arc;

use super::build::{construct, Land, Rule};
use super::{ix, mk, vt, Ref, Shape, SkipForm, NEG, ONE, POS, ZREF};
use crate::limits::Limits;
use crate::OperationError;

/// Where the product of `a` and `b` (references read from one child level)
/// is built: `Done` for a false or leaf result.
#[inline]
pub(crate) fn land_pair(sh: &Shape, a: Ref, b: Ref) -> Land {
    if a == ZREF || b == ZREF {
        return Land::Done(ZREF);
    }
    let (va, vb) = (vt(a), vt(b));
    if va == vb {
        if sh.is_leaf(va) {
            return Land::Done(leaf_and(va, ix(a), ix(b)));
        }
        return Land::At(va);
    }
    if sh.within(vb, va) {
        return Land::At(va);
    }
    if sh.within(va, vb) {
        return Land::At(vb);
    }
    // They part below a level both skip through: each skips the other's
    // single variable, so both are leaves under it.
    debug_assert!(sh.is_leaf(va) && sh.is_leaf(vb) && sh.parent[va as usize] == sh.parent[vb as usize]);
    Land::At(sh.parent[va as usize])
}

#[inline]
fn leaf_and(v: u32, a: u32, b: u32) -> Ref {
    match (a, b) {
        (ONE, x) | (x, ONE) => mk(v, x),
        (POS, POS) => mk(v, POS),
        (NEG, NEG) => mk(v, NEG),
        _ => ZREF,
    }
}

struct Conj<'a> {
    f: &'a SkipForm,
    g: &'a SkipForm,
}

impl Rule for Conj<'_> {
    type Key = (Ref, Ref);

    #[inline]
    fn land(&mut self, (a, b): (Ref, Ref)) -> Land {
        land_pair(&self.f.shape, a, b)
    }

    fn expand(&mut self, u: u32, (a, b): (Ref, Ref), out: &mut Vec<((Ref, Ref), (Ref, Ref))>) {
        let (mut ba, mut bb) = ([(0, 0); 1], [(0, 0); 1]);
        let pa = self.f.pairs_at(u, a, &mut ba);
        let pb = self.g.pairs_at(u, b, &mut bb);
        for &(la, ra) in pa {
            for &(lb, rb) in pb {
                out.push(((la, lb), (ra, rb)));
            }
        }
    }
}

/// `f ∧ g`, pruned, in skip form; not contracted.
pub(crate) fn and(f: &SkipForm, g: &SkipForm, lim: &Limits) -> Result<SkipForm, OperationError> {
    debug_assert!(Arc::ptr_eq(&f.shape, &g.shape));
    let shape = Arc::clone(&f.shape);
    let root = shape.root;
    // Both outputs are read from the root.
    let _ = root;
    construct(&shape, &mut Conj { f, g }, (f.out, g.out), lim)
}
