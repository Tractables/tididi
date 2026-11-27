//! What a test needs from crate-private state and the library does not run:
//! infallible fixture builders, an engine that stops at once, a minimized
//! result, whole-tree
//! vtree rotations, re-homing a diagram at a child of its root, storage
//! equality, and the count of the ways the care restriction's walk took.

use num_bigint::BigUint;

use crate::Engine;
use crate::limits::{LimitConfig, StopDecision};
use crate::value::{Count, CountVec};
use crate::vtree::rotate::rotate_pointers;
use crate::vtree::RotationKind;
use crate::vtree::rotate::RotationInfo;
use crate::diagram::{ChildSide, Tdd, TddNodeId};
use crate::vtree::{Vtree, VtreeIdx, VtreeNode};

/// The same diagram, level for level with the same numbering, not just the
/// same function: what an operation that gives its input back must return.
pub(crate) fn same_storage(a: &Tdd, b: &Tdd) -> bool {
    a.same_levels(b)
}

/// What is known of `t` besides its levels: whether it is certified
/// canonical, whether its levels are closed, whether it keeps level counts,
/// and its worklists. An operation that gives its input back returns it
/// known as it was given.
pub(crate) fn known(t: &Tdd) -> (bool, bool, bool, String) {
    (t.levels.is_canonical(t.output), t.levels.is_closed(), t.has_level_counts(), format!("{:?}", t.dirty))
}

/// Count-column fixture builders that expect every reservation to succeed.
pub(crate) trait CountVecExt {
    /// `push`, infallible.
    fn push_i(&mut self, eng: &Engine, c: Count);
    /// `try_clone`, infallible.
    fn clone_guarded(&self, eng: &Engine) -> Self;
    /// The big table's entry for slot `i`, `None` when the table has none
    /// there (no table yet, or a plain fast slot).
    fn big_val(&self, eng: &Engine, i: usize) -> Option<BigUint>;
}

impl CountVecExt for CountVec {
    fn push_i(&mut self, eng: &Engine, c: Count) {
        self.push(eng, c).expect("test allocation succeeds")
    }

    fn clone_guarded(&self, eng: &Engine) -> Self {
        self.try_clone(eng).expect("test allocation succeeds")
    }

    fn big_val(&self, eng: &Engine, i: usize) -> Option<BigUint> {
        let (_, big) = self.clone_guarded(eng).into_parts();
        big.and_then(|b| b.get(i).cloned())
    }
}

/// A fresh engine whose schedule stops the first operation that asks it —
/// the preemption the deadline tests assert, without a wall clock.
#[must_use]
pub(crate) fn stopping_engine() -> Engine {
    let engine = Engine::new();
    let _prior = engine.limits().install(LimitConfig::none().with_stop_callback(Some(crate::limits::StopCallback::new(|_, _| StopDecision::Stop))));
    engine
}

/// An operation's result, then `minimize`: for tests that compare a result
/// with a canonical one.
#[must_use]
pub(crate) fn minimized(engine: &Engine, r: Result<Tdd, crate::OperationError>) -> Tdd {
    let mut f = r.unwrap();
    engine.minimize(&mut f).unwrap();
    f
}

/// Left-rotate the vtree at node `v`, promoting `v`'s right child, and repair
/// the bottom-up order. `None` if `v` or its right child is a leaf.
///
/// The rotation search rotates through `rotate_pointers` and commits or
/// reverts the pending order itself; this whole-rotation form is what the
/// tests are written against.
pub(crate) fn rotate_left(vtree: &mut Vtree, v: VtreeIdx) -> Option<RotationInfo> {
    rotate(vtree, v, RotationKind::Left)
}

/// Rotate at `v` in direction `kind` and repair the bottom-up order.
pub(crate) fn rotate(vtree: &mut Vtree, v: VtreeIdx, kind: RotationKind) -> Option<RotationInfo> {
    assert!(vtree.link_home().is_none(), "the tests rotate root vtrees");
    vtree.stop_being_home();
    let pending = rotate_pointers(vtree, v, kind)?;
    Some(pending.commit_in_probe(vtree))
}

/// Right-rotate the vtree at node `v`, promoting `v`'s left child, and repair
/// the bottom-up order. `None` if `v` or its left child is a leaf.
pub(crate) fn rotate_right(vtree: &mut Vtree, v: VtreeIdx) -> Option<RotationInfo> {
    rotate(vtree, v, RotationKind::Right)
}

/// Re-home a diagram that depends only on variables under one child of its
/// root so that it is rooted at that child — a genuinely low-rooted diagram.
///
/// Building and applying always root at the vtree root, so re-homing is the
/// only way to reach the differing-root operand shape a tightly-rooted segment
/// would take. The root level must hold a single identity pair, the `g ∧ ⊤`
/// shape a single-region function compiles to.
#[cfg_attr(not(debug_assertions), expect(dead_code, reason = "its one caller is a debug-build test"))]
pub(crate) fn reroot_to_child(t: &Tdd, child: ChildSide) -> Tdd {
    let root = t.output.vtree;
    let (lc, rc) = match *t.vtree.node(root) {
        VtreeNode::Internal { left, right, .. } => (left, right),
        VtreeNode::Leaf { .. } => panic!("reroot_to_child: the root must be internal"),
    };
    let pairs = t.levels[root.idx()].pairs_vec(t.output.local.idx());
    assert_eq!(pairs.len(), 1, "reroot_to_child expects the single-region g ∧ ⊤ shape");
    let p = pairs[0];
    let (child, local) = if child == ChildSide::Left { (lc, p.left) } else { (rc, p.right) };
    Tdd::from_levels_unchecked(
        t.vtree.clone(),
        t.levels.clone().into_vec(),
        TddNodeId { vtree: child, local: t.levels[child.idx()].child_decoder().node(local) },
    )
}

thread_local! {
    /// The care restriction's walks marked off the listings, its walks
    /// marked by the descent, and its levels joined as a spine, by both
    /// children's products and by one child's beside `⊤`, on this thread
    /// ([`note_restrict_path`]).
    static RESTRICT_PATHS: std::cell::Cell<[usize; 5]> = const { std::cell::Cell::new([0; 5]) };
}

/// Count one more of the care restriction's path `k` on this thread: `0` a
/// walk marked off the listings, `1` one marked by the descent, `2` a level
/// joined as a spine, `3` one joined by both children's products, `4` one
/// of those whose other child `care` reads as `⊤`.
pub(crate) fn note_restrict_path(k: usize) {
    RESTRICT_PATHS.with(|c| {
        let mut n = c.get();
        n[k] += 1;
        c.set(n);
    });
}

/// The care restriction's paths taken on this thread so far
/// ([`note_restrict_path`]).
pub(crate) fn restrict_paths() -> [usize; 5] {
    RESTRICT_PATHS.with(std::cell::Cell::get)
}

thread_local! {
    /// The care restriction's caps on this thread when a test sets them
    /// ([`with_restrict_caps`]).
    static RESTRICT_CAPS: std::cell::Cell<Option<(u64, usize)>> = const { std::cell::Cell::new(None) };
}

/// The caps [`with_restrict_caps`] set on this thread, if any.
pub(crate) fn restrict_caps() -> Option<(u64, usize)> {
    RESTRICT_CAPS.with(std::cell::Cell::get)
}

/// Run `f` with the care restriction's joins capped at `list` probes, past
/// which a level is decided on demand, and its grids at `grid`, past which
/// a pair of nodes is indexed on a child, so that small diagrams reach the
/// paths large ones take.
pub(crate) fn with_restrict_caps<R>(list: u64, grid: usize, f: impl FnOnce() -> R) -> R {
    struct Reset(Option<(u64, usize)>);
    impl Drop for Reset {
        fn drop(&mut self) {
            RESTRICT_CAPS.with(|c| c.set(self.0));
        }
    }
    let _reset = Reset(RESTRICT_CAPS.with(|c| c.replace(Some((list, grid)))));
    f()
}

/// `f` on `vtree`, a vtree of its home holding its leaves, every leaf `f`
/// lacks named true: a constructor's result as it was before constructors
/// built on their own variables.
pub(crate) fn padded(f: Tdd, vtree: &std::sync::Arc<Vtree>) -> Tdd {
    if std::sync::Arc::ptr_eq(f.vtree(), vtree) {
        return f;
    }
    let canonical = f.levels.is_canonical(f.output);
    let mut padded = embedded(&f, vtree, |v| v).expect("a vtree of the same home holding the diagram's leaves").0;
    if canonical {
        Engine::new().minimize_plain(&mut padded).unwrap();
    }
    padded
}

/// A copy of `f`, a structural diagram, on `into`, which holds its shape
/// under `map`: each source level at its image, a pass-through level where
/// renamed variables lie on one side only, a true level where none lie, the
/// pass-throughs over a leaf pruned, every internal level owing the
/// contraction passes, certified where `f` was; with where each level went.
pub(crate) fn embedded(
    f: &Tdd,
    into: &std::sync::Arc<Vtree>,
    map: impl Fn(crate::vtree::VarId) -> crate::vtree::VarId,
) -> Result<(Tdd, crate::restructure::Embedding), crate::restructure::EmbedError> {
    use crate::restructure::embed::{place_by, Free, Plan};
    let eng = Engine::new();
    f.require_structure()?;
    let plan = Plan::build(&eng, f.vtree(), into, map, true, Free::Build)?;
    let mut result = place_by(&eng, std::borrow::Cow::Borrowed(f), into, &plan, Free::Build).map_err(|(e, _)| e)?;
    if f.is_zero() || f.is_one() {
        return Ok((result, plan.embedding));
    }
    result.dirty = crate::diagram::Dirty::default().seeded(into, into.internal_bottomup_slice().iter().copied(), Some(&eng))?;
    if plan.pass_throughs(into).any(|&t| {
        let (left, right) = into.children(t);
        into.node(if plan.is_free(left) { right } else { left }).is_leaf()
    }) {
        eng.reduce_plain(&mut result, crate::reduce::ReductionPlan::Prune)?;
    }
    if f.levels.is_canonical(f.output) {
        result.certify_built();
    }
    Ok((result, plan.embedding))
}

/// `f` moved onto `into`, which holds its shape under `map`, free levels
/// built, certified where `f` was and nothing is listed loose.
pub(crate) fn moved(
    eng: &Engine,
    f: Tdd,
    into: &std::sync::Arc<Vtree>,
    map: impl Fn(crate::vtree::VarId) -> crate::vtree::VarId,
) -> Result<(Tdd, crate::restructure::Embedding), crate::restructure::EmbedError> {
    let canonical = f.levels.is_canonical(f.output);
    let (mut result, plan) = crate::restructure::embed::place_moving(eng, f, into, map, crate::restructure::embed::Free::Build)?;
    if canonical && result.dirty.loose().is_some_and(|loose| loose.is_empty()) {
        result.certify_built();
    }
    Ok((result, plan.embedding))
}
