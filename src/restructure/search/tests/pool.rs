//! [`Engine::rotate_pool_if`] and [`Engine::pool_search`]: every member keeps
//! its function under every move, a declined move puts every member back,
//! and the pool always shares one home.

use std::sync::Arc;

use crate::restructure::search::{PoolMove, PoolRestarts, PoolSearchConfig, RotationMove};
use crate::test_helpers::{and2, assert_canonical, compile_clauses, cube};
use crate::vtree::{RotationKind, Vtree, VtreeIdx};
use crate::Tdd;

const VARS: u32 = 6;

thread_local! {
    /// Whether the pool search charges a cached refusal rather than probing
    /// the move again; a test turns it off to compare.
    static REPLAY_REFUSALS: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    /// The refusals the pool search charged without probing.
    static REPLAYED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// The members a scored probe scored.
    static SCORED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// The moves the pool search kept, in order, those of restarts put back
    /// included.
    static ACCEPTED: std::cell::RefCell<Vec<PoolMove>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Count a member a scored probe scores.
pub(super) fn note_score() {
    SCORED.with(|n| n.set(n.get() + 1));
}

/// Log a move the pool search keeps.
pub(super) fn note_accept(mv: PoolMove) {
    ACCEPTED.with(|a| a.borrow_mut().push(mv));
}

/// The search's table at its start against the reader of each member.
pub(super) fn check_reader(carried: &super::Carried, members: &[&mut Tdd]) {
    assert_eq!(
        super::live_pairs(members) + carried.pairs(),
        members.iter().map(|m| m.embedded_pair_count()).sum::<usize>(),
        "the table's total against the reader"
    );
}

/// The search's table taken from scratch: from the members' levels and the
/// groups' leaves under each node of `tree`, without the upkeep of
/// `passing` or of the groups' parents.
pub(super) fn check_carried(carried: &super::Carried, members: &[&mut Tdd], tree: &Vtree, groups: &super::Groups) {
    use super::NONE;
    use std::collections::HashMap;
    let mut passing = vec![0u64; tree.num_nodes()];
    let mut total = 0u64;
    let mut cnt = vec![0usize; tree.num_nodes()];
    let mut cur = vec![NONE; tree.num_nodes()];
    for group in &groups.groups {
        let Some(kids) = &group.kids else { continue };
        if kids.is_empty() {
            continue;
        }
        let k = kids.len().div_ceil(2);
        let mut parent = vec![NONE; kids.len()];
        for (t, &(l, r)) in kids.iter().enumerate() {
            if l.idx() != t {
                parent[l.idx()] = t as u32;
                parent[r.idx()] = t as u32;
            }
        }
        assert_eq!(parent, group.up, "the own parents kept");
        let own = &members[group.members[0]].vtree;
        let leaf_at: HashMap<_, _> = own.leaf_bottomup().map(|(t, var)| (var, t)).collect();
        let mut widths = vec![vec![1u32; kids.len()]; group.members.len()];
        for (j, &i) in group.members.iter().enumerate() {
            super::carried::own_widths(members[i], kids, &mut widths[j]);
            let at = carried.at[i];
            assert_eq!(widths[j][..], carried.widths[at..at + kids.len()], "member {i}'s widths");
        }
        let g = |t: u32| widths.iter().map(|w| u64::from(w[t as usize] - 1)).sum::<u64>();
        for (u, var) in tree.leaf_bottomup() {
            (cnt[u.idx()], cur[u.idx()]) = match leaf_at.get(&var) {
                Some(&t) => (1, t.0),
                None => (0, NONE),
            };
        }
        for (u, l, r) in tree.internal_bottomup() {
            let (cl, cr) = (cnt[l.idx()], cnt[r.idx()]);
            (cnt[u.idx()], cur[u.idx()]) = match (cl > 0, cr > 0) {
                (true, true) => {
                    let p = parent[cur[l.idx()] as usize];
                    assert_eq!(p, parent[cur[r.idx()] as usize], "the union has the group's node");
                    assert_eq!(group.union_of[p as usize], u);
                    (cl + cr, p)
                }
                (false, false) => (0, NONE),
                (true, false) => (cl, cur[l.idx()]),
                (false, true) => (cr, cur[r.idx()]),
            };
            if (cl == 0) != (cr == 0) && cnt[u.idx()] < k {
                total += g(cur[u.idx()]);
            }
        }
        for u in 0..tree.num_nodes() {
            if cnt[u] > 0 && cnt[u] < k {
                passing[u] += g(cur[u]);
            }
        }
    }
    assert_eq!(passing, carried.passing, "the carried widths by node");
    assert_eq!(total, carried.total, "the carried pairs");
}

/// Whether the pool search replays the cached refusal it holds, counting
/// each one it does.
pub(super) fn replays_refusal() -> bool {
    let on = REPLAY_REFUSALS.with(std::cell::Cell::get);
    if on {
        REPLAYED.with(|n| n.set(n.get() + 1));
    }
    on
}

/// Five functions of six variables on the root of one balanced home, each
/// depending on every variable, so they share the root as their vtree.
/// Pools on projections are in [`home`].
fn pool() -> Vec<Tdd> {
    let home = crate::vtree::Home::new(Vtree::balanced(VARS));
    let mut members = vec![
        compile_clauses(&home, &[vec![1, 2], vec![-2, 3], vec![3, 4], vec![-4, 5], vec![5, -6], vec![1, -6]]),
        compile_clauses(&home, &[vec![1, 6], vec![2, -5], vec![-3, 4]]),
        compile_clauses(&home, &[vec![-1, -4], vec![2, 5], vec![3, 6], vec![-2, -3]]),
        compile_clauses(&home, &[vec![1, 2, 3, 4, 5, 6]]),
        compile_clauses(&home, &[vec![-4], vec![1, 2, 3, 5, 6]]),
    ];
    for m in &mut members {
        m.minimize().expect("an unarmed engine refuses nothing");
        on_root(m);
    }
    members
}

/// `f`, which depends on every leaf of its home, on the home's root.
fn on_root(f: &mut Tdd) {
    let root: Arc<Vtree> = Arc::clone(&f.home());
    if Arc::ptr_eq(f.vtree(), &root) {
        return;
    }
    match Vtree::leaf_sets(f.vtree(), &root) {
        Ok(crate::vtree::LeafSets::Equal(same)) => f.reseat(&same),
        other => panic!("not on every leaf of its home: {other:?}"),
    }
}

/// A cube of every variable of `home`, each true iff `positive`, on the
/// home's root.
fn full_cube(home: &crate::vtree::Home, positive: bool) -> Tdd {
    let literals: Vec<(u32, bool)> = (1..=home.num_leaves()).map(|v| (v, positive)).collect();
    let mut f = cube(home, &literals);
    on_root(&mut f);
    f
}

/// The tree a pool search moves, kept as the search keeps its copy: the
/// members' home's root when a member is on it, else the projection on
/// every member's leaves, rotated in place so its nodes keep their indices
/// across moves. The members' vtrees are rebuilt from a new home after each
/// kept move, so their indices do not follow the search's.
struct Moved(Vtree);

impl Moved {
    fn new(members: &[Tdd]) -> Self {
        let home = members[0].home();
        let union = match members.iter().any(|m| m.vtree().link_home().is_none()) {
            true => Arc::clone(&home),
            false => {
                let vars: Vec<_> = members.iter().flat_map(|m| m.vtree().leaf_bottomup().map(|(_, var)| var)).collect();
                match vars.is_empty() {
                    true => Vtree::empty_projection(&home),
                    false => Vtree::projection(&home, vars).unwrap(),
                }
            }
        };
        Moved(Vtree::clone(&union))
    }

    fn internals(&self) -> Vec<VtreeIdx> {
        self.0.internal_bottomup().map(|(v, _, _)| v).collect()
    }

    /// `mv`, at a node of this tree, at the node of the members' home over
    /// the same leaves, as [`Engine::rotate_pool_if`](crate::Engine::rotate_pool_if)
    /// takes it.
    fn at_home(&self, members: &[&mut Tdd], mut mv: PoolMove) -> PoolMove {
        let home = members[0].home();
        let (mut lo, mut hi, mut stack) = (u32::MAX, 0, vec![mv.rotation.pivot]);
        while let Some(t) = stack.pop() {
            match *self.0.node(t) {
                crate::vtree::VtreeNode::Leaf { var, .. } => {
                    let rank = home.rank(var).expect("the home has the members' leaves");
                    (lo, hi) = (lo.min(rank), hi.max(rank));
                }
                crate::vtree::VtreeNode::Internal { left, right, .. } => stack.extend([left, right]),
            }
        }
        mv.rotation.pivot = home.node_over(lo, hi);
        mv
    }

    fn kept(&mut self, mv: PoolMove) {
        super::move_in_place(&mut self.0, mv).expect("a kept move applies").commit_in_probe(&mut self.0);
    }
}

/// Which of the 2^VARS assignments satisfy `f`, found by conjoining `f` with
/// each assignment's cube on `f`'s own vtree.
fn truth_table(f: &Tdd) -> Vec<bool> {
    (0..1u32 << VARS)
        .map(|bits| {
            let literals: Vec<(u32, bool)> = (0..VARS).map(|i| (i + 1, bits >> i & 1 == 1)).collect();
            !and2(f, &cube(&f.home(), &literals)).is_zero()
        })
        .collect()
}

fn moves(vtree: &Vtree) -> Vec<PoolMove> {
    let mut out = Vec::new();
    for i in 0..vtree.num_nodes() {
        let pivot = VtreeIdx(i as u32);
        if vtree.node(pivot).is_leaf() {
            continue;
        }
        for kind in [RotationKind::Left, RotationKind::Right] {
            for crossed in [false, true] {
                out.push(PoolMove { rotation: RotationMove { pivot, kind }, crossed });
            }
        }
    }
    out
}

fn with_members<R>(members: &mut [Tdd], f: impl FnOnce(&mut [&mut Tdd]) -> R) -> R {
    let mut refs: Vec<&mut Tdd> = members.iter_mut().collect();
    f(&mut refs)
}

#[test]
fn a_kept_move_preserves_every_function_and_leaves_one_shared_vtree() {
    let original = pool();
    let tables: Vec<Vec<bool>> = original.iter().map(truth_table).collect();
    let eng = crate::Engine::new();
    let mut kept_moves = [0usize; 2];
    for mv in moves(original[0].vtree()) {
        let mut members = original.clone();
        let kept = with_members(&mut members, |refs| eng.rotate_pool_if(refs, mv, usize::MAX, |_| true)).unwrap();
        if !kept {
            continue;
        }
        kept_moves[usize::from(mv.crossed)] += 1;
        assert!(!members[0].vtree().same_tree(original[0].vtree()), "{mv:?} kept, and the tree did not change");
        let first = Arc::clone(members[0].vtree());
        for (m, table) in members.iter_mut().zip(&tables) {
            assert!(Arc::ptr_eq(m.vtree(), &first), "{mv:?}");
            m.minimize().unwrap();
            assert_canonical(m);
            assert_eq!(&truth_table(m), table, "{mv:?}");
        }
    }
    assert!(kept_moves[0] > 0 && kept_moves[1] > 0, "{kept_moves:?}");
}

#[test]
fn a_declined_move_leaves_every_member_exactly_as_it_was() {
    let original = pool();
    let eng = crate::Engine::new();
    for mv in moves(original[0].vtree()) {
        let mut members = original.clone();
        let kept = with_members(&mut members, |refs| eng.rotate_pool_if(refs, mv, usize::MAX, |_| false)).unwrap();
        assert!(!kept);
        for (m, o) in members.iter().zip(&original) {
            assert!(Arc::ptr_eq(m.vtree(), o.vtree()), "{mv:?}");
            assert_eq!(m.pair_count(), o.pair_count(), "{mv:?}");
            assert_eq!(m.node_count(), o.node_count(), "{mv:?}");
            assert!(m.equivalent(o).unwrap(), "{mv:?}");
            assert!(crate::test_helpers::same_storage(m, o), "{mv:?}");
            assert_canonical(m);
        }
    }
}

/// The pairs a probe shows `accept` are the pairs the kept move's rebuilt
/// levels hold.
#[test]
fn a_kept_move_rebuilds_the_pairs_its_probe_counted() {
    let original = pool();
    let eng = crate::Engine::new();
    let mut kept = 0;
    for mv in moves(original[0].vtree()) {
        let (left, right) = original[0].vtree().children(mv.rotation.pivot);
        let promoted = match mv.rotation.kind {
            RotationKind::Left => right,
            RotationKind::Right => left,
        };
        let mut members = original.clone();
        let mut counted = Vec::new();
        let accept = |p: &super::PoolProbe<'_>| {
            counted = p.after().to_vec();
            true
        };
        if !with_members(&mut members, |refs| eng.rotate_pool_if(refs, mv, usize::MAX, accept)).unwrap() {
            continue;
        }
        kept += 1;
        for (m, &pairs) in members.iter().zip(&counted) {
            let levels = m.levels();
            assert_eq!(levels[mv.rotation.pivot.idx()].live_pairs() + levels[promoted.idx()].live_pairs(), pairs, "{mv:?}");
        }
    }
    assert!(kept > 0);
}

/// A bound the count of a member's levels would pass abandons the move
/// before `accept` is asked; a move counted within the bound is rebuilt
/// within it.
#[test]
fn a_bound_abandons_a_move_before_it_is_asked_or_not_at_all() {
    let original = pool();
    let tables: Vec<Vec<bool>> = original.iter().map(truth_table).collect();
    let eng = crate::Engine::new();
    let (mut abandoned, mut kept) = (0, 0);
    for bound in [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32] {
        for mv in moves(original[0].vtree()) {
            let mut members = original.clone();
            let mut asked = false;
            let outcome = with_members(&mut members, |refs| eng.rotate_pool_if(refs, mv, bound, |_| {
                asked = true;
                true
            })).unwrap();
            assert_eq!(outcome, asked, "bound {bound}, {mv:?}");
            if outcome {
                kept += 1;
                for (m, table) in members.iter_mut().zip(&tables) {
                    m.minimize().unwrap();
                    assert_eq!(&truth_table(m), table, "bound {bound}, {mv:?}");
                }
            } else {
                abandoned += 1;
                for (m, o) in members.iter().zip(&original) {
                    assert!(crate::test_helpers::same_storage(m, o), "bound {bound}, {mv:?}");
                }
            }
        }
    }
    assert!(abandoned > 0 && kept > 0, "{abandoned} abandoned, {kept} kept");
}

#[test]
fn a_crossed_move_changes_the_order_of_the_leaves() {
    let original = pool();
    let eng = crate::Engine::new();
    let order = |t: &Tdd| -> Vec<u32> {
        let vt = t.vtree();
        let mut out = Vec::new();
        let mut stack = vec![vt.root()];
        while let Some(n) = stack.pop() {
            match *vt.node(n) {
                crate::vtree::VtreeNode::Leaf { var, .. } => out.push(var.0),
                crate::vtree::VtreeNode::Internal { left, right, .. } => {
                    stack.push(right);
                    stack.push(left);
                }
            }
        }
        out
    };
    let before = order(&original[0]);
    let mut changed = 0;
    for mv in moves(original[0].vtree()) {
        let mut members = original.clone();
        if with_members(&mut members, |refs| eng.rotate_pool_if(refs, mv, usize::MAX, |_| true)).unwrap() {
            let after = order(&members[0]);
            if mv.crossed {
                changed += usize::from(after != before);
            } else {
                assert_eq!(after, before, "{mv:?}: a plain rotation keeps the leaf order");
            }
        }
    }
    assert!(changed > 0);
}

#[test]
fn members_on_different_vtrees_are_refused() {
    let mut a = pool();
    let other = crate::vtree::Home::new(Vtree::balanced(VARS));
    let mut b = compile_clauses(&other, &[vec![1, 2]]);
    let eng = crate::Engine::new();
    let mut refs: Vec<&mut Tdd> = vec![&mut a[0], &mut b];
    let mv = moves(refs[0].vtree())[0];
    assert_eq!(
        eng.rotate_pool_if(&mut refs, mv, usize::MAX, |_| true),
        Err(crate::OperationError::VtreeMismatch)
    );
}

#[test]
fn a_pool_search_keeps_every_function_and_never_grows_the_pool() {
    let original = pool();
    let tables: Vec<Vec<bool>> = original.iter().map(truth_table).collect();
    let eng = crate::Engine::new();
    for crossed in [false, true] {
        let mut members = original.clone();
        let config = PoolSearchConfig { max_sweeps: 6, crossed, ..PoolSearchConfig::default() };
        let stats = with_members(&mut members, |refs| eng.pool_search(refs, &config)).unwrap();
        assert!(stats.pairs_after <= stats.pairs_before, "{stats:?}");
        assert_eq!(stats.pairs_after, members.iter().map(Tdd::pair_count).sum::<usize>());
        for (m, table) in members.iter().zip(&tables) {
            assert!(Arc::ptr_eq(m.vtree(), members[0].vtree()));
            assert_canonical(m);
            assert_eq!(&truth_table(m), table, "crossed {crossed}");
        }
    }
}

#[test]
fn a_probe_charges_its_rebuilds_to_the_work_clock() {
    let original = pool();
    let eng = crate::Engine::new();
    let mv = moves(original[0].vtree()).into_iter().find(|&mv| {
        let mut members = original.clone();
        with_members(&mut members, |refs| eng.rotate_pool_if(refs, mv, usize::MAX, |_| true)).unwrap()
    });
    let mv = mv.expect("some move applies");
    let mut members = original.clone();
    let start = eng.limits().work_units();
    assert!(!with_members(&mut members, |refs| eng.rotate_pool_if(refs, mv, usize::MAX, |_| false)).unwrap());
    assert!(eng.limits().work_units() > start);
    for m in &members {
        assert_canonical(m);
    }
}

/// A move its scores refuse charges the probe and, for each member, what
/// a score reads: the pairs of its two levels and the rebuild's fixed
/// units. It charges so also where a score's cap refused the move before
/// every member was scored, as on seeded pools of four functions on every
/// home shape of six to eight variables some probes do. The probe is called
/// on the members' shared tree, so the seeded functions take a clause of
/// every variable, and a pool with a member that still misses a variable is
/// skipped.
#[test]
fn a_refused_score_charges_the_pairs_it_reads() {
    use crate::test_helpers::{compile_clauses_on, rand_cnf, vtree_shapes, CnfShape, Lcg};
    let eng = crate::Engine::new();
    let mut rng = Lcg::new(0xca9);
    let mut pools = vec![pool()];
    for n in 6..9 {
        for (_, home) in vtree_shapes(n) {
            let mut members: Vec<Tdd> = (0..4)
                .map(|_| {
                    let mut cnf = rand_cnf(&mut rng, n, CnfShape { clauses: n as usize, width: 3 });
                    cnf.push((1..=n as i32).collect());
                    let mut f = compile_clauses_on(&eng, &home, &cnf);
                    f.minimize().expect("an unarmed engine refuses nothing");
                    f
                })
                .collect();
            if members.iter().all(|m| m.vtree().num_leaves() == home.num_leaves()) {
                members.iter_mut().for_each(on_root);
                pools.push(members);
            }
        }
    }
    let (mut refused, mut capped) = (0, 0);
    for original in &pools {
        for mv in moves(original[0].vtree()) {
            let mut members = original.clone();
            let mut tree = Vtree::clone(members[0].vtree());
            let Some(pending) = super::move_in_place(&mut tree, mv) else { continue };
            let info = pending.info();
            pending.revert(&mut tree);
            let mut read = 0u64;
            for m in &members {
                let (outer, inner) = (m.levels[info.v_idx.idx()].live_pairs(), m.levels[info.w_idx.idx()].live_pairs());
                let old = (outer + inner) as u64;
                read += old + super::REBUILD_UNITS;
            }
            let start = eng.limits().work_units();
            let scored = SCORED.with(std::cell::Cell::get);
            let mut scratch = crate::restructure::scratch::RestructureScratch::default();
            let kept = with_members(&mut members, |refs| {
                let pending = super::move_in_place(&mut tree, mv).expect("the move applied before");
                let every: Vec<usize> = (0..refs.len()).collect();
                let listed = super::Listed { on_union: &every, info: &info, rest: &[] };
                let moved = super::Touched::new(listed, mv, usize::MAX);
                let kept = super::probe_scored(&eng, refs, moved, &mut scratch, super::Twin::Alone, 0, [0; 2], None);
                pending.revert(&mut tree);
                kept
            })
            .unwrap();
            if kept {
                continue;
            }
            refused += 1;
            capped += usize::from(SCORED.with(std::cell::Cell::get) - scored < members.len());
            assert_eq!(eng.limits().work_units() - start, super::PROBE_UNITS + read, "{mv:?}");
            for (m, o) in members.iter().zip(original) {
                assert!(crate::test_helpers::same_storage(m, o), "{mv:?}");
            }
        }
    }
    assert!(refused > 0 && capped > 0, "{refused} refused, {capped} capped");
}

#[test]
fn a_search_with_no_work_to_spend_probes_nothing() {
    let original = pool();
    let eng = crate::Engine::new();
    let mut members = original.clone();
    let config = PoolSearchConfig { max_work_units: 0, ..PoolSearchConfig::default() };
    let stats = with_members(&mut members, |refs| eng.pool_search(refs, &config)).unwrap();
    assert_eq!((stats.probes, stats.accepts, stats.work_units), (0, 0, 0), "{stats:?}");
    for (m, o) in members.iter().zip(&original) {
        assert!(Arc::ptr_eq(m.vtree(), o.vtree()));
        assert!(m.equivalent(o).unwrap());
        assert_canonical(m);
    }
}

#[test]
fn invalid_pivot_is_refused_without_changing_members() {
    let mut members = pool();
    let original = members.clone();
    let eng = crate::Engine::new();
    let pivot = VtreeIdx(u32::MAX);
    let mv = PoolMove { rotation: RotationMove { pivot, kind: RotationKind::Left }, crossed: false };
    let result = with_members(&mut members, |refs| eng.rotate_pool_if(refs, mv, usize::MAX, |_| panic!("invalid move scored")));
    assert_eq!(result, Err(crate::OperationError::LevelNotInVtree(pivot)));
    for (m, before) in members.iter().zip(&original) {
        assert!(crate::test_helpers::same_storage(m, before));
        assert_canonical(m);
    }
}

#[test]
fn a_stop_before_acceptance_restores_the_pool() {
    use crate::limits::{LimitConfig, StopAt, StopRules};
    let eng = crate::Engine::new();
    let vtree = crate::vtree::Home::new(Vtree::balanced(6));
    let original = full_cube(&vtree, true);
    let mut f = original.clone();
    let start = eng.limits().work_units();
    let rules = StopRules { unconditional: Some(StopAt::WorkUnits(start + 1)), after_pairs: None };
    let result = {
        let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(rules));
        let mv = PoolMove { rotation: RotationMove { pivot: vtree.root(), kind: RotationKind::Left }, crossed: false };
        eng.rotate_pool_if(&mut [&mut f], mv, usize::MAX, |_| panic!("stopped move scored"))
    };
    assert_eq!(result, Err(crate::OperationError::Stopped));
    assert!(crate::test_helpers::same_storage(&f, &original));
    assert!(Arc::ptr_eq(f.vtree(), &vtree));
    assert_canonical(&f);
}

#[test]
fn search_stops_before_another_applicable_move() {
    let eng = crate::Engine::new();
    let vtree = crate::vtree::Home::new(Vtree::balanced(6));
    let mut f = full_cube(&vtree, true);
    let config = PoolSearchConfig { max_work_units: 1, ..PoolSearchConfig::default() };
    let stats = eng.pool_search(&mut [&mut f], &config).unwrap();
    // The move, and the one member's two levels of a pair each, scored and
    // charged a rebuild: the move does not lower the pairs, so it is not
    // kept.
    assert_eq!(
        stats.work_units,
        super::PROBE_UNITS + 2 + super::REBUILD_UNITS,
        "only the first applicable probe may overshoot"
    );
    assert_canonical(&f);
}

#[test]
fn a_declined_move_puts_every_member_back() {
    let eng = crate::Engine::new();
    let vtree = crate::vtree::Home::new(Vtree::balanced(VARS));
    let original = vec![full_cube(&vtree, true), full_cube(&vtree, true), full_cube(&vtree, false)];
    let mv = PoolMove { rotation: RotationMove { pivot: vtree.root(), kind: RotationKind::Left }, crossed: false };
    let mut members = original.clone();
    let start = eng.limits().work_units();
    assert!(!with_members(&mut members, |refs| eng.rotate_pool_if(refs, mv, usize::MAX, |p| {
        assert_eq!(p.before(), p.after());
        false
    })).unwrap());
    // Each member's two levels at the root and its right child hold a pair
    // each: charged its two old and two new pairs and a rebuild, and put
    // back.
    let copy = super::PROBE_UNITS + vtree.num_nodes() as u64;
    assert_eq!(eng.limits().work_units() - start, copy + 3 * (4 + super::REBUILD_UNITS));
    for (m, before) in members.iter().zip(&original) {
        assert!(crate::test_helpers::same_storage(m, before));
        assert!(Arc::ptr_eq(m.vtree(), &vtree));
        assert_canonical(m);
    }
    // Kept, every member is rebuilt onto the rotated tree.
    let tables: Vec<Vec<bool>> = original.iter().map(truth_table).collect();
    assert!(with_members(&mut members, |refs| eng.rotate_pool_if(refs, mv, usize::MAX, |_| true)).unwrap());
    let rotated = Arc::clone(members[0].vtree());
    assert!(!rotated.same_tree(&vtree));
    for (m, table) in members.iter_mut().zip(&tables) {
        assert!(Arc::ptr_eq(m.vtree(), &rotated));
        m.minimize().unwrap();
        assert_canonical(m);
        assert_eq!(&truth_table(m), table);
    }
}

#[test]
fn every_refused_pool_reservation_restores_all_members() {
    let eng = crate::Engine::new();
    let original = pool();
    let mv = PoolMove { rotation: RotationMove { pivot: original[0].vtree().root(), kind: RotationKind::Left }, crossed: true };
    let mut granted = false;
    for nth in 0..1000 {
        let mut members = original.clone();
        eng.refuse_nth_reserve(nth);
        let result = with_members(&mut members, |refs| eng.rotate_pool_if(refs, mv, usize::MAX, |_| true));
        eng.limits().grant_every_reserve();
        match result {
            Err(crate::OperationError::OverBudget) => {
                for (m, before) in members.iter().zip(&original) {
                    assert!(crate::test_helpers::same_storage(m, before), "reservation {nth}");
                    assert!(Arc::ptr_eq(m.vtree(), before.vtree()));
                    assert_canonical(m);
                }
            }
            Ok(true) => { granted = true; break; }
            other => panic!("unexpected probe result at reservation {nth}: {other:?}"),
        }
    }
    assert!(granted, "every finite probe eventually has enough reservations");
}

#[test]
fn every_pool_stop_restores_all_members() {
    use crate::limits::{LimitConfig, StopAt, StopRules};
    let eng = crate::Engine::new();
    let original = pool();
    let mv = PoolMove { rotation: RotationMove { pivot: original[0].vtree().root(), kind: RotationKind::Left }, crossed: true };
    let mut completed = false;
    for cut in 0..5000 {
        let mut members = original.clone();
        let outcome = {
            let stop = StopRules { unconditional: Some(StopAt::WorkUnits(eng.limits().work_units() + cut)), after_pairs: None };
            let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(stop));
            with_members(&mut members, |refs| eng.rotate_pool_if(refs, mv, usize::MAX, |_| true))
        };
        match outcome {
            Ok(true) => { completed = true; break; }
            Err(crate::OperationError::Stopped) => {
                for (m, before) in members.iter().zip(&original) {
                    assert!(crate::test_helpers::same_storage(m, before), "stop {cut}");
                    assert!(Arc::ptr_eq(m.vtree(), before.vtree()));
                    assert_canonical(m);
                }
            }
            other => panic!("unexpected pool result: {other:?}"),
        }
    }
    assert!(completed);
}

/// Every member after a pool search that `outcome` ended: its function, on
/// the vtree every member shares. A search that fails keeps the moves it
/// kept, and the probe under way puts back every member it rebuilt.
fn assert_search_kept_every_function(members: &mut [Tdd], tables: &[Vec<bool>], label: &str) {
    let first = Arc::clone(members[0].vtree());
    for (m, table) in members.iter_mut().zip(tables) {
        assert!(Arc::ptr_eq(m.vtree(), &first), "{label}");
        m.minimize().unwrap();
        assert_canonical(m);
        assert_eq!(&truth_table(m), table, "{label}");
    }
}

#[test]
fn every_stop_in_a_pool_search_keeps_every_function() {
    use crate::limits::{LimitConfig, StopAt, StopRules};
    let eng = crate::Engine::new();
    let original = pool();
    let tables: Vec<Vec<bool>> = original.iter().map(truth_table).collect();
    for config in [PoolSearchConfig { max_sweeps: 6, ..PoolSearchConfig::default() }, with_restarts()] {
        let (mut stopped, mut cut) = (0, 0);
        loop {
            let mut members = original.clone();
            let outcome = {
                let stop = StopRules { unconditional: Some(StopAt::WorkUnits(eng.limits().work_units() + cut)), after_pairs: None };
                let _scope = eng.limits().scope(LimitConfig::none().with_stop_rules(stop));
                with_members(&mut members, |refs| eng.pool_search(refs, &config))
            };
            let done = match outcome {
                Ok(_) => true,
                Err(crate::OperationError::Stopped) => {
                    stopped += 1;
                    false
                }
                other => panic!("unexpected pool search result at stop {cut}: {other:?}"),
            };
            assert_search_kept_every_function(&mut members, &tables, &format!("stop {cut}"));
            if done {
                break;
            }
            cut += 1 + cut / 32;
        }
        assert!(stopped > 0);
    }
}

/// A search with restarts that keep moves on [`pool`].
fn with_restarts() -> PoolSearchConfig {
    PoolSearchConfig {
        max_sweeps: usize::MAX,
        restarts: PoolRestarts { count: 3, kick: 4, slack_permille: 300, ..PoolRestarts::default() },
        ..PoolSearchConfig::default()
    }
}

#[test]
fn restarts_keep_every_function_and_count_their_work() {
    let eng = crate::Engine::new();
    let original = pool();
    let tables: Vec<Vec<bool>> = original.iter().map(truth_table).collect();
    let mut members = original.clone();
    let stats = with_members(&mut members, |refs| eng.pool_search(refs, &with_restarts())).unwrap();
    assert_eq!(stats.restarts, 3, "{stats:?}");
    assert!(stats.restart_work_units > 0 && stats.restart_work_units < stats.work_units, "{stats:?}");
    assert_search_kept_every_function(&mut members, &tables, "restarts");
}

#[test]
fn every_refused_reservation_in_a_pool_search_keeps_every_function() {
    let eng = crate::Engine::new();
    let original = pool();
    let tables: Vec<Vec<bool>> = original.iter().map(truth_table).collect();
    for (config, restarts) in [(PoolSearchConfig { max_sweeps: 6, ..PoolSearchConfig::default() }, false), (with_restarts(), true)] {
        let (mut refused, mut absorbed, mut nth) = (0, 0, 0);
        loop {
            let mut members = original.clone();
            eng.refuse_nth_reserve(nth);
            let outcome = with_members(&mut members, |refs| eng.pool_search(refs, &config));
            let pending = eng.limits().refusal_pending();
            eng.limits().grant_every_reserve();
            let done = match outcome {
                Ok(_) if pending => true,
                // A refusal inside a restart ends the restarts, the pool put
                // back where the descent left it.
                Ok(_) => {
                    absorbed += 1;
                    false
                }
                Err(crate::OperationError::OverBudget) => {
                    refused += 1;
                    false
                }
                other => panic!("unexpected pool search result at reservation {nth}: {other:?}"),
            };
            assert_search_kept_every_function(&mut members, &tables, &format!("reservation {nth}"));
            if done {
                break;
            }
            nth += 1 + nth / 32;
        }
        assert!(refused > 0);
        assert_eq!(absorbed > 0, restarts, "{absorbed} refusals ended the restarts");
    }
}

/// A buffer refused inside the first restart, at its undo reservation or
/// later, ends the restarts with the pool the descent left: the same vtree,
/// pairs and kept moves as a search without restarts, the work units those
/// of that search plus the ones the refused restart charged.
#[test]
fn a_refused_restart_returns_the_descents_pool_and_charges() {
    let eng = crate::Engine::new();
    let original = pool();
    let tables: Vec<Vec<bool>> = original.iter().map(truth_table).collect();
    let descent_only = PoolSearchConfig { max_sweeps: usize::MAX, ..PoolSearchConfig::default() };
    let mut descended = original.clone();
    let plain = with_members(&mut descended, |refs| eng.pool_search(refs, &descent_only)).unwrap();
    assert_eq!(plain.restarts, 0, "{plain:?}");
    let pairs: Vec<usize> = descended.iter().map(Tdd::pair_count).collect();
    let vtree = descended[0].vtree().to_text();
    let search = |nth: u32| {
        let mut members = original.clone();
        eng.refuse_nth_reserve(nth);
        let outcome = with_members(&mut members, |refs| eng.pool_search(refs, &with_restarts()));
        let pending = eng.limits().refusal_pending();
        eng.limits().grant_every_reserve();
        (outcome, pending, members)
    };
    // Every refusal in the descent fails the search; on this pool the
    // doubling stops inside the restarts, so the first reservation whose
    // refusal the search survives is the first restart's. Should the
    // doubling skip the restarts, the loop below finds no refused restart
    // and the assertions after it fail.
    let (mut lo, mut hi) = (0, 1);
    while search(hi).0.is_err() {
        (lo, hi) = (hi + 1, hi * 2);
    }
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if search(mid).0.is_ok() {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    let (mut first, mut charged, mut nth) = (0, 0, lo);
    loop {
        let (outcome, pending, mut members) = search(nth);
        let stats = outcome.unwrap();
        if pending || stats.restarts > 1 {
            break;
        }
        let at = format!("reservation {nth}: {stats:?}");
        assert_eq!(stats.restarts, 1, "{at}");
        assert_eq!(stats.restarts_kept, 0, "{at}");
        assert_eq!(stats.accepts, plain.accepts, "{at}");
        assert_eq!(stats.pairs_before, plain.pairs_before, "{at}");
        assert_eq!(stats.pairs_after, plain.pairs_after, "{at}");
        assert_eq!(stats.work_units, plain.work_units + stats.restart_work_units, "{at}");
        assert_eq!(members.iter().map(Tdd::pair_count).collect::<Vec<_>>(), pairs, "{at}");
        assert_eq!(members[0].vtree().to_text(), vtree, "{at}");
        assert_search_kept_every_function(&mut members, &tables, &at);
        if stats.restart_work_units == 0 {
            first += 1;
        } else {
            charged += 1;
        }
        nth += 1 + (nth - lo) / 16;
    }
    assert!(first > 0, "no refusal before the restart's work");
    assert!(charged > 0, "no refusal inside a restart's work");
}

/// The descent without the refused moves skipped: every move of every
/// sweep probed, in the search's order. Returns the moves kept and probed.
fn full_descent(eng: &crate::Engine, members: &mut [Tdd], config: &PoolSearchConfig) -> (usize, usize) {
    for m in members.iter_mut() {
        eng.reduce(m, crate::reduce::ReductionPlan::default()).unwrap();
    }
    let (accepts, probes, _) = full_sweeps(eng, members, &mut Moved::new(members), config);
    for m in members.iter_mut() {
        eng.reduce(m, crate::reduce::ReductionPlan::default()).unwrap();
    }
    (accepts, probes)
}

/// The sweeps of [`full_descent`], without its reductions. Returns the moves
/// kept and probed, and whether a sweep kept nothing.
fn full_sweeps(eng: &crate::Engine, members: &mut [Tdd], moved: &mut Moved, config: &PoolSearchConfig) -> (usize, usize, bool) {
    let crossings: &[bool] = if config.crossed { &[false, true] } else { &[false] };
    let (mut accepts, mut probes) = (0, 0);
    for _ in 0..config.max_sweeps {
        let internals = moved.internals();
        let mut kept = 0;
        for v in internals {
            'pivot: for kind in [RotationKind::Left, RotationKind::Right] {
                for &crossed in crossings {
                    let mv = PoolMove { rotation: RotationMove { pivot: v, kind }, crossed };
                    probes += 1;
                    if with_members(members, |refs| eng.rotate_pool_if(refs, moved.at_home(refs, mv), usize::MAX, |p| p.live_pairs_delta() < 0)).unwrap() {
                        moved.kept(mv);
                        accepts += 1;
                        kept += 1;
                        break 'pivot;
                    }
                }
            }
        }
        if kept == 0 {
            return (accepts, probes, true);
        }
    }
    (accepts, probes, false)
}

/// What [`full_search_with_restarts`] kept.
#[derive(Default)]
struct Kept {
    /// Restarts kept.
    restarts: usize,
    /// Moves kept: the descent's and those of the restarts kept.
    moves: usize,
    /// Restarts kept whose descent kept no move: their random moves alone
    /// made the pool smaller.
    by_their_random_moves: usize,
}

/// [`Engine::pool_search`](crate::Engine::pool_search) with its restarts,
/// written out: the random moves drawn as the search draws them and made
/// with [`rotate_pool_if`](crate::Engine::rotate_pool_if), each restart's
/// descent probing every move, and a restart that does not end smaller put
/// back from a copy of the pool.
fn full_search_with_restarts(eng: &crate::Engine, members: &mut Vec<Tdd>, config: &PoolSearchConfig) -> Kept {
    use crate::vtree::rng::Lcg;
    let pairs = |members: &[Tdd]| members.iter().map(Tdd::pair_count).sum::<usize>();
    for m in members.iter_mut() {
        eng.reduce(m, crate::reduce::ReductionPlan::default()).unwrap();
    }
    let mut won = Kept::default();
    let mut moved = Moved::new(members);
    let (moves, _, converged) = full_sweeps(eng, members, &mut moved, config);
    won.moves = moves;
    if converged {
        let internals = moved.internals();
        let mut rng = Lcg::new(config.restarts.seed);
        for _ in 0..config.restarts.count {
            let start = (members.clone(), moved.0.clone());
            let before = pairs(members);
            let (mut kept, mut draws) = (0, 0);
            while kept < config.restarts.kick && draws < config.restarts.kick * 50 {
                draws += 1;
                let pivot = internals[rng.below(internals.len() as u64) as usize];
                let kind = if rng.below(2) == 0 { RotationKind::Left } else { RotationKind::Right };
                let crossed = config.crossed && rng.below(2) == 0;
                let slack = (pairs(members) as u64 * u64::from(config.restarts.slack_permille) / 1000) as i64 + 1;
                let mv = PoolMove { rotation: RotationMove { pivot, kind }, crossed };
                if with_members(members, |refs| eng.rotate_pool_if(refs, moved.at_home(refs, mv), usize::MAX, |p| p.live_pairs_delta() < slack)).unwrap() {
                    moved.kept(mv);
                    kept += 1;
                }
            }
            let descended = full_sweeps(eng, members, &mut moved, config).0;
            let gain = before as u64 * u64::from(config.restarts.min_gain_permille) / 1000;
            if (pairs(members) as u64) < (before as u64).saturating_sub(gain) {
                won.restarts += 1;
                won.moves += kept + descended;
                won.by_their_random_moves += usize::from(descended == 0);
            } else {
                (*members, moved.0) = start;
            }
        }
    }
    for m in members.iter_mut() {
        eng.reduce(m, crate::reduce::ReductionPlan::default()).unwrap();
    }
    won
}

fn seeded_pool(
    eng: &crate::Engine,
    rng: &mut crate::test_helpers::Lcg,
    vtree: &crate::vtree::Home,
    n: u32,
    shape: crate::test_helpers::CnfShape,
) -> Vec<Tdd> {
    use crate::test_helpers::{compile_clauses_on, rand_cnf};
    (0..4)
        .map(|_| {
            let mut f = compile_clauses_on(eng, vtree, &rand_cnf(rng, n, shape));
            f.minimize().expect("an unarmed engine refuses nothing");
            f
        })
        .collect()
}

fn assert_same_pool_outcome(searched: &[Tdd], full: &[Tdd], label: &str) {
    assert_eq!(searched[0].home().to_text(), full[0].home().to_text(), "{label}");
    let first = searched[0].home();
    for (s, f) in searched.iter().zip(full) {
        assert!(Arc::ptr_eq(&s.home(), &first), "{label}");
        assert_eq!(s.pair_count(), f.pair_count(), "{label}");
        assert_canonical(s);
    }
}

/// Seeded pools of four functions on every vtree shape of six to nine
/// variables: a search whose restarts descend only where their random moves
/// changed a level, and put the pool back from what the moves replaced, ends
/// with the vtree, the pairs and the moves kept of the search that descends
/// from scratch and puts the pool back from a copy.
#[test]
fn restarts_end_where_restarts_descending_from_scratch_end() {
    use crate::test_helpers::{vtree_shapes, CnfShape, Lcg};
    let eng = crate::Engine::new();
    let mut rng = Lcg::new(0x7e57);
    let (mut restarts, mut won, mut pools) = (0, 0, 0);
    for n in 6..10 {
        for (shape, vtree) in vtree_shapes(n) {
            let members = seeded_pool(&eng, &mut rng, &vtree, n, CnfShape { clauses: n as usize, width: 3 });
            for (crossed, min_gain_permille) in [(false, 0), (true, 0), (true, 100)] {
                let config = PoolSearchConfig {
                    max_sweeps: usize::MAX,
                    crossed,
                    restarts: PoolRestarts { count: 4, kick: 3, slack_permille: 200, seed: n as u64, min_gain_permille },
                    ..PoolSearchConfig::default()
                };
                let label = format!("{n} variables, {shape}, crossed {crossed}, minimum gain {min_gain_permille}");
                let mut searched = members.clone();
                let stats = with_members(&mut searched, |refs| eng.pool_search(refs, &config)).unwrap();
                let mut full = members.clone();
                let kept = full_search_with_restarts(&eng, &mut full, &config);
                assert_eq!((stats.restarts_kept, stats.accepts), (kept.restarts, kept.moves), "{label}");
                assert_same_pool_outcome(&searched, &full, &label);
                restarts += stats.restarts;
                won += kept.restarts;
                pools += 1;
            }
        }
    }
    assert!(won > 0 && restarts > won, "{won} of {restarts} restarts kept over {pools} pools");
}

/// Seeded pools of four small functions on every vtree shape of six to eight
/// variables, under several seeds: a restart whose random moves alone make
/// the pool smaller, its descent and the search's keeping no move, is kept
/// with the home its moves made, which every member then shares, as in the
/// search written out.
#[test]
fn a_restart_kept_for_its_random_moves_alone_leaves_their_vtree() {
    use crate::test_helpers::{vtree_shapes, CnfShape, Lcg};
    let eng = crate::Engine::new();
    let mut rng = Lcg::new(0x7e57);
    let (mut by_their_random_moves, mut pools) = (0, 0);
    for n in 6..9 {
        for (shape, vtree) in vtree_shapes(n) {
            let members = seeded_pool(&eng, &mut rng, &vtree, n, CnfShape { clauses: 2, width: 2 });
            for seed in 0..8 {
                let config = PoolSearchConfig {
                    max_sweeps: usize::MAX,
                    restarts: PoolRestarts { count: 4, kick: 20, slack_permille: 50, seed, min_gain_permille: 0 },
                    ..PoolSearchConfig::default()
                };
                let label = format!("{n} variables, {shape}, seed {seed}");
                let mut searched = members.clone();
                let stats = with_members(&mut searched, |refs| eng.pool_search(refs, &config)).unwrap();
                let mut full = members.clone();
                let kept = full_search_with_restarts(&eng, &mut full, &config);
                assert_eq!((stats.restarts_kept, stats.accepts), (kept.restarts, kept.moves), "{label}");
                assert_same_pool_outcome(&searched, &full, &label);
                by_their_random_moves += kept.by_their_random_moves;
                pools += 1;
            }
        }
    }
    assert!(by_their_random_moves > 0, "no restart kept for its random moves alone over {pools} pools");
}

#[test]
fn restarts_short_of_their_minimum_gain_leave_the_descent_as_it_ended() {
    let eng = crate::Engine::new();
    let mut plain = pool();
    let descent = PoolSearchConfig { max_sweeps: usize::MAX, ..PoolSearchConfig::default() };
    let alone = with_members(&mut plain, |refs| eng.pool_search(refs, &descent)).unwrap();
    let mut config = with_restarts();
    config.restarts.min_gain_permille = 1000;
    let mut restarted = pool();
    let stats = with_members(&mut restarted, |refs| eng.pool_search(refs, &config)).unwrap();
    assert_eq!((stats.restarts, stats.restarts_kept), (3, 0), "{stats:?}");
    assert_eq!((stats.accepts, stats.pairs_after), (alone.accepts, alone.pairs_after), "{stats:?} against {alone:?}");
    assert_eq!(restarted[0].vtree().to_text(), plain[0].vtree().to_text());
    for (r, p) in restarted.iter().zip(&plain) {
        assert_eq!(r.pair_count(), p.pair_count());
        assert_canonical(r);
    }
}

#[test]
fn skipping_refused_moves_keeps_the_moves_of_the_full_descent() {
    let eng = crate::Engine::new();
    for crossed in [false, true] {
        let config = PoolSearchConfig { max_sweeps: 6, crossed, ..PoolSearchConfig::default() };
        let mut searched = pool();
        let stats = with_members(&mut searched, |refs| eng.pool_search(refs, &config)).unwrap();
        let mut full = pool();
        let (accepts, probes) = full_descent(&eng, &mut full, &config);
        assert_eq!(stats.accepts, accepts, "crossed {crossed}");
        // The pool takes more than one sweep, and the later ones skip.
        assert!(stats.sweeps > 1 && stats.probes < probes, "crossed {crossed}: {stats:?} against {probes} probes");
        assert_eq!(searched[0].vtree().to_text(), full[0].vtree().to_text(), "crossed {crossed}");
        for (s, f) in searched.iter().zip(&full) {
            assert_eq!(s.pair_count(), f.pair_count(), "crossed {crossed}");
        }
    }
}

/// Seeded pools of four functions on every vtree shape of six to nine
/// variables: the descent that skips the moves it refused since their levels
/// last changed keeps the moves of the descent that probes every one.
#[test]
fn skipping_refused_moves_keeps_the_moves_of_the_full_descent_on_seeded_pools() {
    use crate::test_helpers::{vtree_shapes, CnfShape, Lcg};
    let eng = crate::Engine::new();
    let mut rng = Lcg::new(0x9e11);
    let config = PoolSearchConfig { max_sweeps: usize::MAX, ..PoolSearchConfig::default() };
    let (mut kept, mut skipped) = (0, 0);
    for n in 6..10 {
        for (shape, vtree) in vtree_shapes(n) {
            let members = seeded_pool(&eng, &mut rng, &vtree, n, CnfShape { clauses: n as usize, width: 3 });
            let mut searched = members.clone();
            let stats = with_members(&mut searched, |refs| eng.pool_search(refs, &config)).unwrap();
            let mut full = members;
            let (accepts, probes) = full_descent(&eng, &mut full, &config);
            assert_eq!(stats.accepts, accepts, "{n} variables, {shape}");
            assert_eq!(searched[0].home().to_text(), full[0].home().to_text(), "{n} variables, {shape}");
            for (s, f) in searched.iter().zip(&full) {
                assert_eq!(s.pair_count(), f.pair_count(), "{n} variables, {shape}");
            }
            kept += accepts;
            skipped += probes - stats.probes;
        }
    }
    assert!(kept > 0 && skipped > 0, "{kept} kept, {skipped} skipped");
}

/// Seeded pools of four functions on every vtree shape of six to nine
/// variables, under budgets that stop the search part way: the descent that
/// charges a cached refusal instead of probing the move again keeps,
/// probes and charges exactly what the descent that probes it does.
#[test]
fn a_replayed_refusal_decides_and_charges_as_probing_again() {
    use crate::test_helpers::{vtree_shapes, CnfShape, Lcg};
    let eng = crate::Engine::new();
    let mut rng = Lcg::new(0x51ee);
    let mut replayed = 0;
    for n in 6..10 {
        for (shape, vtree) in vtree_shapes(n) {
            let members = seeded_pool(&eng, &mut rng, &vtree, n, CnfShape { clauses: n as usize, width: 3 });
            for budget in [u64::MAX, 4000, 1500, 600] {
                let config = PoolSearchConfig { max_sweeps: usize::MAX, max_work_units: budget, ..PoolSearchConfig::default() };
                let run = |replay: bool| {
                    REPLAY_REFUSALS.with(|r| r.set(replay));
                    let mut searched = members.clone();
                    let before = REPLAYED.with(std::cell::Cell::get);
                    let stats = with_members(&mut searched, |refs| eng.pool_search(refs, &config)).unwrap();
                    let n = REPLAYED.with(std::cell::Cell::get) - before;
                    REPLAY_REFUSALS.with(|r| r.set(true));
                    (stats, searched, n)
                };
                let (replaying, with, n) = run(true);
                let (probing, without, _) = run(false);
                let label = format!("{n} variables, {shape}, budget {budget}");
                assert_eq!(replaying, probing, "{label}");
                assert_eq!(with[0].home().to_text(), without[0].home().to_text(), "{label}");
                for (a, b) in with.iter().zip(&without) {
                    assert_eq!(a.pair_count(), b.pair_count(), "{label}");
                }
                replayed += n;
            }
        }
    }
    assert!(replayed > 0);
}

/// The pool search's decisions, pinned: on seeded pools of every vtree shape
/// of six to ten variables, four random functions with a literal, the
/// constant and a two-literal clause beside them, under plain, bounded,
/// budgeted and restarting searches, the moves kept in order, the
/// statistics (work units included), the final home and every member's
/// pairs hash to a pinned value. A change that moves this hash changes which
/// moves the search makes or what it charges.
#[test]
fn a_pool_search_makes_the_moves_and_charges_it_made_before() {
    use crate::test_helpers::{compile_clauses_on, vtree_shapes, CnfShape, Lcg};
    const PINNED: u64 = 0x023b_7d6d_c8ba_ff6c;
    let eng = crate::Engine::new();
    let mut rng = Lcg::new(0x9001);
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut fold = |bytes: &[u8]| {
        for &b in bytes {
            hash = (hash ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    };
    let mut accepted = 0usize;
    for n in 6..11u32 {
        for (_, vtree) in vtree_shapes(n) {
            let mut members = seeded_pool(&eng, &mut rng, &vtree, n, CnfShape { clauses: n as usize, width: 3 });
            for clauses in [vec![vec![2]], vec![], vec![vec![1, -(n as i32)]]] {
                let mut f = compile_clauses_on(&eng, &vtree, &clauses);
                f.minimize().expect("an unarmed engine refuses nothing");
                members.push(f);
            }
            let configs = [
                PoolSearchConfig::default(),
                PoolSearchConfig { max_sweeps: usize::MAX, crossed: false, ..PoolSearchConfig::default() },
                PoolSearchConfig { max_inner_pairs: 12, ..PoolSearchConfig::default() },
                PoolSearchConfig { max_work_units: 2500, ..PoolSearchConfig::default() },
                PoolSearchConfig {
                    max_sweeps: usize::MAX,
                    restarts: PoolRestarts { count: 3, kick: 3, slack_permille: 200, seed: u64::from(n), min_gain_permille: 0 },
                    ..PoolSearchConfig::default()
                },
            ];
            for config in &configs {
                ACCEPTED.with(|a| a.borrow_mut().clear());
                let mut searched = members.clone();
                let stats = with_members(&mut searched, |refs| eng.pool_search(refs, config)).unwrap();
                ACCEPTED.with(|a| {
                    for mv in a.borrow().iter() {
                        let kind = u8::from(mv.rotation.kind == RotationKind::Right);
                        fold(&mv.rotation.pivot.0.to_le_bytes());
                        fold(&[kind, u8::from(mv.crossed)]);
                    }
                    accepted += a.borrow().len();
                });
                fold(format!("{stats:?}").as_bytes());
                fold(searched[0].home().to_text().as_bytes());
                for m in &searched {
                    fold(&m.pair_count().to_le_bytes());
                }
            }
        }
    }
    assert!(accepted > 0);
    assert_eq!(hash, PINNED, "hash {hash:#x} over {accepted} kept moves");
}

#[path = "pool/home.rs"]
mod home;
