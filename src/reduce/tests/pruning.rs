//! Pruning.
//!
//! The fixtures these read are in `mod.rs`.

use std::sync::Arc;

use crate::reduce::prune::{PruneScope, Select, below_root_walk_applies, prune_unreachable, settle_loose};

use crate::test_helpers::compile_clauses;
use crate::vtree::Vtree;


/// The prune's one diagram-proportional scratch buffer is reserved through the
/// engine, so both halves of the tracked reserve reach it: the allocation-
/// failure injection refuses it, and a granted reservation is charged to the
/// in-flight byte meter.
#[test]
fn the_prune_scratch_reservation_goes_through_the_engine() {
    let vtree = Arc::new(Vtree::balanced(4));
    let clauses = [vec![1, 2], vec![-2, 3], vec![3, -4]];

    // Refused: the diagram is left exactly as it was.
    let eng = &crate::Engine::new();
    let mut tdd = compile_clauses(&vtree, &clauses);
    tdd.minimize().unwrap();
    let (size_before, mc_before) = (tdd.pair_count(), tdd.model_count().unwrap());
    eng.limits().refuse_nth_reserve(0);
    let refused = prune_unreachable(eng, &mut tdd, PruneScope::Whole);
    eng.limits().grant_every_reserve();
    assert!(refused.is_err(), "the armed injection must refuse the scratch reservation");
    assert_eq!(tdd.pair_count(), size_before, "a refused prune must not touch the diagram");
    assert_eq!(tdd.model_count().unwrap(), mc_before, "a refused prune must not touch the count");

    // Granted: the bytes the scratch takes are charged.
    let eng = &crate::Engine::new();
    let mut tdd = compile_clauses(&vtree, &clauses);
    tdd.minimize().unwrap();
    eng.limits().reset_meters();
    prune_unreachable(eng, &mut tdd, PruneScope::Whole).expect("a tiny scratch reservation cannot fail");
    assert!(
        eng.limits().meters().in_flight_bytes > 0,
        "the scratch reservation must be charged against the byte budget",
    );
}

/// The nodes of each structural level the output reaches, by a walk that
/// shares nothing with the prune: index `t` holds level `t`'s reached slots.
fn reached_by_level(tdd: &crate::Tdd) -> Vec<std::collections::BTreeSet<usize>> {
    let vtree = &tdd.vtree;
    let mut reached = vec![std::collections::BTreeSet::new(); vtree.num_nodes()];
    reached[tdd.output.vtree.idx()].insert(tdd.output.local.idx());
    for &t in vtree.bottomup_slice().iter().rev() {
        if !tdd.is_structural_internal(t) {
            continue;
        }
        let (left, right) = vtree.children(t);
        let (lv, rv) = (tdd.levels[left.idx()].child_decoder(), tdd.levels[right.idx()].child_decoder());
        let nodes: Vec<usize> = reached[t.idx()].iter().copied().collect();
        for i in nodes {
            for pair in tdd.levels[t.idx()].pairs_vec(i) {
                if let Some(s) = lv.child(pair.left).index() {
                    reached[left.idx()].insert(s);
                }
                if let Some(s) = rv.child(pair.right).index() {
                    reached[right.idx()].insert(s);
                }
            }
        }
    }
    reached
}

/// Every node's pairs, level by level, and the output: two diagrams agree
/// here iff they are the same diagram.
fn layout(tdd: &crate::Tdd) -> Vec<Vec<Vec<(u32, u32)>>> {
    let mut out = vec![vec![vec![(tdd.output.vtree.0, tdd.output.local.0)]]];
    for level in &tdd.levels {
        if level.is_marginal() {
            continue;
        }
        out.push(
            (0..level.slot_count())
                .map(|i| level.pairs_vec(i).iter().map(|p| (p.left.0, p.right.0)).collect())
                .collect(),
        );
    }
    out
}

/// A conjunction leaves nodes no parent names, on levels wide enough that
/// their marks take several words. The walk down from the root through the
/// levels the conjunction built and the walk over the whole diagram leave the
/// same diagram, and it holds exactly the nodes an independent walk reaches.
#[test]
fn the_walk_from_the_root_and_the_whole_walk_keep_exactly_the_reached_nodes() {
    use crate::test_helpers::{CnfShape, Lcg, compile_clauses_on, rand_cnf, vtree_shapes};

    let eng = &crate::Engine::new();
    let mut rng = Lcg::new(0x7b31_c4e9);
    let (mut cases, mut seeded, mut wide_losses) = (0usize, 0usize, 0usize);
    for shape in [CnfShape { clauses: 16, width: 6 }, CnfShape { clauses: 32, width: 8 }] {
        for (_name, vtree) in vtree_shapes(20) {
            for _ in 0..6 {
                let ca = rand_cnf(&mut rng, 20, shape);
                let cb = rand_cnf(&mut rng, 20, shape);
                let f = compile_clauses_on(eng, &vtree, &ca);
                let g = compile_clauses_on(eng, &vtree, &cb);
                if f.is_zero() || g.is_zero() {
                    continue;
                }
                let conj = crate::and(f, g).unwrap();
                if conj.is_zero() || conj.dirty.loose().is_none() || conj.has_marginal_level() {
                    continue;
                }
                let reached = reached_by_level(&conj);
                for &t in conj.vtree.bottomup_slice() {
                    let width = conj.levels[t.idx()].slot_count();
                    if conj.is_structural_internal(t) && width > 64 && reached[t.idx()].len() < width {
                        wide_losses += 1;
                    }
                }
                let count = conj.model_count().unwrap();
                seeded += usize::from(below_root_walk_applies(&conj));

                let mut walked = conj.clone();
                prune_unreachable(eng, &mut walked, PruneScope::Whole).unwrap();
                let mut whole = conj;
                whole.dirty.set_loose(None);
                prune_unreachable(eng, &mut whole, PruneScope::Whole).unwrap();

                assert_eq!(layout(&walked), layout(&whole), "the two walks left different diagrams");
                for &t in walked.vtree.bottomup_slice() {
                    if walked.is_structural_internal(t) {
                        assert_eq!(
                            walked.levels[t.idx()].slot_count(),
                            reached[t.idx()].len(),
                            "level {} keeps other than the nodes the output reaches",
                            t.0
                        );
                    }
                }
                assert_eq!(walked.model_count().unwrap(), count);
                cases += 1;
            }
        }
    }
    assert!(cases > 40, "expected a corpus, got {cases} cases");
    assert!(seeded > 40, "expected the walk from the root on most cases, got {seeded}");
    assert!(wide_losses > 30, "expected losses on levels of several words, got {wide_losses}");
}

/// The internal levels below the root of `tdd` that hold a node no pair of
/// their parent level names, in order, by a read of every level that shares
/// nothing with the conjunction.
fn unnamed_levels(tdd: &crate::Tdd) -> Vec<u32> {
    let vtree = &tdd.vtree;
    let mut out = Vec::new();
    for (p, left, right) in vtree.internal_bottomup() {
        if !tdd.is_structural_internal(p) {
            continue;
        }
        for (c, on_left) in [(left, true), (right, false)] {
            if !tdd.is_structural_internal(c) {
                continue;
            }
            let view = tdd.levels[c.idx()].child_decoder();
            let parent = &tdd.levels[p.idx()];
            let named: std::collections::BTreeSet<usize> = (0..parent.slot_count())
                .flat_map(|i| parent.pairs_vec(i))
                .filter_map(|pair| view.child(if on_left { pair.left } else { pair.right }).index())
                .collect();
            if named.len() < tdd.levels[c.idx()].slot_count() {
                out.push(c.0);
            }
        }
    }
    out.sort_unstable();
    out
}

/// A random few of the variables of `wide`, sorted, and `wide` restricted to
/// them, which numbers them from 1 in that order.
fn restricted(rng: &mut crate::test_helpers::Lcg, wide: &Arc<Vtree>) -> (Vec<u32>, Arc<Vtree>) {
    use crate::vtree::VarId;
    let n = wide.num_vars();
    let k = 2 + rng.below(u64::from(n) - 2) as u32;
    let mut vars: Vec<u32> = (1..=n).collect();
    for i in 0..k as usize {
        let j = i + rng.below(u64::from(n) - i as u64) as usize;
        vars.swap(i, j);
    }
    let mut vars = vars[..k as usize].to_vec();
    vars.sort_unstable();
    let local = |v: VarId| vars.binary_search(&v.0).ok().map(|i| VarId(i as u32 + 1));
    let small = Arc::new(wide.project_to_vars(local, k).expect("k variables are kept"));
    (vars, small)
}

/// A diagram over a random few of the variables of `wide`, compiled on
/// `wide` restricted to them and minimized, then placed on `wide` by
/// `embed_moving`: an operand as the evidence walk makes them.
fn placed(eng: &crate::Engine, rng: &mut crate::test_helpers::Lcg, wide: &Arc<Vtree>) -> crate::Tdd {
    use crate::test_helpers::{CnfShape, compile_clauses_on, rand_cnf};
    use crate::vtree::VarId;
    let (vars, small) = restricted(rng, wide);
    let k = vars.len() as u32;
    let mut f = compile_clauses_on(eng, &small, &rand_cnf(rng, k, CnfShape { clauses: k as usize, width: 4 }));
    eng.minimize(&mut f).unwrap();
    eng.embed_moving(f, wide, |v| VarId(vars[v.idx()])).map_err(|r| r.error).unwrap().0
}

/// An embedding lists every level of its result that holds a node no pair
/// of its parent level names, where the source has such levels of its own:
/// the conjunction of two diagrams on a few of the variables, not pruned,
/// placed on the whole vtree. The pass-through chains the embedding builds
/// carry the looseness of the level at their foot up to their top.
#[test]
fn an_embedding_lists_the_levels_holding_an_unnamed_node() {
    use crate::test_helpers::{CnfShape, Lcg, compile_clauses_on, rand_cnf, vtree_shapes};
    use crate::vtree::VarId;

    let eng = &crate::Engine::new();
    let mut rng = Lcg::new(0x51ab_e70f);
    let (mut cases, mut carried) = (0usize, 0usize);
    for (name, wide) in vtree_shapes(14) {
        for case in 0..40 {
            let (vars, small) = restricted(&mut rng, &wide);
            let k = vars.len() as u32;
            let [mut f, mut g] = [(); 2].map(|()| compile_clauses_on(eng, &small, &rand_cnf(&mut rng, k, CnfShape { clauses: k as usize, width: 3 })));
            eng.minimize(&mut f).unwrap();
            eng.minimize(&mut g).unwrap();
            let source = eng.and(f, g).unwrap();
            if source.is_zero() {
                continue;
            }
            let what = format!("{name}, case {case}");
            let unnamed_source = unnamed_levels(&source);
            let count = source.model_count().unwrap();
            let (moved, _) = eng.embed_moving(source, &wide, |v| VarId(vars[v.idx()])).map_err(|r| r.error).unwrap();
            let listed = moved.dirty.loose().expect("the source's loose levels are known");
            for t in unnamed_levels(&moved) {
                assert!(listed.contains(&t), "{what}: level {t} holds an unnamed node and is not listed");
            }
            assert_eq!(moved.model_count().unwrap(), count << (wide.num_vars() - k), "{what}");
            carried += unnamed_source.len();
            cases += 1;
        }
    }
    assert!(cases > 150, "expected a corpus, got {cases} cases");
    assert!(carried > 50, "expected sources with levels holding an unnamed node, got {carried}");
}

/// The root of `tdd` and every level above one of `levels`: the levels the
/// prune enters whatever it finds, with `levels` its loose levels as
/// `settle_loose` leaves them.
fn entered(tdd: &crate::Tdd, levels: &[u32]) -> std::collections::BTreeSet<u32> {
    let mut out = std::collections::BTreeSet::from([tdd.vtree.root().0]);
    for &t in levels {
        let mut up = tdd.vtree.node(crate::vtree::VtreeIdx(t)).parent();
        while let Some(p) = up {
            out.insert(p.0);
            up = tdd.vtree.node(p).parent();
        }
    }
    out
}

/// Operands placed on a wider vtree are conjoined, by `and` and by
/// `and_restoring`, and the result again with a third without a prune in
/// between. Each result lists as loose every level that holds a node no
/// pair of its parent level names; the prune settles the list to one that
/// has it enter the levels a list of exactly those would; and the prune
/// leaves the diagram the prune over the whole diagram leaves, node for
/// node.
#[test]
fn a_conjunction_lists_as_loose_the_levels_holding_an_unnamed_node() {
    use crate::test_helpers::{Lcg, vtree_shapes};

    let eng = &crate::Engine::new();
    let mut rng = Lcg::new(0x9e1f_2d47);
    let (mut cases, mut loose, mut cleared, mut settled_away) = (0usize, 0usize, 0usize, 0usize);
    let mut check = |conj: crate::Tdd, operands: [&[u32]; 2], what: &str| {
        if conj.is_zero() {
            return;
        }
        let mut listed = conj.dirty.loose().expect("both operands' loose levels are known").to_vec();
        listed.sort_unstable();
        listed.dedup();
        let unnamed = unnamed_levels(&conj);
        for t in &unnamed {
            assert!(listed.binary_search(t).is_ok(), "{what}: level {t} holds an unnamed node and is not listed");
        }
        let settled = settle_loose(eng, &conj, &listed).unwrap();
        assert_eq!(entered(&conj, &settled), entered(&conj, &unnamed), "{what}: the levels the prune enters");
        loose += unnamed.len();
        settled_away += listed.len() - settled.len();
        cleared += operands.concat().iter().filter(|&&t| {
            let t = crate::vtree::VtreeIdx(t);
            conj.is_structural_internal(t) && t != conj.vtree.root() && listed.binary_search(&t.0).is_err()
        }).count();
        let count = conj.model_count().unwrap();
        let mut walked = conj.clone();
        prune_unreachable(eng, &mut walked, PruneScope::Whole).unwrap();
        let mut whole = conj;
        whole.dirty.set_loose(None);
        prune_unreachable(eng, &mut whole, PruneScope::Whole).unwrap();
        assert_eq!(layout(&walked), layout(&whole), "{what}: the two prunes left different diagrams");
        assert_eq!(walked.model_count().unwrap(), count, "{what}");
    };
    for (name, wide) in vtree_shapes(14) {
        for case in 0..40 {
            let (f, g, h) = (placed(eng, &mut rng, &wide), placed(eng, &mut rng, &wide), placed(eng, &mut rng, &wide));
            if f.is_zero() || g.is_zero() || h.is_zero() {
                continue;
            }
            let [lf, lg, lh] = [&f, &g, &h].map(|d| d.dirty.loose().expect("an embedding knows its loose levels").to_vec());
            let what = format!("{name}, case {case}");
            for (d, listed) in [(&f, &lf), (&g, &lg), (&h, &lh)] {
                for t in unnamed_levels(d) {
                    assert!(listed.contains(&t), "{what}: the embedding leaves level {t} with an unnamed node unlisted");
                }
            }
            let both = eng.and(f.clone(), g.clone()).unwrap();
            let kept = eng.and_restoring(f, g).map_err(|r| r.error).unwrap();
            assert_eq!(layout(&both), layout(&kept), "{what}: and and and_restoring differ");
            let lb = both.dirty.loose().unwrap().to_vec();
            check(kept, [&lf, &lg], &format!("{what}, f and g"));
            check(eng.and(both, h).unwrap(), [&lb, &lh], &format!("{what}, then h"));
            cases += 1;
        }
    }
    assert!(cases > 180, "expected a corpus, got {cases} cases");
    assert!(loose > 300, "expected levels holding an unnamed node, got {loose}");
    assert!(cleared > 300, "expected levels an operand listed and the result does not, got {cleared}");
    assert!(settled_away > 400, "expected listed levels the prune settles away, got {settled_away}");
}

/// The rank select gives the marked slot of every rank, read in rising
/// order, again, and out of order, over blocks of no marks, every mark and
/// random marks, and nothing past the last mark.
#[test]
fn the_rank_select_reads_every_marked_slot() {
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let mut blocks = vec![vec![0u64; 3], vec![u64::MAX; 3], vec![1u64 << 63, 0, 1]];
    for _ in 0..20 {
        let words = 1 + (next() % 5) as usize;
        blocks.push((0..words).map(|_| next() & next()).collect());
    }
    for block in &blocks {
        let slots: Vec<usize> = (0..block.len() * 64).filter(|&s| block[s >> 6] >> (s & 63) & 1 != 0).collect();
        let mut select = Select::new(block);
        for (j, &s) in slots.iter().enumerate() {
            assert_eq!(select.nth(j), Some(s));
            assert_eq!(select.nth(j), Some(s), "the same rank again");
        }
        assert_eq!(select.nth(slots.len()), None);
        for _ in 0..50 {
            let j = (next() % (slots.len() as u64 + 2)) as usize;
            assert_eq!(select.nth(j), slots.get(j).copied(), "rank {j} out of order");
        }
    }
}
