//! Evaluation through marginal levels: an algebra that values counts reads a
//! summed-out diagram as it reads the diagram before the sum, wherever it
//! values the summed variables as the identity.

use super::*;
use std::cell::Cell;
use num_traits::ToPrimitive;
use crate::diagram::{ChildRef, ColumnAlgebra, EvalAlgebra, LeafLabel, SlotPairs, ValueRef};
use crate::test_helpers::{assert_canonical, assert_marginal_canonical, inline_marginal, marginal_boundary, under};
use crate::vtree::rng::Lcg;
use crate::OperationError;

/// The modulus every value is reduced by: a prime, so that a product taken
/// with the wrong partner or a count read at the wrong slot shows.
const P: u64 = (1 << 61) - 1;

fn add(a: u64, b: u64) -> u64 {
    ((u128::from(a) + u128::from(b)) % u128::from(P)) as u64
}

fn mul(a: u64, b: u64) -> u64 {
    (u128::from(a) * u128::from(b) % u128::from(P)) as u64
}

/// Weighted model counts modulo `P`: a weight pair of its own per variable,
/// but the identity for the variables a marginalization sums out.
/// `regroup` makes the fold sum the products of the pairs that share a right
/// value by that value's address, as an algebra whose product is costly does.
struct Weighted {
    pos: Vec<u64>,
    neg: Vec<u64>,
    regroup: bool,
    counts: Cell<usize>,
}

impl Weighted {
    /// Weights drawn for variables `1..=vars`, `1` for each one in `summed`.
    fn new(rng: &mut Lcg, vars: u32, summed: &[VarId], regroup: bool) -> Self {
        let mut pos = vec![0; vars as usize + 1];
        let mut neg = vec![0; vars as usize + 1];
        for v in 1..=vars as usize {
            (pos[v], neg[v]) = match summed.contains(&VarId(v as u32)) {
                true => (1, 1),
                false => (2 + rng.next_u64() % (P - 2), 2 + rng.next_u64() % (P - 2)),
            };
        }
        Weighted { pos, neg, regroup, counts: Cell::new(0) }
    }

    fn weight(&self, var: VarId, label: LeafLabel) -> u64 {
        let v = var.0 as usize;
        match label {
            LeafLabel::Zero => 0,
            LeafLabel::Pos => self.pos[v],
            LeafLabel::Neg => self.neg[v],
            LeafLabel::One => add(self.pos[v], self.neg[v]),
        }
    }

    fn count_of(&self, n: &BigUint) -> u64 {
        self.counts.set(self.counts.get() + 1);
        (n % BigUint::from(P)).to_u64().expect("reduced below P")
    }
}

impl EvalAlgebra for Weighted {
    type Value = u64;
    fn zero(&self) -> u64 { 0 }
    fn leaf(&self, var: VarId, label: LeafLabel) -> u64 { self.weight(var, label) }
    fn add_assign(&self, a: &mut u64, b: &u64) { *a = add(*a, *b); }
    fn mul(&self, a: &u64, b: &u64) -> u64 { mul(*a, *b) }
    fn sum_of_products<'v>(&self, pairs: impl ExactSizeIterator<Item = (&'v u64, &'v u64)>) -> u64 {
        if !self.regroup {
            return pairs.fold(0, |acc, (a, b)| add(acc, mul(*a, *b)));
        }
        let mut by_right: Vec<(&u64, u64)> = Vec::new();
        for (a, b) in pairs {
            match by_right.iter_mut().find(|(right, _)| std::ptr::eq(*right, b)) {
                Some((_, sum)) => *sum = add(*sum, *a),
                None => by_right.push((b, *a)),
            }
        }
        by_right.iter().fold(0, |acc, &(b, sum)| add(acc, mul(sum, *b)))
    }
    fn count(&self, n: &BigUint) -> Option<u64> {
        Some(self.count_of(n))
    }
}

/// [`Weighted`] in flat columns, every slot first set to `u64::MAX` (no
/// value below `P` is) so that a read of a slot nobody wrote shows.
struct Columns<'a>(&'a Weighted);

impl ColumnAlgebra for Columns<'_> {
    type Column = Vec<u64>;
    type Value = u64;
    fn zero(&self) -> u64 { 0 }
    fn column(&self, _: VtreeIdx, width: usize) -> Vec<u64> { vec![u64::MAX; width] }
    fn leaf(&self, _: VtreeIdx, var: VarId, label: LeafLabel, col: &mut Vec<u64>) {
        col[label as usize] = self.0.weight(var, label);
    }
    fn fold(&self, _: VtreeIdx, slot: usize, pairs: SlotPairs<'_>, left: &Vec<u64>, right: &Vec<u64>, out: &mut Vec<u64>) {
        out[slot] = pairs.fold(0, |acc, (l, r)| {
            assert!(left[l] != u64::MAX && right[r] != u64::MAX, "a pair reads a slot nobody wrote");
            add(acc, mul(left[l], right[r]))
        });
    }
    fn read(&self, _: VtreeIdx, col: Vec<u64>, slot: usize) -> u64 { col[slot] }
    fn count(&self, _: VtreeIdx, slot: usize, n: &BigUint, col: &mut Vec<u64>) -> bool {
        col[slot] = self.0.count_of(n);
        true
    }
}

/// The variables under any of `targets`, each once.
fn summed_vars(vtree: &Vtree, targets: &[VtreeIdx]) -> Vec<VarId> {
    let mut vars: Vec<VarId> = (0..vtree.num_nodes() as u32)
        .map(VtreeIdx)
        .filter(|&t| vtree.node(t).is_leaf() && targets.iter().any(|&s| under(vtree, t, s)))
        .map(|t| vtree.leaf_var(t))
        .collect();
    vars.sort_unstable();
    vars
}

/// How a diagram's structural levels reach their marginal children: the
/// references carrying a count inline, and those naming a stored slot.
#[derive(Default)]
struct Reach {
    inline: usize,
    slots: usize,
}

fn reach(f: &Tdd) -> Reach {
    let vtree = f.vtree();
    let mut seen = Reach::default();
    for t in (0..vtree.num_nodes() as u32).map(VtreeIdx) {
        if vtree.node(t).is_leaf() || f.level(t).is_marginal() { continue; }
        let (l, r) = vtree.children(t);
        let (lv, rv) = (f.level(l).child_decoder(), f.level(r).child_decoder());
        for (_, pairs) in f.level(t).internal_inputs_iter() {
            for p in pairs {
                for c in [lv.child(p.left), rv.child(p.right)] {
                    match c {
                        ChildRef::Value(ValueRef::Inline(_)) => seen.inline += 1,
                        ChildRef::Value(ValueRef::Slot(_)) => seen.slots += 1,
                        ChildRef::Node(_) => {}
                    }
                }
            }
        }
    }
    seen
}

/// Every level's [`Engine::evaluate_at`] against its children's: a marginal
/// level's slot is its count's value, a structural one's the sum over its
/// pairs of the products of its children's values, an inline side read as
/// its count's; the columns of [`Engine::evaluate_columns_at`] agree slot for
/// slot.
fn check_levels(eng: &Engine, m: &Tdd, w: &Weighted, context: &str) {
    let vtree = m.vtree();
    let at: Vec<Vec<u64>> = (0..vtree.num_nodes() as u32)
        .map(|t| eng.evaluate_at(m, VtreeIdx(t), w).unwrap())
        .collect();
    for t in (0..vtree.num_nodes() as u32).map(VtreeIdx) {
        let values = &at[t.idx()];
        assert_eq!(values.len(), m.reference_slot_count(t), "{context}: level {}", t.idx());
        let column = eng.evaluate_columns_at(m, t, &Columns(w)).unwrap();
        assert_eq!(&column, values, "{context}: the columns at level {}", t.idx());
        if vtree.node(t).is_leaf() { continue; }
        let level = m.level(t);
        if let Some(counts) = level.marginal_counts() {
            for (i, &n) in counts.iter().enumerate() {
                let n = match n {
                    u128::MAX => level.marginal_counts_big().and_then(|b| b.get(i)).unwrap().clone(),
                    n => BigUint::from(n),
                };
                assert_eq!(values[i], w.count_of(&n), "{context}: marginal level {}, slot {i}", t.idx());
            }
            continue;
        }
        let (l, r) = vtree.children(t);
        let side = |c: VtreeIdx, side| match m.level(c).child_decoder().child(side) {
            ChildRef::Value(ValueRef::Inline(n)) => w.count_of(&BigUint::from(n)),
            ChildRef::Value(ValueRef::Slot(i)) | ChildRef::Node(crate::diagram::NodeIdx(i)) => at[c.idx()][i as usize],
        };
        for (i, pairs) in level.internal_inputs_iter() {
            let expected = pairs.fold(0, |acc, p| add(acc, mul(side(l, p.left), side(r, p.right))));
            assert_eq!(values[i], expected, "{context}: level {}, node {i}", t.idx());
        }
    }
}

#[test]
fn a_marginal_diagram_evaluates_as_its_summed_structure_does() {
    let eng = Engine::new();
    let mut rng = Lcg::new(41);
    let vars: Vec<VarId> = (1..=12).map(VarId).collect();
    let (mut inline, mut slots, mut cases) = (0, 0, 0);
    for tree in [Vtree::balanced(14), Vtree::linear(14), Vtree::random(14, 5), Vtree::random(14, 23)] {
        let vtree = Arc::new(tree);
        for round in 0..12u64 {
            let rows: Vec<u64> = (0..60 + 40 * round)
                .map(|_| (rng.next_u64() & 0x3ff) | (rng.next_u64() % 3) << 10)
                .collect();
            let f = eng.from_models(&vtree, &vars, &rows).unwrap();
            assert_canonical(&f);
            // A few subtrees summed out, the root among them now and then.
            let mut targets: Vec<VtreeIdx> = (0..vtree.num_nodes() as u32)
                .map(VtreeIdx)
                .filter(|&t| t != vtree.root() && rng.next_u64().is_multiple_of(5))
                .collect();
            if round % 6 == 5 { targets.push(vtree.root()); }
            let summed = summed_vars(&vtree, &targets);
            for regroup in [false, true] {
                let w = Weighted::new(&mut rng, 14, &summed, regroup);
                let expected = eng.evaluate(&f, &w).unwrap();
                let mut m = f.clone();
                eng.marginalize_levels(&mut m, &targets).unwrap();
                // As the pass leaves it, and minimized.
                for minimized in [false, true] {
                    if minimized {
                        eng.minimize(&mut m).unwrap();
                        assert_marginal_canonical(&m);
                    }
                    let context = format!("round {round}, targets {targets:?}, regroup {regroup}, minimized {minimized}");
                    assert_eq!(eng.evaluate(&m, &w).unwrap(), expected, "{context}");
                    assert_eq!(eng.evaluate_columns(&m, &Columns(&w)).unwrap(), expected, "{context}");
                    check_levels(&eng, &m, &w, &context);
                    let seen = reach(&m);
                    (inline, slots, cases) = (inline + seen.inline, slots + seen.slots, cases + 1);
                }
            }
        }
    }
    // Counts this small ride inline; the stored slots are the cases below.
    assert!(cases > 0 && inline > 0, "inline {inline} and slot {slots} references were read");
}

#[test]
fn a_marginalizing_conjunction_evaluates_as_the_plain_one() {
    let eng = Engine::new();
    let mut rng = Lcg::new(43);
    let (low, high): (Vec<VarId>, Vec<VarId>) = ((1..=8).map(VarId).collect(), (5..=12).map(VarId).collect());
    let mut marginal = 0;
    for tree in [Vtree::balanced(13), Vtree::linear(13), Vtree::random(13, 7), Vtree::random(13, 19)] {
        let vtree = Arc::new(tree);
        for round in 0..10u64 {
            let draw = |rng: &mut Lcg| -> Vec<u64> { (0..30 + 20 * round).map(|_| rng.next_u64() & 0xff).collect() };
            let g = eng.from_models(&vtree, &low, &draw(&mut rng)).unwrap();
            let h = eng.from_models(&vtree, &high, &draw(&mut rng)).unwrap();
            let both = eng.and(g.clone(), h.clone()).unwrap();
            // One subtree summed during the product, or several.
            let inner: Vec<VtreeIdx> = (0..vtree.num_nodes() as u32).map(VtreeIdx).filter(|&t| t != vtree.root()).collect();
            let targets: Vec<VtreeIdx> = match round % 2 {
                0 => vec![inner[rng.next_u64() as usize % inner.len()]],
                _ => inner.iter().copied().filter(|_| rng.next_u64().is_multiple_of(4)).collect(),
            };
            let summed = summed_vars(&vtree, &targets);
            let w = Weighted::new(&mut rng, 13, &summed, round % 3 == 0);
            let m = eng.and_marginalizing(g, h, &targets).unwrap();
            let context = format!("round {round}, targets {targets:?}");
            let expected = eng.evaluate(&both, &w).unwrap();
            assert_eq!(eng.evaluate(&m, &w).unwrap(), expected, "{context}");
            assert_eq!(eng.evaluate_columns(&m, &Columns(&w)).unwrap(), expected, "{context}");
            check_levels(&eng, &m, &w, &context);
            marginal += usize::from(m.levels().iter().any(|l| l.is_marginal()));
        }
    }
    assert!(marginal > 0, "some conjunction kept a marginal level");
}

#[test]
fn a_count_past_u128_is_valued_exactly() {
    // On the right-linear vtree the root's right child spans x2..x140, which
    // `x2 ∨ x3` leaves 3 · 2^137 models: past `u128`, so the level stores
    // the count in its overflow table.
    let eng = Engine::new();
    let mut rng = Lcg::new(47);
    let vtree = Arc::new(Vtree::linear(140));
    let f = eng.and(Tdd::clause(&vtree, [1]).unwrap(), Tdd::clause(&vtree, [2, 3]).unwrap()).unwrap();
    let wide = eng.or(f.clone(), Tdd::clause(&vtree, [-1, -2]).unwrap()).unwrap();
    let right = vtree.children(vtree.root()).1;
    let summed = summed_vars(&vtree, &[right]);
    assert_eq!(summed.len(), 139);
    for g in [f, wide] {
        let w = Weighted::new(&mut rng, 140, &summed, false);
        let expected = eng.evaluate(&g, &w).unwrap();
        let mut m = g;
        eng.marginalize_levels(&mut m, &[right]).unwrap();
        assert!(m.level(right).marginal_counts().unwrap().contains(&u128::MAX), "the count overflowed");
        assert!(reach(&m).slots > 0, "the root reads the count from its slot");
        assert_eq!(eng.evaluate(&m, &w).unwrap(), expected);
        assert_eq!(eng.evaluate_columns(&m, &Columns(&w)).unwrap(), expected);
        check_levels(&eng, &m, &w, "past u128");
    }
}

#[test]
fn hand_built_marginal_levels_read_their_counts() {
    let eng = Engine::new();
    let mut rng = Lcg::new(53);
    // `x1 · 5` beside a dead `¬x1 · 0` over the summed x2, joined with `x3`
    // and `¬x3` over a free x4: only the first output pair has models.
    let (f, [v4, _, _]) = inline_marginal();
    let w = Weighted::new(&mut rng, 4, &[VarId(2)], false);
    let x = |v: u32, label| w.weight(VarId(v), label);
    let expected = mul(mul(x(1, LeafLabel::Pos), 5), mul(x(3, LeafLabel::Pos), x(4, LeafLabel::One)));
    assert_eq!(eng.evaluate(&f, &w).unwrap(), expected);
    assert_eq!(eng.evaluate_columns(&f, &Columns(&w)).unwrap(), expected);
    assert_eq!(eng.evaluate_at(&f, v4, &w).unwrap(), vec![mul(x(1, LeafLabel::Pos), 5), 0]);
    // Two stored slots, 2^31 and 2^32, too large to ride inline.
    for second in [LeafLabel::Pos, LeafLabel::Neg] {
        let (f, [m, _, _, _]) = marginal_boundary(second);
        let counts = f.level(m).marginal_counts().unwrap().to_vec();
        let w = Weighted::new(&mut rng, 5, &[VarId(2), VarId(5)], true);
        let x = |v: u32, label| w.weight(VarId(v), label);
        let right = mul(x(3, LeafLabel::Pos), x(4, LeafLabel::One));
        let expected = add(
            mul(mul(x(1, LeafLabel::Pos), (counts[0] % u128::from(P)) as u64), right),
            mul(mul(x(1, second), (counts[1] % u128::from(P)) as u64), right),
        );
        assert_eq!(eng.evaluate(&f, &w).unwrap(), expected, "{second:?}");
        assert_eq!(eng.evaluate_columns(&f, &Columns(&w)).unwrap(), expected, "{second:?}");
        check_levels(&eng, &f, &w, "boundary");
    }
}

#[test]
fn an_algebra_without_counts_is_refused_a_marginal_level() {
    struct Plain;
    impl EvalAlgebra for Plain {
        type Value = u64;
        fn zero(&self) -> u64 { 0 }
        fn leaf(&self, _: VarId, label: LeafLabel) -> u64 { u64::from(label != LeafLabel::Zero) }
        fn add_assign(&self, a: &mut u64, b: &u64) { *a += b; }
        fn mul(&self, a: &u64, b: &u64) -> u64 { a * b }
    }
    let eng = Engine::new();
    let (f, [v4, _, _]) = inline_marginal();
    // The inline counts are valued before the walk, and the error names the
    // summed leaf that carries them; the stored ones as the walk reaches them.
    let leaf = f.vtree().children(v4).1;
    assert_eq!(eng.evaluate(&f, &Plain), Err(OperationError::MarginalLevel(leaf)));
    let (f, [m, _, _, _]) = marginal_boundary(LeafLabel::Pos);
    assert_eq!(eng.evaluate(&f, &Plain), Err(OperationError::MarginalLevel(m)));
    assert_eq!(eng.evaluate_at(&f, m, &Plain), Err(OperationError::MarginalLevel(m)));
}
