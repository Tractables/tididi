//! Public operations on the unreduced diagrams other public operations return.
//!
//! `random_sequences` runs seeded sequences of random operations on small
//! vtrees and feeds every result, reduced or not, to the next operations.
//! Each result is checked against a truth table computed from the operands'
//! tables, for determinism over every stored node, live or not, and for a
//! correct negation of a random variable expansion of it.
//! `TIDIDI_UNREDUCED_SEQS` (default 250) sets the number of sequences,
//! `TIDIDI_UNREDUCED_SEED` the stream and `TIDIDI_UNREDUCED_ONLY` the one
//! sequence to replay.

use std::sync::Arc;

use num_bigint::BigUint;
use tididi::diagram::Literal;
use tididi::reduce::ReductionPlan;
use tididi::restructure::VariableExpansion;
use tididi::test_helpers::check::check_determinism;
use tididi::test_helpers::{assert_canonical, eval};
use tididi::vtree::{VarId, Vtree};
use tididi::{Engine, Tdd};

/// The second complement, unpruned, used to keep the first one's operand
/// beside it at the root: two nodes with the same pairs, which the expansion
/// moves below a new root.
#[test]
fn expansion_of_a_twice_negated_diagram_negates() {
    let eng = Engine::new();
    let vtree = Arc::new(Vtree::balanced(3));
    let f = Tdd::one(&vtree).and_clause([-1i32, -2]).unwrap();
    let a = eng.negate_with(&f, ReductionPlan::Contract).unwrap();
    let b = eng.negate_with(&a, ReductionPlan::Contract).unwrap();
    check_determinism(&b).unwrap();
    let g = eng.expand_variables(&b, &identity_with_free(3, 1)).unwrap();
    let neg = eng.negate(&g).unwrap();
    assert_eq!(g.model_count().unwrap(), 12u32.into());
    assert_eq!(neg.model_count().unwrap(), 4u32.into());
}

fn identity_with_free(n: u32, free: u32) -> VariableExpansion {
    VariableExpansion {
        classes: (1..=n).map(|v| vec![Literal::pos(VarId(v))]).collect(),
        constants: Vec::new(),
        free: (n + 1..=n + free).map(VarId).collect(),
        num_vars: n + free,
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 { self.next() % n }
    fn coin(&mut self) -> bool { self.next() & 1 == 1 }
}

/// A truth table: bit `m` is the value at the assignment whose variable `v`
/// is bit `v - 1` of `m`.
type Table = Vec<bool>;

fn table_of(f: &Tdd, n: u32) -> Table {
    (0..1u32 << n).map(|m| eval(f, &bits(m, n))).collect()
}

fn bits(m: u32, n: u32) -> Vec<bool> { (0..n).map(|i| m >> i & 1 == 1).collect() }

fn tabulate(n: u32, f: impl Fn(u32) -> bool) -> Table { (0..1u32 << n).map(f).collect() }

fn set(m: u32, v: VarId, value: bool) -> u32 {
    let bit = 1 << (v.0 - 1);
    if value { m | bit } else { m & !bit }
}

fn quantified(t: &Table, n: u32, vars: &[VarId], any: bool) -> Table {
    let mut out = t.clone();
    for &v in vars {
        out = tabulate(n, |m| {
            let (lo, hi) = (out[set(m, v, false) as usize], out[set(m, v, true) as usize]);
            if any { lo || hi } else { lo && hi }
        });
    }
    out
}

fn literal_holds(m: u32, l: Literal) -> bool { (m >> (l.var.0 - 1) & 1 == 1) == l.sign }

fn subset(rng: &mut Rng, n: u32) -> Vec<VarId> { (1..=n).filter(|_| rng.below(3) == 0).map(VarId).collect() }

fn lits(rng: &mut Rng, n: u32, k: usize) -> Vec<Literal> {
    let mut vars: Vec<u32> = (1..=n).collect();
    (0..k.min(n as usize))
        .map(|_| {
            let i = rng.below(vars.len() as u64) as usize;
            Literal::new(VarId(vars.swap_remove(i)), rng.coin())
        })
        .collect()
}

fn random_expansion(rng: &mut Rng, n: u32) -> VariableExpansion {
    let sizes: Vec<u32> = (0..n).map(|_| 1 + rng.below(3) as u32).collect();
    let (constants, free) = (rng.below(3) as u32, rng.below(3) as u32);
    let num_vars = sizes.iter().sum::<u32>() + constants + free;
    let mut ids: Vec<u32> = (1..=num_vars).collect();
    for i in (1..ids.len()).rev() {
        ids.swap(i, rng.below(i as u64 + 1) as usize);
    }
    let mut next = ids.into_iter();
    let mut take = |rng: &mut Rng| Literal::new(VarId(next.next().unwrap()), rng.coin());
    let classes = sizes.iter().map(|&size| (0..size).map(|_| take(rng)).collect()).collect();
    let constants = (0..constants).map(|_| take(rng)).collect();
    let free = (0..free).map(|_| take(rng).var).collect();
    VariableExpansion { classes, constants, free, num_vars }
}

/// The checks every result gets, or the first that fails.
fn check_result(eng: &Engine, r: &Tdd, n: u32, want: &Table) -> Result<(), String> {
    if table_of(r, n) != *want {
        return Err("wrong function".into());
    }
    let count = want.iter().filter(|&&b| b).count();
    if r.model_count().map_err(|e| e.to_string())? != count.into() {
        return Err(format!("model count is not {count}"));
    }
    check_determinism(r)?;
    let neg = eng.negate(r).map_err(|e| e.to_string())?;
    if table_of(&neg, n).iter().zip(want).any(|(a, b)| a == b) {
        return Err("negation wrong".into());
    }
    Ok(())
}

/// A random expansion of `f` counts `|f| · 2^free` and negates correctly.
fn check_expansion(eng: &Engine, f: &Tdd, n: u32, rng: &mut Rng) -> Result<(), String> {
    let ex = random_expansion(rng, n);
    let g = eng.expand_variables(f, &ex).map_err(|e| e.to_string())?;
    let want = f.model_count().unwrap() << ex.free.len();
    let total = BigUint::from(1u32) << ex.num_vars;
    let got = g.model_count().unwrap();
    if got != want {
        return Err(format!("expansion counts {got}, not {want}; {ex:?}"));
    }
    let neg = eng.negate(&g).map_err(|e| e.to_string())?;
    if neg.model_count().unwrap() != total - want || !eng.and(&g, &neg).unwrap().is_zero() {
        return Err(format!("negation of the expansion wrong; {ex:?}"));
    }
    Ok(())
}

/// One random operation on the pool, the result and the table it must have.
fn step(eng: &Engine, rng: &mut Rng, vtree: &Arc<Vtree>, pool: &[(Tdd, Table)], n: u32, op: usize) -> Option<(Tdd, Table)> {
    let pick = |rng: &mut Rng| &pool[rng.below(pool.len() as u64) as usize];
    let ((f, tf), (g, tg), (h, th)) = (pick(rng), pick(rng), pick(rng));
    let vars = subset(rng, n);
    let x = VarId(1 + rng.below(u64::from(n)) as u32);
    let k = 1 + rng.below(3) as usize;
    let pointwise = |op: fn(bool, bool) -> bool| tabulate(n, |m| op(tf[m as usize], tg[m as usize]));
    let out = match op {
        0 => (eng.and(f, g).unwrap(), pointwise(|a, b| a && b)),
        1 => (eng.or(f.clone(), g.clone()).unwrap(), pointwise(|a, b| a || b)),
        2 => (eng.negate(f).unwrap(), tabulate(n, |m| !tf[m as usize])),
        3 => (eng.xor(f.clone(), g.clone()).unwrap(), pointwise(|a, b| a != b)),
        4 => (eng.and_not(f.clone(), g.clone()).unwrap(), pointwise(|a, b| a && !b)),
        5 => (
            eng.ite(f.clone(), g.clone(), h.clone()).unwrap(),
            tabulate(n, |m| if tf[m as usize] { tg[m as usize] } else { th[m as usize] }),
        ),
        6 => (eng.exists_vars(f.clone(), &vars).unwrap(), quantified(tf, n, &vars, true)),
        7 => (eng.forall_vars(f.clone(), &vars).unwrap(), quantified(tf, n, &vars, false)),
        8 => {
            let value = rng.coin();
            (eng.condition_var(f.clone(), x, value).unwrap(), tabulate(n, |m| tf[set(m, x, value) as usize]))
        }
        9 => {
            let c = lits(rng, n, k);
            let want = tabulate(n, |m| tf[m as usize] && c.iter().any(|&l| literal_holds(m, l)));
            (f.clone().and_clause(c).unwrap(), want)
        }
        10 => {
            let c = lits(rng, n, k);
            let want = tabulate(n, |m| tf[m as usize] || c.iter().all(|&l| literal_holds(m, l)));
            (f.clone().or_cube(c).unwrap(), want)
        }
        11 => (eng.and_in_place(f.clone(), g).unwrap(), pointwise(|a, b| a && b)),
        12 => (eng.and_in_place_loose(f.clone(), g).unwrap(), pointwise(|a, b| a && b)),
        13 => match eng.and_restoring(f, g) {
            Ok(r) => (r, pointwise(|a, b| a && b)),
            Err(_) => return None,
        },
        14 => {
            let y = VarId(1 + rng.below(u64::from(n)) as u32);
            if x == y {
                return None;
            }
            let swap = |m: u32| set(set(m, x, m >> (y.0 - 1) & 1 == 1), y, m >> (x.0 - 1) & 1 == 1);
            (eng.rename_vars(f.clone(), &[(x, y), (y, x)]).unwrap(), tabulate(n, |m| tf[swap(m) as usize]))
        }
        15 => (
            eng.substitute(f.clone(), &[(x, g)]).unwrap(),
            tabulate(n, |m| tf[set(m, x, tg[m as usize]) as usize]),
        ),
        16 => (eng.negate_with(f, ReductionPlan::Prune).unwrap(), tabulate(n, |m| !tf[m as usize])),
        17 => (eng.negate_with(f, ReductionPlan::Contract).unwrap(), tabulate(n, |m| !tf[m as usize])),
        18 => (eng.exists_vars_with(f.clone(), &vars, ReductionPlan::Prune).unwrap(), quantified(tf, n, &vars, true)),
        19 => (eng.exists_vars_with(f.clone(), &vars, ReductionPlan::Contract).unwrap(), quantified(tf, n, &vars, true)),
        20 => {
            let mut r = f.clone();
            r.minimize().unwrap();
            (r, tf.clone())
        }
        21 => {
            let mut buf = Vec::new();
            tididi::io::write_tdd(&mut buf, f).unwrap();
            (tididi::io::read_tdd(&mut &buf[..], vtree).unwrap(), tf.clone())
        }
        22 => {
            let mut buf = Vec::new();
            tididi::io::write_tdd_binary(&mut buf, f).unwrap();
            (tididi::io::read_tdd_binary(&mut &buf[..], vtree).unwrap(), tf.clone())
        }
        23 => {
            let model = lits(rng, n, n as usize);
            let at = model.iter().fold(0, |m, &l| set(m, l.var, l.sign)) as usize;
            let (mut r, mut want) = (f.clone(), tf.clone());
            want[at] = rng.coin();
            if want[at] { r.insert_model(&model).unwrap() } else { r.remove_model(&model).unwrap() }
            (r, want)
        }
        24 => {
            let keep = subset(rng, n);
            if keep.is_empty() {
                return None;
            }
            let local = |v: VarId| keep.iter().position(|&k| k == v).map(|p| VarId(p as u32 + 1));
            let p = eng.project_to_vars(f, local, keep.len() as u32).unwrap();
            let back = |v: VarId| keep[v.idx()];
            let gone: Vec<VarId> = (1..=n).map(VarId).filter(|v| !keep.contains(v)).collect();
            (eng.embed(&p, vtree, back).unwrap().0, quantified(tf, n, &gone, true))
        }
        25 => {
            let mut r = f.clone();
            let target = Arc::new(Vtree::random(n, rng.next()));
            if eng.restructure_to(&mut r, &target, 1 << 20).is_err() {
                return None;
            }
            let back = Arc::clone(vtree);
            if eng.restructure_to(&mut r, &back, 1 << 20).is_err() {
                return None;
            }
            (r, tf.clone())
        }
        _ => (
            eng.or_many(vec![f.clone(), g.clone(), h.clone()]).unwrap(),
            tabulate(n, |m| tf[m as usize] || tg[m as usize] || th[m as usize]),
        ),
    };
    Some(out)
}

#[test]
fn random_sequences() {
    let var = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<u64>().ok());
    let seqs = var("TIDIDI_UNREDUCED_SEQS").unwrap_or(250);
    let seed = var("TIDIDI_UNREDUCED_SEED").unwrap_or(0x9e37_79b9_7f4a_7c15);
    let eng = Engine::new();
    let mut failures = Vec::new();
    let only = var("TIDIDI_UNREDUCED_ONLY");
    for s in (0..seqs).filter(|&s| only.is_none_or(|o| o == s)) {
        let mut rng = Rng(seed ^ s.wrapping_mul(0x2545_f491_4f6c_dd1d) | 1);
        let n = 2 + rng.below(7) as u32;
        let vtree = Arc::new(match rng.below(4) {
            0 => Vtree::balanced(n),
            1 => Vtree::linear(n),
            2 => Vtree::reverse_linear(n),
            _ => Vtree::random(n, rng.next()),
        });
        let mut pool: Vec<(Tdd, Table)> = (0..4)
            .map(|_| {
                let k = 1 + rng.below(3) as usize;
                let mut f = Tdd::one(&vtree);
                for _ in 0..1 + rng.below(4) {
                    f = f.and_clause(lits(&mut rng, n, k)).unwrap();
                }
                let t = table_of(&f, n);
                (f, t)
            })
            .collect();
        let mut trail = Vec::new();
        for _ in 0..14 {
            let at = rng.below(4) as usize;
            let mut fork = Rng(rng.next() | 1);
            let op = fork.below(27) as usize;
            trail.push(op);
            let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| step(&eng, &mut fork, &vtree, &pool, n, op)));
            let Ok(out) = ran else {
                failures.push(format!("sequence {s}: ops {trail:?}: panic"));
                break;
            };
            let Some((r, want)) = out else { continue };
            let checked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                check_result(&eng, &r, n, &want).and_then(|()| check_expansion(&eng, &r, n, &mut fork))
            }))
            .unwrap_or_else(|_| Err("check panicked".into()));
            if let Err(e) = checked {
                failures.push(format!("sequence {s}: ops {trail:?}: {e}"));
                break;
            }
            pool[at] = (r, want);
        }
        for (f, _) in &mut pool {
            f.minimize().unwrap();
            assert_canonical(f);
        }
    }
    for f in failures.iter().take(20) {
        eprintln!("{f}");
    }
    assert!(failures.is_empty(), "{} of {seqs} sequences failed", failures.len());
}
