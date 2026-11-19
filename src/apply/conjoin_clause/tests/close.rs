//! The clause entries on a home end in the trimmed form: no leaf named
//! `One` and no constant with a node. Their results are not minimized, so a
//! leaf the result does not depend on goes at the next minimize.

use std::sync::Arc;

use crate::test_helpers::check::check_trimmed;
use crate::vtree::{Home, VarId, Vtree};
use crate::{Engine, Literal, Tdd};

fn leaf(v: u32) -> Vtree {
    Vtree::leaf(VarId(v))
}

fn trimmed(home: &Home, mut f: Tdd) -> Tdd {
    f.trim(home).unwrap();
    f
}

fn closed(eng: &Engine, home: &Home, got: &Tdd, expected: &Tdd) -> Result<(), String> {
    let on_home = |f: &Tdd| eng.embed(f, home.vtree(), |v| v).unwrap().0;
    if !eng.equivalent(&on_home(got), &on_home(expected)).unwrap() {
        return Err("a different function".into());
    }
    if !got.vtree().is_on(home) && got.vtree().num_nodes() != 0 {
        return Err("off the home".into());
    }
    check_trimmed(got, false)?;
    let mut minimized = got.clone();
    eng.minimize(&mut minimized).unwrap();
    check_trimmed(&minimized, true).map_err(|e| format!("after minimize: {e}"))
}

fn assert_all(failed: Vec<String>) {
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

#[test]
fn and_clause_on_a_home_ends_trimmed() {
    let eng = Engine::new();
    let home = Home::new(Vtree::join(&leaf(1), &leaf(2)).unwrap());
    let two = Arc::clone(home.vtree());
    let y = trimmed(&home, eng.literal(&two, 2).unwrap());
    let x_or_y = eng.clause(&two, [1, 2]).unwrap();
    let cases: [(&str, Tdd, &[i32], Tdd); 5] = [
        ("y and (x or not x)", y.clone(), &[1, -1], y.clone()),
        ("y and the empty clause", y.clone(), &[], Tdd::zero(&two)),
        ("false and x", Tdd::zero(&Vtree::empty_projection(&home)), &[1], Tdd::zero(&two)),
        ("y and (x or y)", y.clone(), &[1, 2], y.clone()),
        ("(x or y) and x", x_or_y, &[1], eng.literal(&two, 1).unwrap()),
    ];
    let mut failed = Vec::new();
    for (what, f, clause, expected) in cases {
        let got = eng.and_clause(f, clause).unwrap();
        failed.extend(closed(&eng, &home, &got, &expected).err().map(|e| format!("{what}: {e}")));
    }
    assert_all(failed);
}

#[test]
fn or_cube_on_a_home_ends_trimmed() {
    let eng = Engine::new();
    let home = Home::new(Vtree::join(&Vtree::join(&leaf(1), &leaf(2)).unwrap(), &leaf(3)).unwrap());
    let xyz = Arc::clone(home.vtree());
    let x_or_z = trimmed(&home, eng.clause(&xyz, [1, 3]).unwrap());
    let x = trimmed(&home, eng.literal(&xyz, 1).unwrap());
    let cases: [(&str, Tdd, &[i32], Tdd); 4] = [
        ("(x or z) or (x and y)", x_or_z.clone(), &[1, 2], x_or_z.clone()),
        ("(x or z) or the empty cube", x_or_z.clone(), &[], Tdd::one(&xyz)),
        ("x or (x and y)", x.clone(), &[1, 2], x.clone()),
        ("x or (y and not y)", x.clone(), &[2, -2], x.clone()),
    ];
    let mut failed = Vec::new();
    for (what, f, cube, expected) in cases {
        let got = eng.or_cube(f, cube).unwrap();
        failed.extend(closed(&eng, &home, &got, &expected).err().map(|e| format!("{what}: {e}")));
    }
    let typed = [Literal::pos(VarId(1)), Literal::pos(VarId(2))];
    let got = eng.or_cube(x_or_z.clone(), &typed).unwrap();
    failed.extend(closed(&eng, &home, &got, &x_or_z).err().map(|e| format!("typed: {e}")));
    assert_all(failed);
}

/// A literal on a home knows its loose levels, and a literal or a diagram
/// trimmed to one leaf lists every loose one where it knows them.
#[test]
fn one_leaf_results_list_their_loose_levels() {
    let eng = Engine::new();
    let mut rng = crate::test_helpers::Lcg::new(0x1005e);
    let mut failed = Vec::new();
    for seed in 0..40u64 {
        let home = Home::new(Vtree::random(6, seed));
        let lit = 1 + rng.below(6) as i32;
        let x = Tdd::clause(home.vtree(), [lit]).unwrap();
        let one_leaf = trimmed(&home, eng.and_clause(eng.clause(home.vtree(), [lit, 1 + lit % 6]).unwrap(), &[lit]).unwrap());
        for (what, got) in [("a literal on a home", x), ("trimmed to one leaf", one_leaf)] {
            if what == "a literal on a home" && got.dirty.loose().is_none() {
                failed.push(format!("seed {seed}, {what}: loose levels not known"));
            }
            failed.extend(check_trimmed(&got, false).err().map(|e| format!("seed {seed}, {what}: {e}")));
        }
    }
    assert_all(failed);
}
