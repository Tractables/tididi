"""Every circuit a public entry returns stores a level for a variable only
when it reads it.

A vtree built here is a home, so a circuit sits on the home's projection on
the variables its pairs name: no leaf is named true, and after `minimize`
the leaves are exactly the support. The formulas are random clause sets over
a few variables scattered across a random vtree of many.
"""
import random
import re

import tididi as td

SUBSCRIPTS = str.maketrans("\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089", "0123456789")


def random_vtree(rng, n):
    trees = [td.Vtree.leaf(v) for v in rng.sample(range(1, n + 1), n)]
    while len(trees) > 1:
        i = rng.randrange(len(trees) - 1)
        trees[i:i + 2] = [td.Vtree.join(trees[i], trees[i + 1])]
    return trees[0]


def random_formula(rng, vtree, variables):
    f = td.one(vtree)
    for _ in range(rng.randint(1, 4)):
        width = rng.randint(1, min(3, len(variables)))
        f = f & td.clause(vtree, [v if rng.getrandbits(1) else -v for v in rng.sample(variables, width)])
    return f


def leaves(c, label):
    # `to_dot` labels a leaf node `X`, `¬X` or `1` with its variable in
    # subscript.
    return {int(m.group(1).translate(SUBSCRIPTS)) for m in re.finditer(label + r'([\u2080-\u2089]+)"', c.to_dot())}


def check(c, what):
    assert not leaves(c, 'label="1'), f"{what}: a leaf is named true"
    minimized = c.copy().minimize()
    assert leaves(minimized, 'label="\u00ac?X') == set(minimized.support()), f"{what}: minimized: the leaves are not the support"


def test_every_producer_stores_only_used_levels(tmp_path):
    rng = random.Random(1124)
    for case in range(60):
        n = rng.randint(4, 16)
        v = random_vtree(rng, n)
        variables = rng.sample(range(1, n + 1), rng.randint(2, min(5, n)))
        f, g, h = (random_formula(rng, v, variables) for _ in range(3))
        x, y = rng.sample(variables, 2)
        some = rng.sample(variables, rng.randint(1, len(variables)))
        label = f"case {case} on {n} variables, formulas over {sorted(variables)}"
        for name, c in [("operand", f.copy()),
                        ("literal", td.literal(v, x)),
                        ("cube", td.cube(v, [x, -y])),
                        ("clause", td.clause(v, [-x, y])),
                        ("one", td.one(v)),
                        ("zero", td.zero(v)),
                        ("from_models", td.from_models(v, [x, y], [[True, False], [False, True]])),
                        ("and_", td.and_(f.copy(), g.copy())),
                        ("&", f.copy() & g.copy()),
                        ("or_", td.or_(f.copy(), g.copy())),
                        ("|", f.copy() | g.copy()),
                        ("xor", td.xor(f.copy(), g.copy())),
                        ("^", f.copy() ^ g.copy()),
                        ("ite", td.ite(f.copy(), g.copy(), h.copy())),
                        ("or_many", td.or_many([f.copy(), g.copy(), h.copy()])),
                        ("and_exists", td.and_exists(f.copy(), g.copy(), some)),
                        ("negate", f.copy().negate()),
                        ("~", ~f.copy()),
                        ("condition", f.copy().condition([x, -y])),
                        ("exists", f.copy().exists(some)),
                        ("forall", f.copy().forall(some)),
                        ("rename", f.copy().rename({x: y, y: x})),
                        ("minimize", f.copy().minimize()),
                        ("update", f.copy().update(insert=[[x, y]], remove=[[-x, -y]])),
                        ("copy", f.copy()),
                        ("from_bytes", td.Circuit.from_bytes(v, f.to_bytes())),
                        ("counter().finish", f.copy().counter().finish()),
                        ("evaluator().finish", f.copy().evaluator({u: (1, 2) for u in v.variables}).finish())]:
            check(c, f"{label}: {name}")
        path = tmp_path / "circuit.tdd"
        f.save(path)
        check(td.Circuit.load(v, path), f"{label}: load")
