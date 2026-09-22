"""Ownership boundaries and semantic checks against small truth tables."""
import copy
import itertools
import random
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import pytest
import tididi as td


def test_boolean_operators_consume_and_aliases_observe_it():
    v = td.Vtree.balanced(3)
    a, b = td.literal(v, 1), td.literal(v, 2)
    alias = a
    f = a & b
    assert f.model_count() == 2
    assert a.is_consumed and alias.is_consumed and b.is_consumed
    for query in [a.model_count, a.is_sat, a.copy, a.to_bytes, a.node_sizes, lambda: a.vtree, lambda: bool(a)]:
        with pytest.raises(td.ConsumedCircuitError, match="Copy it before"):
            query()
    assert repr(a) == "Circuit(consumed)"
    g = ~f
    assert f.is_consumed and g.model_count() == 6


def test_explicit_copies_share_domain_and_leave_originals_usable():
    v = td.Vtree.balanced(2)
    original = td.literal(v, 1)
    for duplicate in [original.copy(), copy.copy(original), copy.deepcopy(original)]:
        assert duplicate.equivalent(original)
        assert (duplicate & td.literal(original.vtree, 2)).model_count() == 1
    assert original.equivalent(original)
    assert original.implies(original)
    assert original.model_count() == 2


@pytest.mark.parametrize("combine", [lambda a: a & a, lambda a: td.or_(a, a),
                                     lambda a: td.or_many([a, a]), lambda a: td.ite(a, a, a)])
def test_repeated_operands_rejected_before_consumption(combine):
    a = td.literal(td.Vtree.balanced(2), 1)
    with pytest.raises(ValueError, match="same circuit occurs twice"):
        combine(a)
    assert a.model_count() == 2


def test_preflight_errors_do_not_consume_other_operands():
    v = td.Vtree.balanced(3)
    a, b = td.literal(v, 1), td.literal(v, 2)
    ~b
    with pytest.raises(td.ConsumedCircuitError):
        td.and_(a, b)
    with pytest.raises(ValueError, match="share"):
        td.and_(a, td.literal(td.Vtree.balanced(3), 1))
    with pytest.raises(TypeError):
        td.and_(a, 5)
    with pytest.raises(ValueError):
        a.exists([4])
    with pytest.raises(ValueError):
        a.rename({1: 9})
    with pytest.raises(ValueError):
        a.condition([1, 4])
    assert a.model_count() == 4


def test_iterable_failure_is_atomic():
    a = td.literal(td.Vtree.balanced(2), 1)

    def operands():
        yield a
        raise LookupError("broken iterator")

    with pytest.raises(LookupError, match="broken iterator"):
        td.or_many(operands())
    assert not a.is_consumed
    with pytest.raises(ValueError, match="at least one"):
        td.or_many([])


def test_truthiness_and_unsupported_operators_do_not_consume():
    a = td.literal(td.Vtree.balanced(2), 1)
    with pytest.raises(TypeError, match="no Python truth value"):
        bool(a)
    with pytest.raises(TypeError):
        a & 1
    assert not a.is_consumed


def test_queries_and_transformations_against_truth_table():
    v = td.Vtree.balanced(4)
    a, b, c = [td.literal(v, i) for i in [1, 2, 3]]
    f = (a | b.copy()) & (~b | c)
    models = [bits for bits in itertools.product([False, True], repeat=4)
              if (bits[0] or bits[1]) and (not bits[1] or bits[2])]
    assert f.model_count() == len(models) == 8
    assert f.support() == [1, 2, 3]
    assert f.projected_model_count([1, 2, 3]) == 4
    assert f.projected_model_count([]) == 1
    witness = f.satisfying_assignment()
    assert set(x.variable for x in witness) == {1, 2, 3, 4}
    assert td.cube(v, witness).implies(f)
    assert f.copy().exists([1, 2, 3]).model_count() == 16
    assert f.copy().condition([2]).model_count() == 8
    assert f.copy().rename({1: 2, 2: 1}).rename({1: 2, 2: 1}).equivalent(f)
    assert f.copy().minimize().equivalent(f)
    impossible = td.zero(v)
    assert impossible.satisfying_assignment() is None
    assert set(map(int, impossible.implied_literals())) == {1, -1, 2, -2, 3, -3, 4, -4}


def test_composition_and_fused_projection():
    v = td.Vtree.balanced(3)
    a, b, c = [td.literal(v, i) for i in [1, 2, 3]]
    chosen = td.ite(a.copy(), b.copy(), c.copy())
    expected = (a.copy() & b.copy()) | (~a.copy() & c.copy())
    assert chosen.equivalent(expected)
    assert (a.copy() ^ b.copy()).model_count() == 4
    fused = td.and_exists(a.copy(), b.copy(), [1])
    assert fused.equivalent((a & b).exists([1]))
    union = td.or_many(td.literal(v, i) for i in [1, 2, 3])
    assert union.model_count() == 7


def test_big_integer_and_sparse_domains():
    assert td.one(td.Vtree.balanced(257)).model_count() == 2**257
    v = td.Vtree.balanced_over([1, 3])
    f = td.literal(v, 3)
    assert f.model_count() == 2
    assert f.weighted_count({1: (1, 1), 3: (1, 1)}) == 2
    assert f.projected_model_count([3]) == 1
    with pytest.raises(ValueError):
        td.literal(v, 2)


def test_vtrees_and_named_literals():
    v = td.Vtree.join(td.Vtree.leaf(1), td.Vtree.balanced_over([3, 2]))
    assert v.variables == [1, 3, 2]
    assert td.Vtree.from_text(v.to_text()).to_text() == v.to_text()
    assert td.Vtree.linear([3, 1, 2]).variables == [3, 1, 2]
    l = td.Literal(3)
    assert l == td.Literal(3) and hash(l) == hash(td.Literal(3))
    assert l.variable == 3 and l.sign and int(~l) == -3
    assert td.literal(v, l).model_count() == 4
    assert td.cube(v, [l, ~td.Literal(1)]).model_count() == 2
    for build in [lambda: td.Vtree.balanced(0), lambda: td.Vtree.balanced_over([]),
                  lambda: td.Vtree.balanced_over([1, 1]), lambda: td.Literal(0),
                  lambda: td.literal(v, 0), lambda: td.literal(v, 2**33)]:
        with pytest.raises(ValueError):
            build()
    with pytest.raises(TypeError):
        td.literal(v, True)


def test_limits_consume_only_after_validation_and_queries_survive():
    v = td.Vtree.balanced(8)
    a, b = td.clause(v, [1, 2]), td.clause(v, [3, 4])
    with pytest.raises(td.ResourceLimitError):
        td.and_(a, b, limits=td.Limits(timeout=0))
    assert a.is_consumed and b.is_consumed
    f = td.clause(v, [1, 2, 3])
    with pytest.raises(td.ResourceLimitError):
        f.model_count(limits=td.Limits(timeout=0))
    assert f.model_count() == 224
    v.clear_scratch()
    assert f.model_count() == 224
    for timeout in [-1, float("inf"), float("nan")]:
        with pytest.raises(ValueError):
            td.Limits(timeout=timeout)


def test_counter_owns_circuit_reuses_pins_and_returns_original():
    v = td.Vtree.balanced(4)
    f = td.clause(v, [1, 2])
    expected = f.copy()
    counter = f.counter()
    assert f.is_consumed and counter.model_count() == 12
    counter.observe([td.Literal(1)])
    assert counter.model_count() == 8
    counter.observe([-1])
    assert counter.model_count() == 4
    with pytest.raises(ValueError):
        counter.observe([1, 9])
    assert counter.model_count() == 4
    counter.clear(1)
    assert counter.model_count() == 12
    counter.observe([1, -2])
    assert counter.model_count() == 4
    counter.clear_observations()
    assert counter.model_count() == 12
    recovered = counter.finish()
    assert recovered.equivalent(expected)
    with pytest.raises(RuntimeError, match="finished"):
        counter.model_count()


def test_table_updates_and_rows_crossing_word_boundaries():
    rng = random.Random(721)
    v = td.Vtree.balanced(65)
    variables = list(range(1, 66))
    rows = [tuple(bool(rng.getrandbits(1)) for _ in variables) for _ in range(20)]
    table = td.from_models(v, variables, rows + rows[:5])
    assert table.model_count() == len(set(rows))
    new_row = [True] * 65
    removed = [var if value else -var for var, value in zip(variables, rows[0])]
    table = table.update(insert=[variables], remove=[removed])
    assert table.model_count() == 20
    assert td.cube(v, variables).implies(table)
    with pytest.raises(ValueError, match="one bool"):
        td.from_models(v, variables, [new_row[:-1]])
    small = td.Vtree.balanced(3)
    assert td.from_models(small, [], []).model_count() == 0
    assert td.from_models(small, [], [[]]).model_count() == 8
    assert td.one(small).update(remove=[[]]).model_count() == 0


def test_exact_weights_and_custom_algebra_errors():
    v = td.Vtree.balanced(3)
    f = td.clause(v, [1, 2])
    weights = {1: (Fraction(4, 5), Fraction(1, 5)),
               2: (Fraction(9, 10), Fraction(1, 10)), 3: (Fraction(3, 5), Fraction(2, 5))}
    assert f.weighted_count(weights) == Fraction(7, 25)
    with pytest.raises(ValueError, match="missing weights"):
        f.weighted_count({1: (1, 1)})
    with pytest.raises(TypeError, match="Fraction"):
        f.weighted_count({1: (0.5, 0.5), 2: (1, 1), 3: (1, 1)})

    class Count:
        def zero(self): return 0
        def leaf(self, variable, sign): return 2 if sign is None else 1
        def add(self, a, b): return a + b
        def mul(self, a, b): return a * b

    assert f.evaluate(Count()) == f.model_count()

    class Broken(Count):
        def leaf(self, variable, sign): raise LookupError("callback failed")

    with pytest.raises(LookupError, match="callback failed"):
        f.evaluate(Broken())
    assert f.model_count() == 6

    class Reentrant(Count):
        def leaf(self, variable, sign):
            ~f
            return 1

    with pytest.raises(RuntimeError):
        f.evaluate(Reentrant())
    assert f.model_count() == 6


def test_serialization_and_storage_snapshot(tmp_path):
    v = td.Vtree.balanced(4)
    f = td.clause(v, [1, -2])
    assert td.Circuit.from_bytes(v, f.to_bytes()).equivalent(f)
    path = tmp_path / "circuit.tdd"
    f.save(path)
    assert td.Circuit.load(v, path).equivalent(f)
    with pytest.raises(FileNotFoundError):
        td.Circuit.load(v, tmp_path / "absent")
    with pytest.raises(ValueError):
        td.Circuit.from_bytes(v, b"not a circuit")
    sizes = f.node_sizes()
    assert sum(pairs for _, _, pairs in sizes) == f.pair_count()
    assert "graph tdd" in f.to_dot() and "graph" in v.to_dot()
    ~f
    assert sizes


def test_native_calls_from_threads_share_vtree_safely():
    v = td.Vtree.balanced(8)

    def compute(i):
        return (td.literal(v, i) & td.literal(v, 8)).model_count()

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(compute, range(1, 8))) == [64] * 7


def test_seeded_boolean_formulas_match_independent_assignments():
    rng = random.Random(31017)
    v = td.Vtree.balanced(4)
    universe = set(itertools.product([False, True], repeat=4))

    def formula(depth):
        if depth == 0:
            variable = rng.randrange(1, 5)
            sign = bool(rng.getrandbits(1))
            return td.literal(v, variable if sign else -variable), {a for a in universe if a[variable - 1] == sign}
        left, models_left = formula(depth - 1)
        right, models_right = formula(depth - 1)
        operation = rng.choice(["and", "or", "xor"])
        if operation == "and":
            return left & right, models_left & models_right
        if operation == "or":
            return left | right, models_left | models_right
        return left ^ right, models_left ^ models_right

    for _ in range(40):
        circuit, models = formula(3)
        assert circuit.model_count() == len(models)
        assert circuit.projected_model_count([1, 3]) == len({(a[0], a[2]) for a in models})
        counter = circuit.counter()
        for assignment in universe:
            counter.observe([i if truth else -i for i, truth in enumerate(assignment, 1)])
            assert counter.model_count() == int(assignment in models)
        circuit = counter.finish()
        assert (~circuit).model_count() == 16 - len(models)


def test_conditioning_repeats_and_conflicts_match_rust_semantics():
    vtree = td.Vtree.balanced(3)
    original = td.clause(vtree, [1, 2])
    assert original.copy().condition([1, 1]).equivalent(original.copy().condition([1]))
    conflicted = original.copy()
    assert not conflicted.condition([1, -1]).is_sat()
    assert conflicted.is_consumed
    # Contradiction must not hide another invalid variable.
    with pytest.raises(ValueError):
        original.condition([1, -1, 4])
    assert original.model_count() == 6


@pytest.mark.parametrize("build", [td.Vtree.leaf, lambda v: td.Vtree.balanced_over([v]),
                                   lambda v: td.Vtree.linear([v])])
def test_vtree_constructors_report_oversized_variable_spaces_as_value_errors(build):
    with pytest.raises(ValueError, match="variable-id space"):
        build(2**32 - 1)


@pytest.mark.parametrize("transform", [lambda f, **kw: f.negate(**kw),
                                      lambda f, **kw: f.condition([1], **kw),
                                      lambda f, **kw: f.exists([1], **kw),
                                      lambda f, **kw: f.rename({1: 2}, **kw),
                                      lambda f, **kw: f.minimize(**kw),
                                      lambda f, **kw: f.update(insert=[[1]], **kw)])
def test_unary_transformations_share_consumption_and_error_behavior(transform):
    vtree = td.Vtree.balanced(4)
    original = td.clause(vtree, [1, 2])
    with pytest.raises(TypeError):
        transform(original, limits="invalid")
    assert original.model_count() == 12
    result = transform(original)
    assert original.is_consumed
    assert isinstance(result.model_count(), int)
    with pytest.raises(td.ConsumedCircuitError):
        transform(original)
    refused = td.clause(vtree, [1, 2])
    with pytest.raises(td.ResourceLimitError):
        transform(refused, limits=td.Limits(timeout=0))
    assert refused.is_consumed
