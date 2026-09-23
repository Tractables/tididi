"""Shared operation traces, with answers generated independently of tididi."""
from pathlib import Path
import runpy

import tididi as td

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests/fixtures/conformance.txt"


def test_shared_behavioral_traces():
    generator = runpy.run_path(str(ROOT / "tests/conformance_cases.py"))
    assert FIXTURE.read_text() == generator["generate"]()
    circuits = []
    for number, line in enumerate(FIXTURE.read_text().splitlines(), 1):
        if line.startswith("#"):
            continue
        op, a, b, c, truth = line.split()
        a, b, c, truth = int(a), int(b), int(c), int(truth, 16)
        if op == "case":
            n = a
            vtree = td.Vtree.balanced(n) if b == 0 else td.Vtree.linear(list(range(1, n + 1)))
            circuits.clear()
            continue
        def copy(i): return circuits[i].copy()
        if op == "literal": f = td.literal(vtree, a)
        elif op == "one": f = td.one(vtree)
        elif op == "zero": f = td.zero(vtree)
        elif op == "and": f = copy(a) & copy(b)
        elif op == "or": f = copy(a) | copy(b)
        elif op == "xor": f = copy(a) ^ copy(b)
        elif op == "ite": f = td.ite(copy(a), copy(b), copy(c))
        elif op == "negate": f = ~copy(a)
        elif op == "condition": f = copy(a).condition([b])
        elif op == "exists": f = copy(a).exists([b])
        elif op == "swap": f = copy(a).rename({b: c, c: b})
        elif op == "rename": f = copy(a).rename({b: c})
        elif op == "minimize": f = copy(a).minimize()
        elif op == "roundtrip": f = td.Circuit.from_bytes(vtree, circuits[a].to_bytes())
        else: raise AssertionError(op)
        context = f"line {number}: {line}"
        assert f.model_count() == truth.bit_count(), context
        counter = f.counter()
        for bits in range(1 << n):
            counter.observe([v if bits & (1 << (v - 1)) else -v for v in range(1, n + 1)])
            assert counter.model_count() == (truth >> bits) & 1, (context, bits)
        counter.clear_observations()
        assert counter.model_count() == truth.bit_count(), context
        f = counter.finish()
        expected_support = [v for v in range(1, n + 1) if any(
            ((truth >> x) ^ (truth >> (x ^ (1 << (v - 1))))) & 1 for x in range(1 << n))]
        assert f.support() == expected_support, context
        expected_implied = [v for v in range(-n, n + 1) if v and all(
            not (truth >> x) & 1 or bool(x & (1 << (abs(v) - 1))) == (v > 0) for x in range(1 << n))]
        assert sorted(map(int, f.implied_literals())) == expected_implied, context
        circuits.append(f)
