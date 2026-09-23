"""Generate shared Boolean traces using only Python integers as truth tables.

Run with --write after extending a scenario. The three API interpreters consume
conformance.txt; this generator never imports tididi or executes library code.
"""
from pathlib import Path
import random
import sys

FIXTURE = Path(__file__).with_name("fixtures") / "conformance.txt"


def generate():
    lines = ["# op a b c truth-mask; circuit IDs are zero-based row numbers within each case.",
             "# case variables linear seed; signed literals and variable IDs are one-based."]
    for n, linear, seed in [(1, 0, 719), (4, 0, 31017), (4, 1, 31017), (5, 0, 67021)]:
        lines.append(f"case {n} {linear} {seed} 0")
        rng = random.Random(seed)
        full = (1 << (1 << n)) - 1
        tables = []

        def emit(op, a=0, b=0, c=0):
            def at(index, assignment):
                return (tables[index] >> assignment) & 1
            if op == "literal":
                truth = sum(1 << x for x in range(1 << n)
                            if bool(x & (1 << (abs(a) - 1))) == (a > 0))
            elif op == "one": truth = full
            elif op == "zero": truth = 0
            elif op == "and": truth = tables[a] & tables[b]
            elif op == "or": truth = tables[a] | tables[b]
            elif op == "xor": truth = tables[a] ^ tables[b]
            elif op == "negate": truth = full ^ tables[a]
            elif op == "ite": truth = (tables[a] & tables[b]) | ((full ^ tables[a]) & tables[c])
            elif op == "condition":
                bit = 1 << (abs(b) - 1)
                truth = sum(at(a, (x & ~bit) | (bit if b > 0 else 0)) << x for x in range(1 << n))
            elif op == "exists":
                bit = 1 << (b - 1)
                truth = sum((at(a, x & ~bit) | at(a, x | bit)) << x for x in range(1 << n))
            elif op in {"swap", "rename"}:
                source, target = 1 << (b - 1), 1 << (c - 1)
                def mapped(x):
                    y = (x & ~source) | (source if x & target else 0)
                    if op == "swap": y = (y & ~target) | (target if x & source else 0)
                    return y
                truth = sum(at(a, mapped(x)) << x for x in range(1 << n))
            elif op in {"minimize", "roundtrip"}: truth = tables[a]
            else: raise AssertionError(op)
            lines.append(f"{op} {a} {b} {c} {truth:x}")
            tables.append(truth)
            return len(tables) - 1

        for var in range(1, n + 1):
            emit("literal", var)
            emit("literal", -var)
        emit("one")
        emit("zero")
        # Explicit shared operands, contradictions, tautologies and non-injective renames.
        emit("and", 0, 1)
        emit("or", 0, 1)
        parity = emit("xor", 0, 2 if n > 1 else 0)
        if n > 1:
            emit("rename", parity, 1, 2)
            emit("swap", parity, 1, 2)
        ops = ["and", "or", "xor", "negate", "ite", "condition", "exists", "minimize", "roundtrip"]
        if n > 1: ops += ["swap", "rename"]
        for _ in range(77):
            op = rng.choice(ops)
            a, b, c = [rng.randrange(len(tables)) for _ in range(3)]
            if op == "condition": b, c = rng.choice([-1, 1]) * rng.randint(1, n), 0
            elif op == "exists": b, c = rng.randint(1, n), 0
            elif op in {"swap", "rename"}: b, c = rng.sample(range(1, n + 1), 2)
            elif op in {"negate", "minimize", "roundtrip"}: b = c = 0
            elif op != "ite": c = 0
            emit(op, a, b, c)
    return "\n".join(lines) + "\n"


SESSION_FIXTURE = FIXTURE.with_name("conformance_sessions.txt")


def generate_sessions():
    """Expected answers for ownership, refusal and cached-evidence sequences."""
    lines = ["# op a b c expected-count; case gives variables, linear, weighted, truth-mask.",
             "# save/recover copy the circuit; open/finish transfer it to/from a cached query."]
    for n, linear in [(2, 0), (4, 0), (5, 1)]:
        for weighted in [0, 1]:
            truth = sum(1 << x for x in range(1 << n) if x & 3)
            lines.append(f"case {n} {linear} {weighted} {truth:x}")
            pins = {}
            rng = random.Random(71967025 + n)

            def emit(op, a=0, b=0):
                if op in {"observe", "refuse_dirty"}:
                    for v in [a, b]:
                        if v: pins[abs(v)] = v > 0
                elif op == "clear_one": pins.pop(a, None)
                elif op in {"clear", "finish"}: pins.clear()
                count = sum(bool(truth >> x & 1) and all(
                    bool(x & (1 << (v - 1))) == value for v, value in pins.items())
                    for x in range(1 << n))
                lines.append(f"{op} {a} {b} 0 {count:x}")

            emit("save")
            emit("open")
            emit("refuse_read")  # Cold cache.
            emit("observe", -1)
            emit("reject_observe", 1, n + 1)  # Valid prefix must not take effect.
            emit("refuse_read")  # Warm cache.
            for _ in range(12):
                a = rng.choice([-1, 1]) * rng.randint(1, n)
                b = rng.choice([-1, 1]) * rng.randint(1, n)
                emit("observe", a, b)
                emit("refuse_dirty", -a, -b)  # Fail a dirty refresh, then retry.
                emit("clear_one", abs(a))
            emit("clear")
            emit("finish")
            emit("refuse_transform")
            emit("recover")
            emit("roundtrip")
            emit("minimize")
            emit("open")
            emit("observe", -1, -2)
            emit("finish")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    for path, expected in [(FIXTURE, generate()), (SESSION_FIXTURE, generate_sessions())]:
        if sys.argv[1:] == ["--write"]:
            path.write_text(expected)
        elif path.read_text() != expected:
            raise SystemExit(f"{path.name} is stale; run python3 tests/conformance_cases.py --write")
