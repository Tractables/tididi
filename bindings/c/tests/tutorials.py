"""Semantic checks for the compiled tutorials; displayed output is captured separately."""
import re


def verify(outputs):
    expected = {
        "first_circuit": ["Satisfying assignments: 5", "Original: 5; complement: 3"],
        "configurations": ["Valid configurations: 8", "Valid remote configurations: 4",
                           "Forced literals: 2 3", "Remote: 4", "Remote, notifications off: 2", "All choices open: 8"],
        "counting": ["Valid configurations: 8", "With remote backups: 4", "Observed remote backups: 4",
                     "After substituting remote = true: 8", "Distinct remaining choices: 4",
                     "Valid destination choices: 3", "Destination rule over the full vtree: 12", "Destination rule projected: 3"],
        "probability": ["P(wet) = 7/25", "P(rain | wet) = 5/7", "P(wet) = 16/25", "P(rain | wet) = 15/16"],
        "tables": ["Distinct rows: 3", "After updating: 3", "Rows allowing sharing: 2",
                   "After withdrawing sharing: 1"],
        "persistence": ["Restored valid configurations: 4"],
        "vtrees": ["Related variables together: 4 models, 5 pairs", "Related variables separated: 4 models, 12 pairs"],
        "execution": ["Operation exceeded its memory budget; retrying with fresh copies.", "Encrypted configurations: 6"],
        "minimum_cost": ["Minimum cost: 3", "With local storage discounted: 1", "Remote without encryption is feasible: false"],
        "statistics": ["Largest node: 2 pairs"],
        "reachability": ["States after one step: 2", "Unreachable: 3", "Unreachable: 12", "Unreachable: 13",
                         "Unreachable: 14", "Unreachable: 15"],
    }
    assert set(outputs) == set(expected), "register each new tutorial's expected semantics"
    for name, lines in expected.items():
        for line in lines:
            assert line in outputs[name].splitlines(), f"{name}: missing expected result {line!r}"
    iterations = re.findall(r"Step (\d+): (\d+) states, (\d+) nodes, (\d+) pairs(.*)", outputs["reachability"])
    assert [int(row[1]) for row in iterations] == [3, 6, 8, 10, 11, 11]
    assert iterations[-1][2:4] == iterations[-2][2:4], "fixed-point storage should stabilize"
    assert iterations[-1][4] == " (fixed point)"
    assert outputs["reachability"].count("Unreachable:") == 5
    assert re.search(r"Models: 8; stored nodes: \d+; pairs: 4", outputs["statistics"])
