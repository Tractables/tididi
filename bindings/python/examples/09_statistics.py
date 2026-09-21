# scenario: docs/scenarios.md#statistics

"""
Inspecting circuit size
===========================
Model counts describe a function's solutions. Nodes and pairs describe the
stored circuit. We can inspect that storage to locate the node with the
largest number of alternatives.
"""
from tididi import Vtree, literal

vtree = Vtree.balanced(4)
x1 = literal(vtree, 1)
x2 = literal(vtree, 2)
exclusive = x1.copy() ^ x2

# %%
# ``node_sizes()`` returns a snapshot of the stored internal nodes as
# ``(vtree_node, local_node, pair_count)`` tuples. The snapshot does not borrow
# native storage, so it stays readable after the circuit is consumed.
def widest_node(circuit):
    return max(circuit.node_sizes(), key=lambda node: node[2], default=None)


level, node, pairs = widest_node(exclusive)
print(f"Widest XOR node: {pairs} pairs at vtree node {level}")
print("Widest literal node:", widest_node(x1)[2], "pair")
print("Total XOR pairs:", exclusive.pair_count())
print("Satisfying assignments:", exclusive.model_count())

# %%
# Storage identifiers belong to this particular diagram and can change after
# a transformation. Use semantic queries such as ``equivalent()`` to compare
# functions. For a visual inspection, ``to_dot()`` exports a Graphviz graph.
