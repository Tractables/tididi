"""
Reachability in a directed graph
====================================
Starting at node 0, which nodes of this graph can we reach?

.. image:: /_static/reachability.svg
   :alt: Sixteen nodes. Node 3 points into the reachable grid but has no incoming edges. Nodes 12 through 15 form a separate cycle.
   :width: 780

We represent a state with one Boolean indicator per node. Exactly one indicator
is true. For example,
``A₀(x₀, …, x₁₅) = x₀ ∧ ¬x₁ ∧ ⋯ ∧ ¬x₁₅``.
A set of states is the disjunction of their indicators.

With current-state variables ``x`` and next-state variables ``x′``,
the transition relation is
``T(x, x′) = ⋁_{(i,j)∈E} (Aᵢ(x) ∧ Aⱼ(x′))``.
"""
from tididi import Vtree, and_exists, cube, or_many

nodes = 16
edges = [
    (0, 1), (1, 2), (3, 2), (4, 5), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11),
    (0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (6, 10), (7, 11),
    (4, 0), (5, 1), (10, 6), (11, 7), (12, 13), (13, 15), (15, 14), (14, 12),
]
vtree = Vtree.balanced(2 * nodes)
current_vars = list(range(1, nodes + 1))

# %%
# Build state indicators
# --------------------------
# Variable IDs begin at 1; graph nodes begin at 0. The state function asserts
# one indicator and negates every other indicator in the supplied group.
def state(vtree, indicators, node):
    return cube(vtree, (var if i == node else -var for i, var in enumerate(indicators)))


at_current = [state(vtree, current_vars, node) for node in range(nodes)]
reached = at_current[0].copy()

# %%
# The second group describes the state after a transition. Each edge
# contributes one conjunction to the relation; ``or_many`` combines them.
next_vars = list(range(nodes + 1, 2 * nodes + 1))
at_next = [state(vtree, next_vars, node) for node in range(nodes)]
transition = or_many(at_current[a].copy() & at_next[b].copy() for a, b in edges)
print("Transition pairs:", transition.pair_count())

# %%
# Take one step
# -----------------
# Conjoin the current states with the transition relation, then forget the
# old state: ``∃x. R(x) ∧ T(x, x′)``.
possible_steps = reached.copy() & transition.copy()
successors = possible_steps.exists(current_vars)

# %%
# The result describes next-state variables. Rename those back to current-state
# variables so it can be used as the input to another step.
next_to_current = dict(zip(next_vars, current_vars))
successors = successors.rename(next_to_current)
print("States after one step:", successors.projected_model_count(current_vars))

# %%
# ``and_exists`` performs the conjunction and quantification we just wrote
# as two separate operations in a single call. It computes the same function
# and can avoid building parts of the intermediate conjunction.
def image(states, relation):
    return and_exists(states, relation, current_vars).rename(next_to_current)


combined = image(reached.copy(), transition.copy())
print("Combined operation agrees:", combined.equivalent(successors))

# %%
# Iterate to a fixed point
# ----------------------------
# Keep the states already reached and add their successors. Semantic
# equivalence tells us when this adds no new states. We count projections
# onto current-state variables, since the next-state variables are free here.
iteration = 0
while True:
    successors = image(reached.copy(), transition.copy())
    enlarged = reached.copy() | successors
    iteration += 1
    count = enlarged.projected_model_count(current_vars)
    print(f"Iteration {iteration}: {count} states, "
          f"{enlarged.node_count()} circuit nodes, {enlarged.pair_count()} pairs")
    fixed = enlarged.equivalent(reached)
    reached = enlarged
    if fixed:
        print("Fixed point reached")
        break

# %%
# Two reasons for being unreachable
# -------------------------------------
# Node 3 is connected to the grid, but its edges point away from it and there
# is no path from 0 to 3. Nodes 12–15 form an entirely separate component.
print("Node 3 unreachable:", reached.implies(~at_current[3].copy()))
separate_component = or_many(node.copy() for node in at_current[12:])
print("Nodes 12–15 unreachable:", reached.implies(~separate_component))
target = reached & at_current[11].copy()
witness = target.satisfying_assignment()
active = [node for node, var in enumerate(current_vars)
          if any(choice.variable == var and choice.sign for choice in witness)]
print("Reachable target:", active)
