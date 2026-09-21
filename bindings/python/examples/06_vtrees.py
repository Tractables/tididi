# scenario: docs/scenarios.md#vtrees

"""
Choosing a variable grouping
================================
A vtree groups variables into subproblems. Consider the function
``(x₁ ↔ x₃) ∧ (x₂ ↔ x₄)``.
One vtree puts each equality together; another splits them across the root.

.. image:: /_static/vtree-grouping.svg
   :alt: Grouped vtree pairs x1 with x3 and x2 with x4. Split vtree pairs x1 with x2 and x3 with x4.
   :width: 780
"""
from tididi import Vtree, literal


def equalities(vtree):
    x1, x2, x3, x4 = [literal(vtree, variable) for variable in range(1, 5)]
    return (~(x1 ^ x3) & ~(x2 ^ x4)).minimize()


grouped = equalities(Vtree.balanced_over([1, 3, 2, 4]))
split = equalities(Vtree.balanced_over([1, 2, 3, 4]))
print("Models:", grouped.model_count(), split.model_count())
print("Pairs with grouped equalities:", grouped.pair_count())
print("Pairs with split equalities:", split.pair_count())

# %%
# The function has four models in either case, but the circuit size changes.
# A useful grouping keeps strongly related variables together.
#
# Use ``Vtree.join(left, right)`` to assemble a particular grouping or
# ``Vtree.linear(variable_order)`` for a right-linear vtree. These constructors
# create new domains. Construct or reload all operands on the resulting
# shared vtree before combining them.
