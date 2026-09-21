# scenario: docs/scenarios.md#minimum-cost

"""
A custom minimum-cost calculation
=====================================
The same Boolean circuit can answer more than counting questions. Here we
find the cheapest valid backup configuration: local storage costs 5, remote
storage 2, and encryption 1. Notifications are free.
"""
from math import inf
from tididi import Vtree, literal

vtree = Vtree.balanced(4)
local = literal(vtree, 1)
remote = literal(vtree, 2)
encrypted = literal(vtree, 3)
configurations = (local | remote.copy()) & (~remote.copy() | encrypted.copy())

# %%
# Supply the algebra
# ----------------------
# A false function has infinite cost. A positive literal pays its variable's
# cost; a negative literal costs zero. For a free variable, either value is
# possible, so choose the cheaper one. Disjunction chooses the minimum;
# conjunction adds costs over disjoint variable groups.
class Costs:
    def __init__(self, prices):
        self.prices = prices

    def zero(self):
        return inf

    def leaf(self, variable, sign):
        price = self.prices[variable]
        if sign is None:
            return min(0, price)
        return price if sign else 0

    def add(self, a, b):
        return min(a, b)

    def mul(self, a, b):
        return a + b


costs = Costs({1: 5, 2: 2, 3: 1, 4: 0})
print("Minimum configuration cost:", configurations.evaluate(costs))
discount = Costs({1: 1, 2: 2, 3: 1, 4: 0})
print("Minimum with discount:", configurations.evaluate(discount))

# %%
# Reuse the circuit with new costs, or constrain it with another Boolean
# condition. Evaluation borrows; Boolean operations consume their operands.
with_remote = configurations.copy() & remote
print("Minimum with remote backups:", with_remote.evaluate(discount))
without_encryption = with_remote & ~encrypted
print("Minimum without encryption:", without_encryption.evaluate(costs))

# %%
# Algebra values must be immutable, and ``add`` and ``mul`` must return values
# without mutating their arguments. Python callbacks make this flexible but
# add interpreter overhead per evaluation step. For exact weighted sums,
# ``weighted_count()`` runs the arithmetic directly in Rust.
