# scenario: docs/scenarios.md#probability

"""
Probabilities under changing assumptions
============================================
Rain or a sprinkler makes the ground wet:
``W = R ∨ S``. Given wet ground, what is the probability of rain?
We build the events once, then evaluate them under two sets of independent
priors. A third variable, wind, is unconstrained by these events.
"""
from fractions import Fraction
from tididi import Vtree, literal

vtree = Vtree.balanced(3)
rain = literal(vtree, 1)
sprinkler = literal(vtree, 2)
wet = rain.copy() | sprinkler
rain_and_wet = rain & wet.copy()

# %%
# Exact fractions
# -------------------
# ``Fraction`` comes from Python's standard library. Each variable has a pair
# of weights: false first, true second. For independent probabilities these
# sum to one. In particular, the free wind variable contributes one.
#
# In both scenarios below the evidence has positive probability, so
# ``P(R | W) = P(R ∧ W) / P(W)`` is defined.
for rain_probability in [Fraction(1, 5), Fraction(3, 5)]:
    priors = {1: rain_probability, 2: Fraction(1, 10), 3: Fraction(2, 5)}
    weights = {variable: (1 - p, p) for variable, p in priors.items()}
    wet_probability = wet.weighted_count(weights)
    conditional = rain_and_wet.weighted_count(weights) / wet_probability
    print(f"P(rain) = {rain_probability}, P(wet) = {wet_probability}")
    print("P(rain | wet) =", conditional)

# %%
# Both evaluations borrowed their circuits. No rebuilding or copying was
# needed when the priors changed. More generally, weights need not sum to
# one: unit weights recover the model count.
unit_weights = {variable: (1, 1) for variable in vtree.variables}
print("Models of wet:", wet.weighted_count(unit_weights))
