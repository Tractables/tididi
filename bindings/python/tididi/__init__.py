# scenario: docs/scenarios.md#minimum-cost
"""Boolean circuits with explicit ownership, exact counts, and weighted evaluation."""
from typing import Protocol, TypeVar
from ._tididi import (
    Circuit, ConsumedCircuitError, Counter, Evaluator, Limits, Literal, ResourceLimitError,
    Vtree, and_, and_exists, clause, cube, from_models, ite, literal, one,
    or_, or_many, xor, zero, __version__,
)

__all__ = [
    "Algebra", "Circuit", "ConsumedCircuitError", "Counter", "Evaluator", "Limits", "Literal",
    "ResourceLimitError", "Vtree", "and_", "and_exists", "clause", "cube",
    "from_models", "ite", "literal", "one", "or_", "or_many", "xor", "zero",
]

_T = TypeVar("_T")


class Algebra(Protocol[_T]):
    """Evaluate disjoint alternatives with add and independent groups with mul.

    Values must be immutable. Both operations must be associative and commutative;
    mul distributes over add. zero is the identity for add and absorbing for mul.
    For each variable, leaf(variable, None) must equal
    add(leaf(variable, False), leaf(variable, True)): a free variable includes
    both assignments. Violating these laws can make equivalent circuits evaluate
    differently. Floating-point arithmetic only approximates the algebraic laws;
    use exact values when exact results matter.
    """

    def zero(self) -> _T:
        """The value of an impossible event (the additive identity)."""
        ...

    def leaf(self, variable: int, sign: bool | None) -> _T:
        """Evaluate a positive, negative, or free leaf."""
        ...

    def add(self, a: _T, b: _T) -> _T:
        """Combine disjoint alternatives without mutating either argument."""
        ...

    def mul(self, a: _T, b: _T) -> _T:
        """Combine independent variable groups without mutating either argument."""
        ...
