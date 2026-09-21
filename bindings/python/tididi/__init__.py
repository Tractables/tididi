"""Boolean circuits with explicit ownership, exact counts, and weighted evaluation."""
from typing import Protocol, TypeVar
from ._tididi import (
    Circuit, ConsumedCircuitError, Counter, Limits, Literal, ResourceLimitError,
    Vtree, and_, and_exists, clause, cube, from_models, ite, literal, one,
    or_, or_many, xor, zero, __version__,
)

__all__ = [
    "Algebra", "Circuit", "ConsumedCircuitError", "Counter", "Limits", "Literal",
    "ResourceLimitError", "Vtree", "and_", "and_exists", "clause", "cube",
    "from_models", "ite", "literal", "one", "or_", "or_many", "xor", "zero",
]

_T = TypeVar("_T")


class Algebra(Protocol[_T]):
    """Evaluation operations over immutable values; leaf sign None denotes a free variable."""

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
