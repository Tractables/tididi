"""
Errors, limits, and retained scratch
========================================
Ordinary operators raise exceptions on failure. Named operations additionally
accept a ``limits=`` keyword. This lets a caller bound one expensive operation
while using the same circuits and vtree as before.
"""
from tididi import Limits, Vtree, and_, clause, literal

vtree = Vtree.balanced(4)
try:
    destination = clause(vtree, [1, 2], limits=Limits(memory_bytes=0))
except MemoryError:
    print("Not enough budget to build the destination rule")

# %%
# Limits belong to the call. They do not remain installed on later operations.
destination = clause(vtree, [1, 2])
print("Destination choices:", destination.model_count())

# %%
# Keep inputs for a retry
# ---------------------------
# A resource error can occur after a consuming call has taken its inputs.
# Explicit copies preserve the originals for another attempt.
encrypted = literal(vtree, 3)
try:
    secured = and_(destination.copy(), encrypted.copy(), limits=Limits(memory_bytes=0))
except MemoryError:
    print("Original operands still available:", not destination.is_consumed, not encrypted.is_consumed)
secured = destination & encrypted
print("Secured choices:", secured.model_count())

# %%
# A query borrows its circuit even if its budget is refused. A timeout is
# checked cooperatively; an output-node cap can bound the number of nodes an
# operation emits. Byte limits cover storage charged by the native operation,
# not the Python process or input conversion buffers.
print("Bounded count:", secured.model_count(limits=Limits(memory_bytes=1_000_000)))

# %%
# Operations on a shared vtree reuse scratch buffers automatically. Clear
# idle buffers between batches when retaining capacity is no longer useful.
vtree.clear_scratch()
print("Circuit remains usable:", secured.is_sat())
