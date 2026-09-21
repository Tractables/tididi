# scenario: docs/scenarios.md#persistence

"""
Saving and loading related circuits
=======================================
Save the vtree once and each circuit separately. On reload, build one vtree
object and pass it to every circuit that needs to be combined.
"""
from tididi import Circuit, Vtree, literal

vtree = Vtree.balanced(3)
local = literal(vtree, 1)
remote = literal(vtree, 2)
encrypted = literal(vtree, 3)
destination = local | remote.copy()
encryption_rule = ~remote | encrypted

vtree_text = vtree.to_text()
destination_bytes = destination.to_bytes()
encryption_bytes = encryption_rule.to_bytes()

# %%
# These are ordinary Python strings and bytes, suitable for a database, file,
# or message. The serialization queries leave the original circuits usable.
# ``Circuit.save(path)`` and ``Circuit.load(vtree, path)`` also accept paths
# directly, including ``pathlib.Path`` objects.
restored_vtree = Vtree.from_text(vtree_text)
destination = Circuit.from_bytes(restored_vtree, destination_bytes)
encryption_rule = Circuit.from_bytes(restored_vtree, encryption_bytes)
configurations = destination & encryption_rule
print("Restored rules allow", configurations.model_count(), "configurations")

# %%
# The restored circuits share one domain. Rebuilding the same rules on that
# domain lets us check their Boolean equivalence.
local = literal(restored_vtree, 1)
remote = literal(restored_vtree, 2)
encrypted = literal(restored_vtree, 3)
expected = (local | remote.copy()) & (~remote | encrypted)
print("Equivalent to the original rules:", configurations.equivalent(expected))
