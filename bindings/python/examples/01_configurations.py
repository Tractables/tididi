# scenario: docs/scenarios.md#configurations

"""
Configuration rules
=======================
A backup application offers local storage, remote storage, encryption, and
notifications. It needs at least one storage destination; remote backups
require encryption. The rule is

``(L ∨ R) ∧ (¬R ∨ E)``.

Notifications are optional. We will count the valid configurations, inspect
forced choices, and follow a user changing their selections.
"""
from tididi import Literal, Vtree, cube, literal

vtree = Vtree.balanced(4)
local_choice = Literal(1)
remote_choice = Literal(2)
encrypted_choice = Literal(3)
notifications_choice = Literal(4)
names = {1: "local backups", 2: "remote backups", 3: "encryption", 4: "notifications"}

local = literal(vtree, local_choice)
remote = literal(vtree, remote_choice)
encrypted = literal(vtree, encrypted_choice)
notifications = literal(vtree, notifications_choice)

# %%
# Build and count
# -------------------
# ``remote.copy()`` creates an operand for this operation so the original
# remains available. The resulting circuit owns its storage.
destination = local | remote.copy()
encryption_rule = ~remote.copy() | encrypted.copy()
configurations = destination & encryption_rule
print("Valid configurations:", configurations.model_count())
with_notifications = configurations.copy() & notifications
print("With notifications:", with_notifications.model_count())

# %%
# Inspect a user's choices
# ----------------------------
# Requiring remote backups leaves only configurations with encryption.
with_remote = configurations.copy() & remote
print("With remote backups:", with_remote.model_count())
for choice in with_remote.implied_literals():
    print(f"Required choice: {names[choice.variable]} = {choice.sign}")
remote_without_encryption = with_remote & ~encrypted
print("Satisfiable:", remote_without_encryption.is_sat())

# %%
# A witness chooses one value for every variable. It is a list of immutable
# ``Literal`` values that can be reused to construct a cube.
witness = configurations.satisfying_assignment()
for choice in witness:
    print(f"{names[choice.variable]}: {choice.sign}")
selected = configurations.copy() & cube(vtree, witness)
print("Selected valid configurations:", selected.model_count())

# %%
# Count under changing observations
# -------------------------------------
# A counter owns the circuit while it retains cached counts. Creating it
# consumes ``configurations``. Observations use the named literal values,
# which are reusable and do not own circuits.
counter = configurations.counter()
counter.observe([remote_choice])
print("Matching configurations:", counter.model_count())
counter.observe([~notifications_choice])
print("Matching configurations:", counter.model_count())
counter.observe([~remote_choice, ~encrypted_choice])
print("Matching configurations:", counter.model_count())
counter.clear_observations()
print("Matching configurations:", counter.model_count())

# %%
# Finish the counter to recover its original diagram. Its observations never
# changed that diagram. Minimization returns a canonical circuit for this vtree.
configurations = counter.finish().minimize()
print("Valid configurations:", configurations.model_count())
print("Minimized pairs:", configurations.pair_count())
