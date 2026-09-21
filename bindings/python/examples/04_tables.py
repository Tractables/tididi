"""
Tables and updates
======================
A permission table records which combinations of read, write, and share are
allowed. Construct a circuit directly from Boolean rows, update the allowed
set, and query it with another Boolean condition.
"""
from tididi import Vtree, from_models, literal

vtree = Vtree.balanced(3)
read, write, share = 1, 2, 3
rows = [
    (True, False, False),
    (True, True, False),
    (True, False, True),
    (True, True, False),
]
permissions = from_models(vtree, [read, write, share], rows)
print("Distinct permission sets:", permissions.model_count())

# %%
# Rows describe a set: the repeated row contributes only once. The positions
# in each row correspond to the supplied variable list. Other vtree variables,
# if any, would be free.
#
# Insert the combination with all three permissions and remove read/write
# without sharing. Updates use signed literals; ``-share`` means sharing is
# false. The operation consumes the old circuit and returns its replacement.
permissions = permissions.update(insert=[[read, write, share]], remove=[[read, write, -share]])
print("After updates:", permissions.model_count())
sharing = permissions & literal(vtree, share)
print("Permission sets allowing sharing:", sharing.model_count())

# %%
# An update can also describe several rows at once. A partial cube leaves its
# omitted variables free: removing ``[share]`` would remove every combination
# that permits sharing. An empty cube matches all assignments.
