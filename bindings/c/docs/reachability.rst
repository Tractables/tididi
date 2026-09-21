.. scenario: docs/scenarios.md#reachability

Explore a directed graph
========================

Starting at node 0, which nodes can we reach by following directed edges?

.. image:: _static/reachability.svg
   :alt: A sixteen-node directed graph with node 3 pointing outward and nodes 12 through 15 in a separate component.

Use one Boolean indicator per node. A state means exactly one indicator is true:

.. raw:: html

   <p><strong>S<sub>i</sub>(x<sub>0</sub>, …, x<sub>15</sub>) =
   x<sub>i</sub> ∧ ⋀<sub>j ≠ i</sub> ¬x<sub>j</sub></strong>.</p>

A second set of indicators describes the next state. The transition relation is

.. raw:: html

   <p><strong>T(x, x′) = ⋁<sub>(i,j) ∈ E</sub>
   (S<sub>i</sub>(x) ∧ S<sub>j</sub>(x′))</strong>.</p>

Name the variables
------------------

The edge list describes the graph. Each current and next indicator gets its own
variable in one shared vtree.

.. literalinclude:: ../examples/reachability.c
   :language: c
   :start-after: // begin: graph
   :end-before: // end: graph
   :dedent: 4

The helper constructs one state as a conjunction of sixteen signed literals.

.. literalinclude:: ../examples/reachability.c
   :language: c
   :start-after: // begin: state
   :end-before: // end: state

Compile the relation
--------------------

For each edge, conjoin its source state with its destination state, then union
all edges. The initial reachable set contains only node 0.

.. literalinclude:: ../examples/reachability.c
   :language: c
   :start-after: // begin: transition
   :end-before: // end: transition
   :dedent: 4

Take one step
-------------

Conjoin the current set with the transition relation. Quantify the current
variables to leave possible destinations, then rename next indicators to current
indicators. Count only the current variables; next variables are free after
renaming and should not multiply the number of states.

.. literalinclude:: ../examples/reachability.c
   :language: c
   :start-after: // begin: image
   :end-before: // end: image
   :dedent: 4

Output:

.. literalinclude:: _generated/reachability-image.txt
   :language: text

Reach a fixed point
-------------------

``tididi_and_exists`` performs the AND followed by existential quantification
from the preceding step in one call. It can avoid constructing the full
intermediate conjunction. Union the image with the states already reached,
minimize, and compare Boolean functions to detect convergence.

.. literalinclude:: ../examples/reachability.c
   :language: c
   :start-after: // begin: fixed_point
   :end-before: // end: fixed_point
   :dedent: 4

Output:

.. literalinclude:: _generated/reachability-fixed_point.txt
   :language: text

Inspect the result
------------------

Node 3 has edges leading into the reachable region but no directed path from
node 0 reaches it. Nodes 12 through 15 form a separate component. Test each
state by intersecting it with the final reachable set.

.. literalinclude:: ../examples/reachability.c
   :language: c
   :start-after: // begin: unreachable
   :end-before: // end: unreachable
   :dedent: 4

Output:

.. literalinclude:: _generated/reachability-unreachable.txt
   :language: text

Complete program
----------------

:download:`Download reachability.c <../examples/reachability.c>` and the
:download:`shared helper <../examples/example.h>`. The source includes cleanup
for all handles. Both files are included in the source checkout.
