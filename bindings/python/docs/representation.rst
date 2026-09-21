.. scenario: docs/scenarios.md#representation

Functions, vtrees, and circuit size
=======================================

A Boolean function identifies a set of assignments. Its TDD stores that function
as a circuit, sharing repeated subfunctions. A vtree groups the variables:
each internal vtree node divides them into a left and a right group.

.. image:: /_static/tdd-basics.svg
   :alt: A vtree groups x and y; the TDD joins pairs for x and not y, or not x and y.
   :width: 700

At an internal circuit node, each pair conjoins a left subfunction with a right
subfunction. The node disjoins its pairs. The representation's determinism
keeps those alternatives disjoint, which permits exact counting by combining
counts rather than enumerating assignments.

TDDs can be much smaller in practice than binary decision diagrams. Minimizing
a TDD makes it canonical for its vtree: equivalent functions have the same
diagram up to storage numbering. Use ``equivalent()`` to compare functions;
Python object identity and local node IDs do not establish equivalence.

The choice of variable grouping can change circuit size substantially. The
:doc:`tutorials/06_vtrees` walkthrough compares two groupings of the same
function. ``node_count()`` and ``pair_count()`` describe stored structure;
neither is the number of satisfying assignments.

A circuit owns its nodes and retains its vtree. The vtree also provides reusable
scratch storage for operations; ``clear_scratch()`` releases idle buffers.
There is no manager object to keep alive separately. Two independently
constructed vtrees are distinct domains even when they have the same shape;
reuse one object, or obtain it from ``circuit.vtree``, when combining circuits.

The representation and minimization algorithm are described in
`A Canonical Generalization of OBDD <https://arxiv.org/abs/2604.05537>`_.
