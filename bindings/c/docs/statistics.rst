.. scenario: docs/scenarios.md#statistics

Inspect circuit size
====================

A circuit for **x₁ XOR x₂** over four variables has
many assignments but little stored structure. Model count measures assignments;
node and pair counts measure the representation.

Inspect storage
---------------

Obtain totals, then inspect a snapshot to find the largest node. The snapshot
owns its data and remains usable if the circuit is later transformed or freed.

.. literalinclude:: ../examples/statistics.c
   :language: c
   :start-after: // begin: sizes
   :end-before: // end: sizes
   :dedent: 4

Output:

.. literalinclude:: _generated/statistics-sizes.txt
   :language: text

Complete program
----------------

:download:`Download statistics.c <../examples/statistics.c>` and the
:download:`shared helper <../examples/example.h>`. The source includes cleanup
for all handles. Both files are included in the source checkout.
