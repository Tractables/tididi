.. scenario: docs/scenarios.md#execution

Handle operation limits
=======================

Ordinary calls pass ``NULL`` for limits. When processing input of uncertain size,
initialize a ``TididiLimits`` with ``tididi_limits_default`` and set the fields
you need: memory bytes, emitted nodes, or a cooperative timeout in seconds.

Keep inputs for a retry
-----------------------

This example combines local-or-remote storage with encryption. A zero memory
budget makes the first attempt fail. The originals survive because only copies
were passed to the consuming operation; a retry uses fresh copies.

.. literalinclude:: ../examples/execution.c
   :language: c
   :start-after: // begin: retry
   :end-before: // end: retry
   :dedent: 4

Output:

.. literalinclude:: _generated/execution-retry.txt
   :language: text

Complete program
----------------

:download:`Download execution.c <../examples/execution.c>` and the
:download:`shared helper <../examples/example.h>`. The source includes cleanup
for all handles. Both files are included in the source checkout.
