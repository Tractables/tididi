.. scenario: docs/scenarios.md#configurations

Choose valid configurations
===========================

A backup service offers local storage, remote storage, encryption and notifications.
At least one destination is required, and remote storage requires encryption:

**(L ∨ R) ∧ (¬R ∨ E)**.

Notifications are optional. They remain part of each complete configuration,
even though they do not occur in the rules.

Build the rules
---------------

A clause is a disjunction of signed literals. Join the two clauses with AND.

.. literalinclude:: ../examples/configurations.c
   :language: c
   :start-after: // begin: rules
   :end-before: // end: rules
   :dedent: 4

Output:

.. literalinclude:: _generated/configurations-rules.txt
   :language: text

Select remote storage
---------------------

Intersect the valid configurations with the remote literal, then inspect what
that choice forces. Positive returned literals are true; negative ones are false.

.. literalinclude:: ../examples/configurations.c
   :language: c
   :start-after: // begin: forced
   :end-before: // end: forced
   :dedent: 4

Output:

.. literalinclude:: _generated/configurations-forced.txt
   :language: text

Change choices repeatedly
-------------------------

A counter owns a circuit and reuses its counting cache while observations change.
Observations restrict the assignments counted; they do not consume the counter.
Copy the rules when you also need to keep them outside the counter.

.. literalinclude:: ../examples/configurations.c
   :language: c
   :start-after: // begin: observe
   :end-before: // end: observe
   :dedent: 4

Output:

.. literalinclude:: _generated/configurations-observe.txt
   :language: text

Complete program
----------------

:download:`Download configurations.c <../examples/configurations.c>` and the
:download:`shared helper <../examples/example.h>`. The source includes cleanup
for all handles. Both files are included in the source checkout.
