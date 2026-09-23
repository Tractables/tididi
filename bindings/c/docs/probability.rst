.. scenario: docs/scenarios.md#probability

Evaluate probabilities
======================

A lawn is wet when it rains or the sprinkler runs: **W = R ∨ S**.
Assume independent rain, sprinkler and wind variables. Wind does not affect
wetness, but belongs to the model's universe.

Build events
------------

Build the wet event and the joint event **R ∧ W**.

.. literalinclude:: ../examples/probability.c
   :language: c
   :start-after: // begin: model
   :end-before: // end: model
   :dedent: 4

Assign probabilities
--------------------

For each variable, give the probabilities of false and true as exact fraction
strings. The weighted sum is the probability of the event. Conditional
probability divides the joint event's mass by the evidence's mass:
**P(R | W) = P(R ∧ W) / P(W)**. Here the evidence has positive probability.
The returned strings preserve exact arithmetic without a separate C rational type.

.. literalinclude:: ../examples/probability.c
   :language: c
   :start-after: // begin: probability
   :end-before: // end: probability
   :dedent: 4

Output:

.. literalinclude:: _generated/probability-probability.txt
   :language: text

Change a prior
--------------

Changing weights does not require rebuilding either event.

.. literalinclude:: ../examples/probability.c
   :language: c
   :start-after: // begin: new_prior
   :end-before: // end: new_prior
   :dedent: 4

Output:

.. literalinclude:: _generated/probability-new_prior.txt
   :language: text

Change observations
-------------------

An evaluator consumes its circuit and retains values for repeated evidence
updates. With no rain observed, its value is the joint probability **P(W ∧ ¬R)**.
Clearing observations restores the unobserved value; ``tididi_evaluator_finish``
returns the original circuit.

.. literalinclude:: ../examples/probability.c
   :language: c
   :start-after: // begin: observations
   :end-before: // end: observations
   :dedent: 4

Output:

.. literalinclude:: _generated/probability-observations.txt
   :language: text

Complete program
----------------

:download:`Download probability.c <../examples/probability.c>` and the
:download:`shared helper <../examples/example.h>`. The source includes cleanup
for all handles. Both files are included in the source checkout.
