.. scenario: docs/scenarios.md#minimum-cost

Find a cheapest configuration
=============================

Reuse the backup rules **(L ∨ R) ∧ (¬R ∨ E)**.
Local storage costs 5, remote storage 2, encryption 1, and notifications 0.
To find the cheapest valid choice, evaluate alternatives with minimum and
independent choices with addition.

Define the evaluation
---------------------

A false branch has infinite cost. A true literal pays its price; a false
literal costs zero. A free variable chooses its cheaper sign. The callback
interface passes your price array through ``userdata``.

.. literalinclude:: ../examples/minimum_cost.c
   :language: c
   :start-after: // begin: algebra
   :end-before: // end: algebra

Evaluate and change prices
--------------------------

The same circuit can be evaluated with another price list. Values here are
``double``; exact probabilities use the separate rational-weight API.

.. literalinclude:: ../examples/minimum_cost.c
   :language: c
   :start-after: // begin: cost
   :end-before: // end: cost
   :dedent: 4

Output:

.. literalinclude:: _generated/minimum_cost-cost.txt
   :language: text

Recognize an impossible choice
------------------------------

Remote storage without encryption violates the rules. Its minimum cost is
infinity, so there is no feasible configuration.

.. literalinclude:: ../examples/minimum_cost.c
   :language: c
   :start-after: // begin: conflict
   :end-before: // end: conflict
   :dedent: 4

Output:

.. literalinclude:: _generated/minimum_cost-conflict.txt
   :language: text

Complete program
----------------

:download:`Download minimum_cost.c <../examples/minimum_cost.c>` and the
:download:`shared helper <../examples/example.h>`. The source includes cleanup
for all handles. Both files are included in the source checkout.
