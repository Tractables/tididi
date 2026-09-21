.. scenario: docs/scenarios.md#vtrees

Group related variables
=======================

The function **(x₁ ↔ x₃) ∧ (x₂ ↔ x₄)**
requires two pairs of variables to agree. A vtree determines how the variables
are grouped inside each circuit.

.. image:: _static/vtree-grouping.svg
   :alt: Two balanced vtrees, one grouping related variables together and one separating them.

Compile the same formula
------------------------

Negated XOR expresses agreement. Build the two agreements, conjoin them,
then minimize.

.. literalinclude:: ../examples/vtrees.c
   :language: c
   :start-after: // begin: formula
   :end-before: // end: formula

Compare groupings
-----------------

Both circuits represent four assignments. Their storage differs because one
vtree places related variables together. Build all operands on the selected
vtree; vtree shape alone does not make independently constructed domains shared.

.. literalinclude:: ../examples/vtrees.c
   :language: c
   :start-after: // begin: grouping
   :end-before: // end: grouping
   :dedent: 4

Output:

.. literalinclude:: _generated/vtrees-grouping.txt
   :language: text

Complete program
----------------

:download:`Download vtrees.c <../examples/vtrees.c>` and the
:download:`shared helper <../examples/example.h>`. The source includes cleanup
for all handles. Both files are included in the source checkout.
