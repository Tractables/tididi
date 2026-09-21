.. scenario: docs/scenarios.md#first-circuit

Your first circuit
==================

Build the C package using the commands in the `package README
<https://github.com/Tractables/tididi/blob/main/bindings/c/README.rst>`_.
Your application's CMake file needs only:

.. code-block:: cmake

   find_package(tididi CONFIG REQUIRED)
   target_link_libraries(your_program PRIVATE tididi::tididi)

We will represent **(x ∧ y) ∨ z**: either both ``x`` and ``y``
are true, or ``z`` is true. A vtree groups the variables used by related circuits.
Here it contains three variables, numbered 1 through 3.

The examples include ``example.h``, which includes ``tididi.h`` and a few
small helpers. ``check`` handles an error; ``count`` calls the counting API:

.. literalinclude:: ../examples/example.h
   :language: c
   :start-at: static inline void check
   :end-before: /* The documentation

Construct the three literals, then combine them. A final ``NULL`` means no
operation limits. Initialize every owned output pointer to ``NULL``.

.. literalinclude:: ../examples/first_circuit.c
   :language: c
   :start-after: // begin: build
   :end-before: // end: build
   :dedent: 4

Output:

.. literalinclude:: _generated/first_circuit-build.txt
   :language: text

Keep a circuit for later
------------------------

``tididi_and`` consumes ``x`` and ``y``; ``tididi_or`` then consumes ``xy``
and ``z``. To keep a circuit while transforming it, copy it first. Here the copy
is consumed by negation while the original formula remains available.

.. literalinclude:: ../examples/first_circuit.c
   :language: c
   :start-after: // begin: copy
   :end-before: // end: copy
   :dedent: 4

Output:

.. literalinclude:: _generated/first_circuit-copy.txt
   :language: text

Release handles
---------------

Consumption moves the circuit out of its handle. The empty handle still needs
to be freed, along with the result handles. ``release`` calls
``tididi_circuit_free`` and checks its result.

.. literalinclude:: ../examples/first_circuit.c
   :language: c
   :start-after: // begin: cleanup
   :end-before: // end: cleanup
   :dedent: 4

Complete program
----------------

:download:`Download first_circuit.c <../examples/first_circuit.c>` and the
:download:`shared helper <../examples/example.h>`. The source includes cleanup
for all handles. Both files are included in the source checkout.
