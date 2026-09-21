A first circuit
===================

From a source checkout, install the Python package with Rust available:

.. code-block:: console

   python -m pip install ./bindings/python

Once installed, ordinary use needs only Python. A vtree groups the variables
that your circuits will share. Here we have three Boolean variables and the
function ``(x ∧ y) ∨ z``:

.. doctest::

   >>> from tididi import Vtree, literal
   >>> vtree = Vtree.balanced(3)
   >>> x = literal(vtree, 1)
   >>> y = literal(vtree, 2)
   >>> z = literal(vtree, 3)
   >>> f = (x & y) | z
   >>> print(f.model_count())
   5

There are four assignments with ``z`` true and one more with both ``x`` and
``y`` true while ``z`` is false. All variables in the vtree participate in
counting, including ones a function leaves free.

Use ``&``, ``|``, ``^``, and ``~`` for conjunction, disjunction, exclusive-or,
and negation. Parenthesize compound expressions. Python's ``and``, ``or``, and
``not`` test an object's truth value and cannot construct circuits; tididi
raises ``TypeError`` if a circuit is used that way.

Operations consume their circuit operands. The result ``f`` above is usable;
``x``, ``y``, and ``z`` are consumed. Querying a circuit leaves it usable, and
``copy()`` makes an explicit copy for a transformation:

.. doctest::

   >>> complement = ~f.copy()
   >>> print(f.model_count(), complement.model_count())
   5 3

Use the same vtree for circuits you intend to combine. ``literal(vtree, -2)``
builds the negation of variable 2. For named choices that you can pass repeatedly
to construction and observation operations, use :class:`tididi.Literal`:

.. doctest::

   >>> from tididi import Literal
   >>> enabled = Literal(2)
   >>> disabled = ~enabled
   >>> print(enabled.variable, enabled.sign, int(disabled))
   2 True -2

Continue with :doc:`tutorials/01_configurations` to build and query a complete
set of rules, or read :doc:`ownership` for copying, aliasing, and error behavior.
