.. scenario: docs/scenarios.md#api-overview

Python API
==============

Construct circuits on a shared :class:`~tididi.Vtree`; combine them with
operators or the named functions below. Transformation methods consume their
circuit inputs. Query methods borrow them. See :doc:`ownership` for the rules
that apply across the API.

.. currentmodule:: tididi

Domains and values
----------------------

.. autoclass:: Vtree
   :members:

.. autoclass:: Literal
   :members:

Construction
----------------

.. autofunction:: literal
.. autofunction:: cube
.. autofunction:: clause
.. autofunction:: from_models
.. autofunction:: one
.. autofunction:: zero

Boolean composition
-----------------------

.. autofunction:: and_
.. autofunction:: or_
.. autofunction:: xor
.. autofunction:: ite
.. autofunction:: or_many
.. autofunction:: and_exists

Circuits and counters
-------------------------

.. autoclass:: Circuit
   :members:

.. autoclass:: Counter
   :members:

Custom evaluation
---------------------

.. autoclass:: Algebra
   :members:

Execution and errors
------------------------

.. autoclass:: Limits
.. autoexception:: ConsumedCircuitError
.. autoexception:: ResourceLimitError
