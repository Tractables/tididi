.. scenario: docs/scenarios.md#api-overview

API reference
=============

Include ``tididi.h`` and link ``tididi::tididi`` from CMake. The header works in
both C and C++; all exported functions use the C calling convention.
:doc:`ownership` defines the common pointer and lifetime rules.

Construction and transformations return owned circuits through an output slot.
Queries borrow circuits. Optional final ``TididiLimits *`` arguments accept
``NULL`` for unrestricted work. Literal values are signed, one-based integers:
``3`` means variable 3 is true, ``-3`` means false; zero is invalid.

Types
-----

.. literalinclude:: ../include/tididi.h
   :language: c
   :start-after: #include <stdlib.h>
   :end-before: #ifdef __cplusplus

Functions
---------

.. include:: _generated/functions.rst
