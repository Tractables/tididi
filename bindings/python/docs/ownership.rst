.. scenario: docs/scenarios.md#ownership

Using and reusing circuits
==============================

A ``Circuit`` owns diagram storage. A transformation moves that storage into
its result; the old Python object remains present but is marked consumed.
Queries such as ``model_count()``, ``is_sat()``, and ``equivalent()`` borrow
the circuit and can be repeated.

.. doctest::

   >>> from tididi import Vtree, literal, ConsumedCircuitError
   >>> vtree = Vtree.balanced(2)
   >>> a = literal(vtree, 1)
   >>> b = literal(vtree, 2)
   >>> result = a & b
   >>> print(result.model_count(), a.is_consumed, b.is_consumed)
   1 True True
   >>> try:
   ...     a.model_count()
   ... except ConsumedCircuitError as error:
   ...     print(error)
   Circuit has been consumed. Copy it before passing it to a consuming operation.

Copies and aliases
----------------------

``a.copy()`` duplicates the circuit's storage and shares its vtree. Python's
``copy.copy(a)`` and ``copy.deepcopy(a)`` have the same behavior: the vtree
stays shared so the copies can still be combined.

Assignment creates an alias, not a copy:

.. doctest::

   >>> a = literal(vtree, 1)
   >>> alias = a
   >>> saved = a.copy()
   >>> negated = ~a
   >>> print(alias.is_consumed, saved.model_count(), negated.model_count())
   True 2 2

The same object cannot occupy two consuming operand positions. This includes
``a & a``, aliases of ``a``, and repeated entries in ``or_many``. Use
``a.copy() & a`` when you need two occurrences. Validation rejects duplicates
before consuming any operand. Borrowing queries such as ``a.equivalent(a)``
accept aliases.

Errors and resource limits
------------------------------

Both operators and named functions raise Python exceptions. A wrong argument
type, a consumed operand, mismatched vtrees, or an invalid variable is rejected
before the wrapper transfers valid operands. Once the Rust computation starts,
a consuming operation keeps its inputs consumed even if it fails. Copy before
the call if you need the original for a retry.

``MemoryError`` reports a refused allocation or byte budget;
:class:`tididi.ResourceLimitError` reports a deadline or output-node cap;
``ValueError`` reports invalid input; ``OverflowError`` reports an exhausted
index range. Borrowing queries leave their inputs usable on failure.

``circuit = circuit.minimize()`` and ``circuit = circuit.update(...)`` follow
the same consuming convention. A counter takes ownership with
``counter = circuit.counter()``; ``circuit = counter.finish()`` returns the
original diagram and discards the counter's observations and cache.

Threaded callers
--------------------

Native construction, Boolean operations, counting, and weighted evaluation
release the interpreter lock while they run. Each operation uses the calling
thread; tididi starts no worker threads. Use independent circuit objects for
concurrent transformations. Attempts to mutate an object already borrowed by
another operation raise a runtime borrowing error. Python algebra callbacks
run with the interpreter attached and propagate their exceptions.
