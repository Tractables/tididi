.. scenario: docs/scenarios.md#ownership

Ownership and errors
====================

A ``TididiCircuit *`` is an owned handle. Queries borrow its circuit;
transformations such as ``tididi_and`` consume the circuit stored inside.
The emptied handle remains valid until you call ``tididi_circuit_free``.
Reusing it reports ``TIDIDI_ERROR_CODE_CONSUMED_CIRCUIT``. Passing it to
``tididi_copy`` before the transformation makes a separate circuit you can keep.

Every owned output pointer starts as ``NULL``. After freeing an output, set its
variable to ``NULL`` before using that variable as an output slot again. This
avoids overwriting another live allocation. Scalar outputs need valid storage
and remain unchanged on failure.

Error handling
--------------

A function returning ``TididiError *`` returns ``NULL`` on success. Otherwise,
read its code or message, then free the error. The tutorial ``check`` helper
prints the message and exits; an application usually handles the error locally.
For example, :doc:`execution` retries an operation after a memory-limit error.

Argument checks happen before inputs are consumed: duplicate handles, mismatched
vtrees and invalid variable lists preserve their inputs. Once execution begins,
a consuming operation may leave its inputs consumed even on failure. Retain
copies when you need to retry. ``tididi_is_consumed`` can inspect a live handle.

Two circuits combined in one operation must share a vtree. Constructing two
identical vtrees produces two different domains. ``tididi_circuit_vtree`` obtains
another handle to an existing circuit's domain, and :doc:`persistence` shows how
to restore several circuits onto one domain.

Cleanup
-------

Free every owned result with its matching function, including consumed circuit
handles and finished counters. Vtrees remain alive internally while circuits
need them, so the original vtree handle may be freed earlier. Data returned by
``*_data`` is borrowed from its list or buffer and must not be freed separately.
Use ``tididi_string_free`` for returned strings, never C's ``free``.

As with other C libraries, freed handles and invalid pointers cannot be checked:
do not reuse them. Pointer arguments must refer to live, aligned storage of the
declared type. Output storage must not overlap input storage. An array pointer
may be ``NULL`` only when its length is zero. Strings are NUL-terminated UTF-8.

Callbacks and threads
---------------------

Synchronize access to each handle across threads; the library creates no threads.
Separate circuit handles may share one vtree. Evaluation callbacks and their
``userdata`` remain borrowed only for the duration of ``tididi_evaluate_f64``.
A callback that tries to consume or free the circuit being evaluated receives
``TIDIDI_ERROR_CODE_BORROW_CONFLICT``. Do not throw a C++ exception or use
``longjmp`` across a callback boundary.

The binding catches unwinding Rust panics and reports an internal error. Invalid
C pointers, process-level allocation failure, and builds configured to abort on
panic cannot be converted into errors.
