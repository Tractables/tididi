.. scenario: docs/scenarios.md#persistence

Save related circuits
=====================

Save the shared vtree once and each circuit separately. In this example the two
circuits are backup rules: local or remote storage, and remote implies encryption.

Serialize
---------

The API returns owned text and byte buffers. Write the vtree text and each
buffer to your preferred files or storage; use ``tididi_bytes_data`` and
``tididi_bytes_len`` to obtain the bytes. Serialization borrows the circuits.

.. literalinclude:: ../examples/persistence.c
   :language: c
   :start-after: // begin: save
   :end-before: // end: save
   :dedent: 4

Restore one shared domain
-------------------------

Parse the vtree once, then load both circuits onto that same handle. They can
then be combined immediately.

.. literalinclude:: ../examples/persistence.c
   :language: c
   :start-after: // begin: restore
   :end-before: // end: restore
   :dedent: 4

Output:

.. literalinclude:: _generated/persistence-restore.txt
   :language: text

Complete program
----------------

:download:`Download persistence.c <../examples/persistence.c>` and the
:download:`shared helper <../examples/example.h>`. The source includes cleanup
for all handles. Both files are included in the source checkout.
