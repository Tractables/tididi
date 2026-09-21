.. scenario: docs/scenarios.md#tables

Build and update a Boolean table
================================

Treat rows as a set of permission assignments. The columns indicate read, write
and share permission. Duplicate input rows represent the same assignment.

Load rows
---------

Pass a row-major array of bytes, each 0 or 1. Column IDs specify which variable
each position describes.

.. literalinclude:: ../examples/tables.c
   :language: c
   :start-after: // begin: rows
   :end-before: // end: rows
   :dedent: 4

Output:

.. literalinclude:: _generated/tables-rows.txt
   :language: text

Change the set
--------------

Insert the all-permissions row, then remove read-and-write without sharing.
An update consumes the old circuit and returns a minimized replacement.

.. literalinclude:: ../examples/tables.c
   :language: c
   :start-after: // begin: update
   :end-before: // end: update
   :dedent: 4

Output:

.. literalinclude:: _generated/tables-update.txt
   :language: text

Filter rows
-----------

Conjunction with the share literal selects rows that permit sharing.

.. literalinclude:: ../examples/tables.c
   :language: c
   :start-after: // begin: filter
   :end-before: // end: filter
   :dedent: 4

Output:

.. literalinclude:: _generated/tables-filter.txt
   :language: text

Complete program
----------------

:download:`Download tables.c <../examples/tables.c>` and the
:download:`shared helper <../examples/example.h>`. The source includes cleanup
for all handles. Both files are included in the source checkout.
