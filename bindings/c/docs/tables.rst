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

The updated table contains read-only, read-and-share, and all three permissions.
The count is unchanged, but its rows differ.

Filter rows
-----------

Conjunction with the share literal selects rows that permit sharing. Filter a
copy so the full table stays available for the next update.

.. literalinclude:: ../examples/tables.c
   :language: c
   :start-after: // begin: filter
   :end-before: // end: filter
   :dedent: 4

Output:

.. literalinclude:: _generated/tables-filter.txt
   :language: text

Remove a group of rows
----------------------

Suppose sharing is withdrawn entirely. A partial assignment names just the
permissions to match: ``{SHARE}`` selects every row where sharing is true,
whatever its read and write values. Pass no insertions and one deletion cube:

.. literalinclude:: ../examples/tables.c
   :language: c
   :start-after: // begin: bulk_remove
   :end-before: // end: bulk_remove
   :dedent: 4

Output:

.. literalinclude:: _generated/tables-bulk_remove.txt
   :language: text

Only read access without write or share remains. Removing matching rows does
not change their share column to false. An empty cube matches all assignments.

Complete program
----------------

:download:`Download tables.c <../examples/tables.c>` and the
:download:`shared helper <../examples/example.h>`. The source includes cleanup
for all handles. Both files are included in the source checkout.
