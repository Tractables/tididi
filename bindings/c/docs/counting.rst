.. scenario: docs/scenarios.md#counting-choices

What are we counting?
=====================

A backup configuration has local storage **L**, remote storage **R**, encryption
**E**, and notifications **N**. The rules are **(L ∨ R) ∧ (¬R ∨ E)**;
notifications are optional. Three similar operations answer different questions:
count configurations that match a choice, substitute its value into the rules,
or count only selected options.

.. literalinclude:: ../examples/counting.c
   :language: c
   :start-after: // begin: rules
   :end-before: // end: rules
   :dedent: 4

Output:

.. literalinclude:: _generated/counting-rules.txt
   :language: text

Keep configurations consistent with a choice
--------------------------------------------

Selecting remote backups means counting **rules ∧ R**. Conjunction creates
that set; a counter answers the same question while retaining its counting cache:

.. literalinclude:: ../examples/counting.c
   :language: c
   :start-after: // begin: observe
   :end-before: // end: observe
   :dedent: 4

Output:

.. literalinclude:: _generated/counting-observe.txt
   :language: text

Remote and encryption are true. Local storage and notifications each have two
choices, giving four configurations.

Substitute a value into the rules
---------------------------------

Substituting **R = true** simplifies the rules to **E**. The new function no
longer depends on R, but the vtree still contains it. An ordinary count includes
both values of that free variable. A projected count over the remaining options
excludes it:

.. literalinclude:: ../examples/counting.c
   :language: c
   :start-after: // begin: substitute
   :end-before: // end: substitute
   :dedent: 4

Output:

.. literalinclude:: _generated/counting-substitute.txt
   :language: text

Use :c:func:`tididi_condition` for the remaining function; use observation or
conjunction to retain the selected value.

Count distinct choices for a subset of options
----------------------------------------------

Local only, remote only, and both are valid destinations: three choices.
A projected count excludes variations in encryption and notifications:

.. literalinclude:: ../examples/counting.c
   :language: c
   :start-after: // begin: project
   :end-before: // end: project
   :dedent: 4

Output:

.. literalinclude:: _generated/counting-project.txt
   :language: text

Existential quantification builds **L ∨ R**, leaving E and N free in the unchanged
vtree. Use :c:func:`tididi_projected_model_count` for the count and
:c:func:`tididi_exists` for a circuit you can combine with other rules, as in
:doc:`reachability`.

Complete program
----------------

:download:`Download counting.c <../examples/counting.c>` and the
:download:`shared helper <../examples/example.h>`. Cleanup of consumed handles
appears in the complete program.
