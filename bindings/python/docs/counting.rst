.. scenario: docs/scenarios.md#counting-choices

What are we counting?
=====================

A backup configuration has local storage **L**, remote storage **R**, encryption
**E**, and notifications **N**. The rules are **(L ∨ R) ∧ (¬R ∨ E)**;
notifications are optional. We will distinguish counting configurations that
match a choice, substituting a value into the rules, and counting only selected
options.

.. doctest::

   >>> from tididi import Literal, Vtree, clause, literal
   >>> vtree = Vtree.balanced(4)
   >>> local, remote, encrypted, notifications = [Literal(i) for i in range(1, 5)]
   >>> rules = clause(vtree, [local, remote]) & clause(vtree, [~remote, encrypted])
   >>> print("Valid configurations:", rules.model_count())
   Valid configurations: 8

Keep configurations consistent with a choice
--------------------------------------------

Selecting remote backups means counting **rules ∧ R**. Conjoin the remote
literal, or observe it through a counter when choices will change repeatedly:

.. doctest::

   >>> selected = rules.copy() & literal(vtree, remote)
   >>> print("With remote backups:", selected.model_count())
   With remote backups: 4
   >>> counter = rules.copy().counter()
   >>> counter.observe([remote])
   >>> print("Observed remote backups:", counter.model_count())
   Observed remote backups: 4

Remote and encryption are true. Local storage and notifications each have two
choices, giving four configurations.

Substitute a value into the rules
---------------------------------

Substituting **R = true** simplifies the rules to **E**. The new function no
longer depends on R, but its vtree still includes R. An ordinary count includes
both values of that free variable. Count only the remaining options to exclude it:

.. doctest::

   >>> residual = rules.copy().condition([remote])
   >>> print("After substituting remote = true:", residual.model_count())
   After substituting remote = true: 8
   >>> remaining = [local.variable, encrypted.variable, notifications.variable]
   >>> print("Distinct remaining choices:", residual.projected_model_count(remaining))
   Distinct remaining choices: 4

Use :meth:`~tididi.Circuit.condition` when you need the remaining function;
use observation or conjunction when you want to retain the selected value.

Count distinct choices for a subset of options
----------------------------------------------

Local only, remote only, and both are valid storage destinations: three choices.
A projected count excludes the variations in encryption and notifications:

.. doctest::

   >>> destinations = [local.variable, remote.variable]
   >>> print("Valid destination choices:", rules.projected_model_count(destinations))
   Valid destination choices: 3
   >>> destination_rule = rules.copy().exists([encrypted.variable, notifications.variable])
   >>> print("Destination rule over the full vtree:", destination_rule.model_count())
   Destination rule over the full vtree: 12
   >>> print("Destination rule projected:", destination_rule.projected_model_count(destinations))
   Destination rule projected: 3

Existential quantification builds **L ∨ R**, leaving E and N free in the unchanged
vtree. Use :meth:`~tididi.Circuit.projected_model_count` when you need the count,
and :meth:`~tididi.Circuit.exists` when you need a circuit for further composition.
The :doc:`reachability tutorial <tutorials/03_reachability>` uses that second form.
