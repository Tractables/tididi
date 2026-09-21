.. scenario: docs/scenarios.md#reading-order

tididi for Python
=====================

tididi represents Boolean functions as Tree Decision Diagrams (TDDs). Build a
circuit from rules, combine it with other circuits, and query the satisfying
assignments without listing them. Counts are exact Python integers; weighted
sums can be exact fractions.

Start with :doc:`getting_started`. The configuration tutorial builds a small
rule system and follows a user's choices. The reachability tutorial uses
conjunction, quantification, and renaming to explore a directed graph.

.. toctree::
   :maxdepth: 1

   getting_started
   ownership

.. toctree::
   :caption: Worked examples
   :numbered:
   :maxdepth: 1

   tutorials/01_configurations
   counting
   tutorials/02_probability
   tutorials/03_reachability
   tutorials/04_tables
   tutorials/05_persistence
   tutorials/06_vtrees
   tutorials/07_execution
   tutorials/08_minimum_cost
   tutorials/09_statistics

.. toctree::
   :caption: Reference
   :maxdepth: 1

   api
   representation

Download an application tutorial as a runnable program or notebook from the
bottom of its page.
