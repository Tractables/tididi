.. scenario: docs/scenarios.md#reading-order

tididi for C
============

A circuit represents a Boolean function. You can combine circuits to describe
valid configurations, count or weight their solutions, and transform relations
to explore a transition system. The C interface calls the same implementation
as the Rust and Python libraries.

Start with a small formula, then choose a tutorial close to your application.
The later chapters cover ownership, resource limits, and custom evaluation.

.. toctree::
   :maxdepth: 1
   :numbered:

   getting_started
   configurations
   probability
   reachability
   tables
   persistence
   vtrees
   ownership
   execution
   minimum_cost
   statistics
   api
