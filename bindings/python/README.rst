.. scenario: docs/scenarios.md#first-circuit

.. image:: https://raw.githubusercontent.com/Tractables/tididi/main/docs/logo.svg
   :alt: tididi
   :width: 400px
   :align: center

|ci| |python-docs| |arxiv| |license|

tididi for Python
=================

**Tree Decision Diagrams for Python.**

tididi represents Boolean functions as circuits called **Tree Decision Diagrams
(TDDs)**, so you can work with their satisfying assignments without listing
them. Build a circuit once, then combine it with other circuits or query it as
your inputs change. This package calls the same Rust implementation as the
`Rust crate <https://crates.io/crates/tididi>`_ and the
`C library <https://github.com/Tractables/tididi/blob/main/bindings/c/README.rst>`_.

Like binary decision diagrams, minimized TDDs are canonical and combine
efficiently under Boolean operations; in practice they can be much smaller.

Install
-------

The package is not on PyPI yet. Build it from a checkout, with Rust and
Python 3.10 or later installed:

.. code-block:: sh

   git clone https://github.com/Tractables/tididi
   python -m pip install ./tididi/bindings/python

Getting started
---------------

This example builds ``(x ∧ y) ∨ z`` and counts its satisfying assignments:

.. code-block:: python

   from tididi import Vtree, literal

   vtree = Vtree.balanced(3)
   x = literal(vtree, 1)
   y = literal(vtree, 2)
   z = literal(vtree, 3)
   f = (x & y) | z
   print("Satisfying assignments:", f.model_count())

Output:

.. code-block:: text

   Satisfying assignments: 5

Four assignments have ``z`` true, and one more has ``z`` false with ``x`` and
``y`` true.

A **vtree** is a binary tree that groups the variables; circuits you combine
must share the same vtree. Literals are signed variable numbers: ``1`` is ``x``
and ``-2`` is ``¬y``. Boolean operations consume their operands, so use
``a.copy() & b.copy()`` to keep both; queries such as ``model_count()`` leave a
circuit usable, and reusing a consumed circuit raises ``ConsumedCircuitError``.

The tutorials in the `Python guide`_ cover what the library does, with executed
output and downloadable scripts and notebooks:

- **Build and transform functions** by combining circuits, fixing variables
  and quantifying them out: the `reachability tutorial`_ explores a transition
  system.
- **Find and count solutions**, and test equivalence or implication: the
  `configuration tutorial`_ counts a backup application's valid configurations
  as a user changes their choices.
- **Evaluate weighted models** such as probabilities, changing the weights
  without rebuilding the circuit: the `probability tutorial`_ answers a
  conditional-probability query, and custom algebras compute other quantities,
  such as a `minimum configuration cost`_.

Further tutorials cover tables, saving and loading, variable grouping, resource
limits and circuit statistics. The `API reference`_ documents every function.

Applications built on tididi
----------------------------

- `tididi-cnf <https://github.com/Tractables/tididi-cnf>`_ compiles CNF formulas
  into TDDs for model counting, weighted model counting and projected model
  counting, from Rust, Python or C.
- `tddlog <https://github.com/Tractables/tddlog>`_ performs exact probabilistic
  inference for higher-level languages: ProbLog programs, Bayesian and Markov
  networks, and discrete probabilistic programs. It has Python, C and
  WebAssembly bindings and `runs in the browser <https://tractables.github.io/tddlog/>`_.

Develop
-------

From ``bindings/python`` in a checkout, build in place and run the checks:

.. code-block:: sh

   python -m venv .venv
   . .venv/bin/activate
   python -m pip install 'maturin>=1.9.4,<2'
   maturin develop --release
   python -m pip install '.[test,docs]'
   python -m pytest
   python -m sphinx -W --keep-going -b html docs docs/_build/html

On Windows activate with ``.venv\Scripts\activate`` instead. Sphinx executes
the tutorials while building the pages; open ``docs/_build/html/index.html``.

Citing
------

Please cite `A Canonical Generalization of OBDD <https://arxiv.org/abs/2604.05537>`_,
which introduced TDDs.

Acknowledgements
----------------

This material is based upon work supported by the Defense Advanced Research
Projects Agency (DARPA) under Agreement No. HR0011269E224.

.. _Python guide: https://tractables.github.io/tididi/python/
.. _reachability tutorial: https://tractables.github.io/tididi/python/tutorials/03_reachability.html
.. _configuration tutorial: https://tractables.github.io/tididi/python/tutorials/01_configurations.html
.. _probability tutorial: https://tractables.github.io/tididi/python/tutorials/02_probability.html
.. _minimum configuration cost: https://tractables.github.io/tididi/python/tutorials/08_minimum_cost.html
.. _API reference: https://tractables.github.io/tididi/python/api.html

.. |ci| image:: https://github.com/Tractables/tididi/actions/workflows/rust.yml/badge.svg
   :target: https://github.com/Tractables/tididi/actions/workflows/rust.yml
   :alt: CI
.. |python-docs| image:: https://img.shields.io/badge/docs-Python-blue
   :target: https://tractables.github.io/tididi/python/
   :alt: Python guide
.. |arxiv| image:: https://img.shields.io/badge/arXiv-2604.05537-b31b1b.svg
   :target: https://arxiv.org/abs/2604.05537
   :alt: arXiv
.. |license| image:: https://img.shields.io/badge/license-Apache--2.0-blue.svg
   :target: https://github.com/Tractables/tididi/blob/main/LICENSE
   :alt: License: Apache-2.0
