.. scenario: docs/scenarios.md#first-circuit

.. image:: https://raw.githubusercontent.com/Tractables/tididi/main/docs/logo.svg
   :alt: tididi
   :width: 400px
   :align: center

|ci| |c-docs| |arxiv|

tididi for C and C++
====================

**Tree Decision Diagrams for C and C++.**

tididi represents Boolean functions as circuits called **Tree Decision Diagrams
(TDDs)**, so you can work with their satisfying assignments without listing
them. Build a circuit once, then combine it with other circuits or query it as
your inputs change. This library exposes the same Rust implementation as the
`Rust crate <https://crates.io/crates/tididi>`_ and the
`Python package <https://github.com/Tractables/tididi/blob/main/bindings/python/README.rst>`_
through a C interface, usable from C and C++.

Like binary decision diagrams, minimized TDDs are canonical and combine
efficiently under Boolean operations; in practice they can be much smaller.

Install
-------

Each `GitHub release <https://github.com/Tractables/tididi/releases>`_ has a
prebuilt package for Linux (x86-64 and ARM64), macOS (ARM64 and x86-64) and
Windows. Unpack it anywhere; it needs no Rust toolchain. The Linux packages
need glibc 2.34 or later (Ubuntu 22.04, Debian 12, RHEL 9 and newer).

To build and install the package from a checkout instead, with Rust, CMake and
a C/C++ compiler installed:

.. code-block:: sh

   git clone https://github.com/Tractables/tididi
   cd tididi
   cmake -S bindings/c -B build/c -DCMAKE_BUILD_TYPE=Release
   cmake --build build/c --config Release --parallel 8
   ctest --test-dir build/c -C Release --output-on-failure
   cmake --install build/c --config Release --prefix /your/install/prefix

An application then needs only ``find_package(tididi CONFIG REQUIRED)`` and
``target_link_libraries(your_program PRIVATE tididi::tididi)``, with
``CMAKE_PREFIX_PATH`` set to the unpacked or installed package. On Windows,
keep ``tididi_c.dll`` beside your executable or on ``PATH``.

Getting started
---------------

This example builds ``(x ∧ y) ∨ z`` and counts its satisfying assignments.
``check`` stops on an error and ``count`` returns the model count; both are
small helpers from the guide's `example.h`_.

.. code-block:: c

   TididiVtree *vtree = NULL;
   TididiCircuit *x = NULL, *y = NULL, *z = NULL;
   TididiCircuit *xy = NULL, *formula = NULL;
   check(tididi_vtree_balanced(3, &vtree));
   check(tididi_literal(vtree, 1, &x, NULL));
   check(tididi_literal(vtree, 2, &y, NULL));
   check(tididi_literal(vtree, 3, &z, NULL));
   check(tididi_and(x, y, &xy, NULL));
   check(tididi_or(xy, z, &formula, NULL));
   printf("Satisfying assignments: %" PRIu64 "\n", count(formula));

Output:

.. code-block:: text

   Satisfying assignments: 5

Four assignments have ``z`` true, and one more has ``z`` false with ``x`` and
``y`` true.

A **vtree** is a binary tree that groups the variables; circuits you combine
must share the same vtree. Literals are signed variable numbers: ``1`` is ``x``
and ``-2`` is ``¬y``. Operations consume their circuit operands, so
``tididi_copy`` a circuit first to keep it, and free every handle with
``tididi_circuit_free`` or ``tididi_vtree_free``. A final ``NULL`` argument
means no operation limits.

The chapters of the `C guide`_ cover what the library does, with compiled
programs and their captured output:

- **Build and transform functions** by combining circuits, fixing variables
  and quantifying them out: the `reachability chapter`_ explores a transition
  system.
- **Find and count solutions**, and test equivalence or implication: the
  `configuration chapter`_ counts a backup application's valid configurations
  as a user changes their choices.
- **Evaluate weighted models** such as probabilities, changing the weights
  without rebuilding the circuit: the `probability chapter`_ answers a
  conditional-probability query, and custom algebras compute other quantities,
  such as a `minimum configuration cost`_.

Further chapters cover tables, saving and loading, variable grouping,
ownership, resource limits and circuit statistics. The `API reference`_ and
the `public header`_ document every function.

Applications built on tididi
----------------------------

- `tididi-cnf <https://github.com/Tractables/tididi-cnf>`_ compiles CNF formulas
  into TDDs for model counting, weighted model counting and projected model
  counting, from Rust, Python or C.
- `tddlog <https://github.com/Tractables/tddlog>`_ performs exact probabilistic
  inference for ProbLog, Bayesian networks and probabilistic programs, also
  `in the browser <https://tractables.github.io/tddlog/>`_.

Develop
-------

Using the library needs no Python, but working on it does: a Python script
drives the checks, and the guide is built with Sphinx. To check the generated
header, test a separate installed consumer, and build the HTML guide with
captured program output:

.. code-block:: sh

   python -m pip install -r bindings/c/docs/requirements.txt
   python bindings/c/check.py --docs

Open ``bindings/c/build/docs/index.html``. After changing exported Rust
declarations, regenerate the header with
``python bindings/c/check.py --write-header``.

Citing
------

Please cite `A Canonical Generalization of OBDD <https://arxiv.org/abs/2604.05537>`_,
which introduced TDDs.

Acknowledgements
----------------

This material is based upon work supported by the Defense Advanced Research
Projects Agency (DARPA) under Agreement No. HR0011269E224.

.. _C guide: https://tractables.github.io/tididi/c/
.. _reachability chapter: https://tractables.github.io/tididi/c/reachability.html
.. _configuration chapter: https://tractables.github.io/tididi/c/configurations.html
.. _probability chapter: https://tractables.github.io/tididi/c/probability.html
.. _minimum configuration cost: https://tractables.github.io/tididi/c/minimum_cost.html
.. _API reference: https://tractables.github.io/tididi/c/api.html
.. _public header: https://github.com/Tractables/tididi/blob/main/bindings/c/include/tididi.h
.. _example.h: https://github.com/Tractables/tididi/blob/main/bindings/c/examples/example.h

.. |ci| image:: https://github.com/Tractables/tididi/actions/workflows/rust.yml/badge.svg
   :target: https://github.com/Tractables/tididi/actions/workflows/rust.yml
   :alt: CI
.. |c-docs| image:: https://img.shields.io/badge/docs-C-blue
   :target: https://tractables.github.io/tididi/c/
   :alt: C guide
.. |arxiv| image:: https://img.shields.io/badge/arXiv-2604.05537-b31b1b.svg
   :target: https://arxiv.org/abs/2604.05537
   :alt: arXiv
