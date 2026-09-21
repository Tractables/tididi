.. scenario: docs/scenarios.md#first-circuit

tididi for Python
=================

Build Boolean circuits, combine their rules, and query their solutions without
listing every assignment. TDDs support exact counting, weighted evaluation,
quantification, and variable renaming.

.. code-block:: python

   from tididi import Vtree, literal

   vtree = Vtree.balanced(3)
   x = literal(vtree, 1)
   y = literal(vtree, 2)
   z = literal(vtree, 3)
   f = (x & y) | z
   print(f.model_count())

Output:

.. code-block:: text

   5

Boolean operations consume their circuit operands. Use ``a.copy() & b.copy()``
to retain both inputs. Queries such as ``model_count()`` leave a circuit usable;
reusing a consumed circuit raises ``ConsumedCircuitError``.

The `Python guide <https://tractables.github.io/tididi/python/>`_ introduces
configuration rules, probabilities, reachability, tables, and advanced evaluation.
Each walkthrough includes its executed output and downloadable Python and notebook versions.

Build from source
-----------------

From this directory, with Rust and Python installed:

.. code-block:: sh

   python -m venv .venv
   . .venv/bin/activate
   python -m pip install 'maturin>=1.9.4,<2'
   maturin develop --release

On Windows activate with ``.venv\Scripts\activate`` instead. For development checks:

.. code-block:: sh

   python -m pip install '.[test,docs]'
   python -m pytest
   python -m sphinx -W --keep-going -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` directly in a browser. Sphinx executes the
narrative examples while building the pages and captures their output.
