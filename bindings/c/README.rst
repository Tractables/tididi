.. scenario: docs/scenarios.md#first-circuit

tididi for C
============

Build Boolean circuits, count their satisfying assignments, evaluate probabilities,
and explore transition systems. This package exposes the same Rust implementation
through a C interface, usable from C and C++.

Start with the `C guide <docs/index.rst>`_ and its executable tutorials.
The `public header <include/tididi.h>`_ documents every function.

From a source checkout, with Rust, CMake and a C/C++ compiler installed::

    cmake -S bindings/c -B build/c -DCMAKE_BUILD_TYPE=Release
    cmake --build build/c --config Release --parallel 8
    ctest --test-dir build/c -C Release --output-on-failure
    cmake --install build/c --config Release --prefix /your/install/prefix

An application can then use ``find_package(tididi CONFIG REQUIRED)`` and link
``tididi::tididi``. Set ``CMAKE_PREFIX_PATH`` to the installation prefix. The
installed library requires no Rust toolchain at runtime. On Windows, keep
``tididi_c.dll`` beside your executable or on ``PATH``.

To check the generated header, test a separate installed consumer, and build the
HTML guide with captured program output::

    python -m pip install -r bindings/c/docs/requirements.txt
    python bindings/c/check.py --docs

Open ``bindings/c/build/docs/index.html``. After changing exported Rust declarations,
regenerate the header with ``python bindings/c/check.py --write-header``.
