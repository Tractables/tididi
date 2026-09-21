"""Keep the installed package, type hints, and documentation entry points aligned."""
import ast
import inspect
from importlib.metadata import distribution
from pathlib import Path

import tididi


def test_stub_covers_public_exports_and_method_signatures():
    stub = Path(tididi.__file__).with_suffix(".pyi")
    tree = ast.parse(stub.read_text())
    declarations = {node.name: node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    assert set(tididi.__all__) == set(declarations)
    for name in tididi.__all__:
        runtime = getattr(tididi, name)
        node = declarations[name]
        if isinstance(node, ast.ClassDef):
            if name.endswith("Error") or name == "Algebra":
                continue
            for member in node.body:
                if not isinstance(member, ast.FunctionDef) or member.name.startswith("__"):
                    continue
                assert hasattr(runtime, member.name), (name, member.name)
                if any(isinstance(d, ast.Name) and d.id == "property" for d in member.decorator_list):
                    continue
                actual = inspect.signature(getattr(runtime, member.name))
                expected = [a.arg for a in member.args.args + member.args.kwonlyargs]
                assert list(actual.parameters) == expected, (name, member.name, actual, expected)
        else:
            actual = inspect.signature(runtime)
            expected = [a.arg for a in node.args.args + node.args.kwonlyargs]
            assert list(actual.parameters) == expected, (name, actual, expected)


def test_installed_metadata_and_license():
    package = distribution("tididi")
    metadata = package.read_text("METADATA")
    assert "Description-Content-Type: text/x-rst" in metadata
    assert "\ntididi for Python\n" in metadata
    assert "License-Expression: Apache-2.0" in metadata
    licenses = [f for f in package.files if f.name == "LICENSE-APACHE"]
    assert len(licenses) == 1
    assert "Apache License" in licenses[0].read_text()
