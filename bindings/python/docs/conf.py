"""Build Python API help and execute the narrative examples."""
from pathlib import Path
import shutil

import tididi

project = "tididi for Python"
release = tididi.__version__
extensions = ["sphinx.ext.autodoc", "sphinx.ext.doctest", "sphinx_gallery.gen_gallery"]
html_theme = "furo"
html_title = "tididi for Python"
html_static_path = ["_static"]
html_logo = "../../../docs/logo.svg"
html_theme_options = {"light_css_variables": {"color-brand-primary": "#2774ae", "color-brand-content": "#2774ae"}}
exclude_patterns = ["_build", "sg_execution_times.rst", "tutorials/sg_execution_times.rst", "tutorials/index.rst"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "tutorials",
    "filename_pattern": r"/\d\d_.*\.py",
    "within_subsection_order": "FileNameSortKey",
    "abort_on_example_error": True,
    "run_stale_examples": True,
    "download_all_examples": False,
    "show_signature": False,
    "write_computation_times": False,
    "min_reported_time": 10**9,
    "image_scrapers": (),
    "reset_modules": (),
    "capture_repr": (),
    "notebook_images": True,
    "doc_module": ("tididi",),
    "backreferences_dir": None,
}

# The Rust and Python tutorials share these figures.
source = Path(__file__).resolve().parents[3] / "docs"
static = Path(__file__).parent / "_static"
static.mkdir(exist_ok=True)
for name in ["reachability.svg", "vtree-grouping.svg", "tdd-basics.svg"]:
    shutil.copyfile(source / name, static / name)
