from pathlib import Path
import shutil

project = "tididi for C"
author = "tididi contributors"
extensions = []
html_theme = "furo"
html_title = "tididi for C"
html_logo = "../../../docs/logo.svg"
html_theme_options = {"light_css_variables": {"color-brand-primary": "#2774ae", "color-brand-content": "#2774ae"}}
html_static_path = ["_static"]
exclude_patterns = ["_generated"]
root = Path(__file__).resolve().parents[3]
static = Path(__file__).parent / "_static"
static.mkdir(exist_ok=True)
for name in ("reachability.svg", "vtree-grouping.svg"):
    shutil.copyfile(root / "docs" / name, static / name)
