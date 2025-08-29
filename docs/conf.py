import os
import sys
from pathlib import Path
from typing import Any, List

root = Path(__file__).parents[1].resolve()
sys.path.insert(0, str(root))
os.environ["BUILD_DOC"] = "1"
autodoc_mock_imports = [
    "torch",
    "triton",
    "flashinfer._build_meta",
    "cuda",
    "numpy",
    "einops",
    "mpi4py",
]
project = "FlashInfer"
author = "FlashInfer Contributors"
copyright = f"2023-2025, {author}"
package_version = (root / "version.txt").read_text().strip()
version = package_version
release = package_version
extensions = [
    "sphinx_tabs.tabs",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
]
autodoc_default_flags = ["members"]
autosummary_generate = True
source_suffix = [".rst"]
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"
todo_include_todos = False
html_theme = "furo"
templates_path: List[Any] = []
html_static_path = ["_static"]
html_theme_options = {
    "logo_only": True,
    "light_logo": "FlashInfer-white-background.png",
    "dark_logo": "FlashInfer-black-background.png",
}
