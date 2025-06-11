import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "voxelsss"
author = "Simon Daubner"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "myst_parser",
    "nbsphinx",
]

nb_execution_mode = "off"
html_theme = "sphinx_rtd_theme"
autodoc_mock_imports = ["matplotlib", "pyvista", "psutil", "ipython", "numpy", "torch"]
exclude_patterns = ["_build"]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
autosummary_generate = True
nbsphinx_execute = "never"
