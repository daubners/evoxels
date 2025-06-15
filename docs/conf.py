import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'voxelsss'
author = 'Simon Daubner'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'myst_parser',
    'nbsphinx',
]

myst_enable_extensions = [
    'amsmath',
    'dollarmath',
]

nb_execution_mode = 'off'
html_theme = 'sphinx_rtd_theme'
autodoc_mock_imports = [
    'matplotlib',
    'pyvista',
    'psutil',
    'IPython',
    'numpy',
    'torch',
    'sympy'
]
exclude_patterns = ['_build']

# Copy images referenced in the README so they appear on ReadTheDocs.
html_extra_path = ['../voxelsss.png', '../voxelsss-graphical.png']

# Automatically document all members using Google style docstrings.
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'special-members': '__init__',
    'inherited-members': False,
}
# Exclude submodule voxelfields as it already is top-level
exclude_patterns += ['**/voxelfields.rst', '**/voxelsss.voxelfields.*']

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
autosummary_generate = True
nbsphinx_execute = 'never'
