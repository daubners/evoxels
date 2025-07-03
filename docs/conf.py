import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'evoxels'
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
    'jax',
    'sympy',
    'diffrax',
    'optimistix'
]
exclude_patterns = ['_build']

# Automatically document all members using Google style docstrings.
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'special-members': '__init__',
    'inherited-members': False,
}
# Exclude submodule voxelfields as it already is top-level
exclude_patterns += ['**/voxelfields.rst', '**/evoxels.voxelfields.*']

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
autosummary_generate = True
nbsphinx_execute = 'never'
