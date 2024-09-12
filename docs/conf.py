# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys, os

sys.path.insert(0, os.path.abspath('../'))
sys.path.append(os.path.abspath('../ingestion_program'))
sys.path.append(os.path.abspath('../scoring_program'))
sys.path.append(os.path.abspath('../simple_one_syst_model'))
sys.path.append(os.path.abspath('../simple_stat_only_model'))
sys.path.append(os.path.abspath('../pages'))


project = 'FAIR Universe HEP Challenge'
copyright = '2024, FAIR-Universe Collaboration'
author = 'FAIR-Universe Collaboration'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc','myst_parser']
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "html_image",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = "_static/logo.png"
html_favicon = '_static/logo.png'
