# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import datetime
import sys

sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Marian NMT'
copyright = '2021, Marian NMT Team'
author = 'Marian NMT Team'

# The full version, including alpha/beta/rc tags
# TODO: add GitHub commit hash to the version
version_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VERSION')
with open(os.path.abspath(version_file)) as f:
    version = f.read().strip()
release = version + ' ' + str(datetime.date.today())


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'breathe',
    'exhale',
    'myst_parser',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    'build',
    'doxygen',
    'venv',
    'README.md',
]

# The file extensions of source files. Sphinx considers the files with
# this suffix as sources. By default, Sphinx only supports 'restructuredtext'
# file type. You can add a new file type using source parser extensions.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
htmlhelp_basename = 'marian'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['css/custom.css']

# The base URL which points to the root of the HTML documentation
html_baseurl = 'http://marian-nmt.github.io/docs/api'


# -- Extension configuration -------------------------------------------------

breathe_projects = { 'marian': './doxygen/xml' }
breathe_default_project = 'marian'

doxygen_config = """
INPUT                = ../src
EXCLUDE             += ../src/3rd_party
EXCLUDE             += ../src/tests
EXCLUDE_PATTERNS     = *.md *.txt
FILE_PATTERNS       += *.cu
EXTENSION_MAPPING   += cu=C++ inc=C++
ENABLE_PREPROCESSING = YES
JAVADOC_AUTOBRIEF    = YES
WARN_IF_UNDOCUMENTED = NO
USE_MATHJAX          = YES
"""

exhale_args = {
    'containmentFolder'     : './api',
    'rootFileName'          : 'library_index.rst',
    'rootFileTitle'         : 'Library API',
    'doxygenStripFromPath'  : '..',
    'createTreeView'        : True,
    'exhaleExecutesDoxygen' : True,
    # 'verboseBuild'          : True, # set True for debugging
    'exhaleDoxygenStdin'    : doxygen_config.strip(),
}

primary_domain = 'cpp'
highlight_language = 'cpp'

# A trick to include markdown files from outside the source directory using
# 'mdinclude'. Warning: all other markdown files not included via 'mdinclude'
# will be rendered using recommonmark as recommended by Sphinx
from m2r import MdInclude

def setup(app):
    # from m2r to make `mdinclude` work
    app.add_config_value('no_underscore_emphasis', False, 'env')
    app.add_config_value('m2r_parse_relative_links', False, 'env')
    app.add_config_value('m2r_anonymous_references', False, 'env')
    app.add_config_value('m2r_disable_inline_math', False, 'env')
    app.add_directive('mdinclude', MdInclude)
