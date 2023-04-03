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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

import os
import sys
from datetime import datetime
from subprocess import Popen

import where
from packaging import version as version_

# Get current location
_DOC_PATH = os.path.dirname(os.path.abspath(__file__))
_PROJ_PATH = os.path.abspath(os.path.join(_DOC_PATH, '..', '..'))
_LIBS_PATH = os.path.join(_DOC_PATH, '_libs')
_SHIMS_PATH = os.path.join(_DOC_PATH, '_shims')
os.chdir(_PROJ_PATH)

# Set environment, remove the pre-installed package
sys.path.insert(0, _PROJ_PATH)
modnames = [mname for mname in sys.modules if mname.startswith('lzero')]
for modname in modnames:
    del sys.modules[modname]

# Build dependencies if needed
if not os.environ.get("NO_CONTENTS_BUILD"):
    _env = dict(os.environ)
    _env.update(dict(
        PYTHONPATH=':'.join([_PROJ_PATH, _LIBS_PATH]),
        PATH=':'.join([_SHIMS_PATH, os.environ.get('PATH', '')]),
    ))

    if os.path.exists(os.path.join(_PROJ_PATH, 'requirements-build.txt')):
        pip_build_cmd = (where.first('pip'), 'install', '-r', os.path.join(_PROJ_PATH, 'requirements-build.txt'))
        print("Install pip requirements {cmd}...".format(cmd=repr(pip_build_cmd)))
        pip_build = Popen(pip_build_cmd, stdout=sys.stdout, stderr=sys.stderr, env=_env, cwd=_PROJ_PATH)
        if pip_build.wait() != 0:
            raise ChildProcessError("Pip install failed with %d." % (pip_build.returncode,))

        make_build_cmd = (where.first('make'), 'clean', 'build')
        print("Try building extensions {cmd}...".format(cmd=repr(make_build_cmd)))
        make_build = Popen(make_build_cmd, stdout=sys.stdout, stderr=sys.stderr, env=_env, cwd=_PROJ_PATH)
        if make_build.wait() != 0:
            raise ChildProcessError("Extension build failed with %d." % (make_build.returncode,))

    pip_cmd = (where.first('pip'), 'install', '-r', os.path.join(_PROJ_PATH, 'requirements.txt'))
    print("Install pip requirements {cmd}...".format(cmd=repr(pip_cmd)))
    pip = Popen(pip_cmd, stdout=sys.stdout, stderr=sys.stderr, env=_env, cwd=_PROJ_PATH)
    if pip.wait() != 0:
        raise ChildProcessError("Pip install failed with %d." % (pip.returncode,))

    pip_docs_cmd = (where.first('pip'), 'install', '-r', os.path.join(_PROJ_PATH, 'requirements-doc.txt'))
    print("Install pip docs requirements {cmd}...".format(cmd=repr(pip_docs_cmd)))
    pip_docs = Popen(pip_docs_cmd, stdout=sys.stdout, stderr=sys.stderr, env=_env, cwd=_PROJ_PATH)
    if pip_docs.wait() != 0:
        raise ChildProcessError("Pip docs install failed with %d." % (pip.returncode,))

    diagrams_cmd = (where.first('make'), '-f', "diagrams.mk", "build")
    print("Building diagrams {cmd} at {cp}...".format(cmd=repr(diagrams_cmd), cp=repr(_DOC_PATH)))
    diagrams = Popen(diagrams_cmd, stdout=sys.stdout, stderr=sys.stderr, env=_env, cwd=_DOC_PATH)
    if diagrams.wait() != 0:
        raise ChildProcessError("Diagrams failed with %d." % (diagrams.returncode,))

    graphviz_cmd = (where.first('make'), '-f', "graphviz.mk", "build")
    print("Building graphs {cmd} at {cp}...".format(cmd=repr(graphviz_cmd), cp=repr(_DOC_PATH)))
    graphviz = Popen(graphviz_cmd, stdout=sys.stdout, stderr=sys.stderr, env=_env, cwd=_DOC_PATH)
    if graphviz.wait() != 0:
        raise ChildProcessError("Graphviz failed with %d." % (graphviz.returncode,))

    demos_cmd = (where.first('make'), '-f', "demos.mk", "build")
    print("Building demos {cmd} at {cp}...".format(cmd=repr(demos_cmd), cp=repr(_DOC_PATH)))
    demos = Popen(demos_cmd, stdout=sys.stdout, stderr=sys.stderr, env=_env, cwd=_DOC_PATH)
    if demos.wait() != 0:
        raise ChildProcessError("Demos failed with %d." % (demos.returncode,))

    notebook_cmd = (where.first('make'), '-f', "notebook.mk", "build")
    print("Executing notebooks {cmd} at {cp}...".format(cmd=repr(notebook_cmd), cp=repr(_DOC_PATH)))
    demos = Popen(notebook_cmd, stdout=sys.stdout, stderr=sys.stderr, env=_env, cwd=_DOC_PATH)
    if demos.wait() != 0:
        raise ChildProcessError("Notebook failed with %d." % (demos.returncode,))

    print("Build of contents complete.")

from lzero.config.meta import __TITLE__, __AUTHOR__, __VERSION__

project = __TITLE__
copyright = '{year}, {author}'.format(year=datetime.now().year, author=__AUTHOR__)
author = __AUTHOR__

# The short X.Y version
version = version_.parse(__VERSION__).base_version
# The full version, including alpha/beta/rc tags
release = __VERSION__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.todo',
    'sphinx.ext.graphviz',
    'enum_tools.autoenum',
    "sphinx_multiversion",
    'nbsphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
htmlhelp_basename = 'LightZero'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

epub_title = project
epub_exclude_files = ['search.html']

# Whitelist pattern for tags (set to None to ignore all tags)
smv_tag_whitelist = r'^v.*$'  # Include all tags start with 'v'
smv_branch_whitelist = r'^.*$'  # Include all branches
smv_remote_whitelist = r'^.*$'  # Use branches from all remotes
smv_released_pattern = r'^tags/.*$'  # Tags only
smv_outputdir_format = '{ref.name}'  # Use the branch/tag name

if not os.environ.get("ENV_PROD"):
    todo_include_todos = True
    todo_emit_warnings = True
