# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS         ?=
SPHINXBUILD        ?= $(shell which sphinx-build)
SPHINXMULTIVERSION ?= $(shell which sphinx-multiversion)
SOURCEDIR          ?= source
BUILDDIR           ?= build

# Minimal makefile for Sphinx documentation
DIAGRAMS_MK := ${SOURCEDIR}/diagrams.mk
DIAGRAMS    := $(MAKE) -f "${DIAGRAMS_MK}" SOURCE=${SOURCEDIR}
GRAPHVIZ_MK := ${SOURCEDIR}/graphviz.mk
GRAPHVIZ    := $(MAKE) -f "${GRAPHVIZ_MK}" SOURCE=${SOURCEDIR}
DEMOS_MK    := ${SOURCEDIR}/demos.mk
DEMOS       := $(MAKE) -f "${DEMOS_MK}" SOURCE=${SOURCEDIR}
NOTEBOOK_MK := ${SOURCEDIR}/notebook.mk
NOTEBOOK    := $(MAKE) -f "${NOTEBOOK_MK}" SOURCE=${SOURCEDIR}

_CURRENT_PATH := ${PATH}
_PROJ_DIR     := $(shell readlink -f ${CURDIR}/..)
_LIBS_DIR     := $(shell readlink -f ${SOURCEDIR}/_libs)
_SHIMS_DIR    := $(shell readlink -f ${SOURCEDIR}/_shims)

.EXPORT_ALL_VARIABLES:

PYTHONPATH = ${_PROJ_DIR}:${_LIBS_DIR}
PATH       = ${_SHIMS_DIR}:${_CURRENT_PATH}
NO_CONTENTS_BUILD = true

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# Put it first so that "make" without argument is like "make help".
.PHONY: help contents build html prod clean sourcedir builddir Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

contents:
	@$(DIAGRAMS) build
	@$(GRAPHVIZ) build
	@$(DEMOS) build
	@$(NOTEBOOK) build
build: html
html: contents
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@touch "$(BUILDDIR)/html/.nojekyll"
prod:
	@NO_CONTENTS_BUILD='' $(SPHINXMULTIVERSION) "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@cp main_page.html "$(BUILDDIR)/html/index.html"
	@touch "$(BUILDDIR)/html/.nojekyll"

clean:
	@$(DIAGRAMS) clean
	@$(GRAPHVIZ) clean
	@$(DEMOS) clean
	@$(NOTEBOOK) clean
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

sourcedir:
	@echo $(shell readlink -f ${SOURCEDIR})
builddir:
	@echo $(shell readlink -f ${BUILDDIR}/html)