# Minimal Makefile for the standalone performance report.
#
# Usage (from this directory):
#   make html       -- build HTML output in _build/html
#   make clean      -- remove the _build directory
#   make livehtml   -- auto-rebuild on file changes (requires sphinx-autobuild)

SPHINXOPTS  ?=
SPHINXBUILD ?= sphinx-build
SOURCEDIR   = .
BUILDDIR    = _build

.PHONY: help html clean livehtml

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo
	@echo "Build finished.  Open $(BUILDDIR)/html/index.html to view."

clean:
	rm -rf $(BUILDDIR)

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
