UV_RUN := uv run

# If NOTEBOOK is set, resolve it to a .py path; otherwise default to all notebooks.
ifdef NOTEBOOK
  # Strip any directory prefix and extension so bare names work (e.g. make plots NOTEBOOK=comparison)
  _NB_BASE := $(basename $(notdir $(NOTEBOOK)))
  _NB_PY   := notebooks/$(_NB_BASE).py
  _NB_IPYNB := notebooks/$(_NB_BASE).ipynb
endif

# Guard: error early if a specific notebook was requested but doesn't exist.
define check_notebook
	@if [ -n "$(NOTEBOOK)" ] && [ ! -f "$(_NB_PY)" ]; then \
		echo "Error: notebook '$(_NB_PY)' not found."; \
		echo "Available notebooks:"; \
		ls notebooks/*.py 2>/dev/null | sed 's|^|  |'; \
		exit 1; \
	fi
endef

.PHONY: help sync plots markdown clean

help: ## Show this help
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | column -t -s '	'

sync: ## Sync .py and .ipynb via jupytext
	$(UV_RUN) jupytext --sync notebooks/*.py

plots: ## Execute notebook(s) and save figures (NOTEBOOK=name to target one)
	$(check_notebook)
ifdef NOTEBOOK
	SAVE_FIGURES=1 $(UV_RUN) jupytext --to notebook --execute $(_NB_PY) --output $(_NB_IPYNB)
else
	@for nb in notebooks/*.py; do \
		echo "==> $$nb"; \
		SAVE_FIGURES=1 $(UV_RUN) jupytext --to notebook --execute "$$nb" --output "$${nb%.py}.ipynb" || exit 1; \
	done
endif

markdown: ## Execute notebook(s) and render as markdown (NOTEBOOK=name to target one)
	$(check_notebook)
ifdef NOTEBOOK
	$(UV_RUN) jupyter nbconvert --to markdown --execute $(_NB_IPYNB) --output-dir notebooks/
else
	$(UV_RUN) jupyter nbconvert --to markdown --execute notebooks/*.ipynb --output-dir notebooks/
endif

clean: ## Remove generated notebooks, markdown, and figures
	rm -f notebooks/*.ipynb notebooks/*.md
	rm -rf notebooks/*_files/
	rm -f figures/*.png figures/*.csv
