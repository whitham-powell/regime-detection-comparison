UV_RUN := uv run

.PHONY: help sync plots markdown clean

help: ## Show this help
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | column -t -s '	'

sync: ## Sync .py and .ipynb via jupytext
	$(UV_RUN) jupytext --sync notebooks/*.py

plots: ## Execute notebooks and save figures
	SAVE_FIGURES=1 $(UV_RUN) jupytext --to notebook --execute notebooks/comparison.py --output notebooks/comparison.ipynb
	SAVE_FIGURES=1 $(UV_RUN) jupytext --to notebook --execute notebooks/comparison_synthetic.py --output notebooks/comparison_synthetic.ipynb

markdown: ## Execute notebooks and render as markdown with figures
	$(UV_RUN) jupyter nbconvert --to markdown --execute notebooks/*.ipynb --output-dir notebooks/

clean: ## Remove generated notebooks, markdown, and figures
	rm -f notebooks/*.ipynb notebooks/*.md
	rm -rf notebooks/*_files/
	rm -f figures/*.png figures/*.csv
