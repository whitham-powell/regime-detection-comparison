.PHONY: sync plots clean

sync:
	jupytext --sync notebooks/*.py

plots:
	SAVE_FIGURES=1 jupytext --to notebook --execute notebooks/comparison.py --output notebooks/comparison.ipynb

clean:
	rm -f notebooks/*.ipynb
	rm -f figures/*.png figures/*.csv
