.PHONY: install data index query evaluate test
install:
	python -m pip install -e ".[dev]"
data:
	python -m src.data.make_dataset
index: data
	python -m src.models.train_model data/raw/big-buck-bunny.mp4
query:
	python -m src.models.predict_model "A bird singing on a branch"
evaluate:
	python -m src.models.evaluate
test:
	python -m pytest -q
