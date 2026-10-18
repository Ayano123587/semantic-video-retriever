.PHONY: install data research-metadata research-data index query research-help test validate-notebooks

INDEX ?= data/processed/my-index
INTERVAL ?= 2
TOP_K ?= 5

install:
	python -m pip install -e ".[dev]" nbformat==5.11.1

data: research-data

research-metadata:
	python -m src.data.didemo --noncommercial-research

research-data:
	python -m src.data.didemo --noncommercial-research --download

index:
	$(if $(strip $(VIDEO)),,$(error Set VIDEO to a real video you own or are licensed to process))
	python -m src.models.train_model "$(VIDEO)" --output "$(INDEX)" --interval "$(INTERVAL)" --device cpu

query:
	$(if $(strip $(QUERY)),,$(error Set QUERY to the description to search for))
	python -m src.models.predict_model "$(QUERY)" --index "$(INDEX)" --top-k "$(TOP_K)" --suppress-seconds 4

research-help:
	python -m src.models.research_search --help

test:
	python -m pytest -q

validate-notebooks:
	python scripts/validate_notebooks.py
