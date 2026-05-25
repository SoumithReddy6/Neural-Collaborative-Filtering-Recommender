.PHONY: install test smoke ab manifest api

install:
	pip install -r requirements.txt

test:
	pytest

smoke:
	python3 scripts/run_smoke_demo.py

ab:
	python3 scripts/run_ab_test.py

manifest:
	python3 scripts/generate_mlflow_runs.py

api:
	uvicorn ncf_recommender.api:app --reload

