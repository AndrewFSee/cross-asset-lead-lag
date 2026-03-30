.PHONY: install test lint data dashboard clean test-cov format

install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=. --cov-report=html

lint:
	ruff check .
	black --check .

format:
	ruff check --fix .
	black .

data:
	python -c "from data.ingestion import fetch_all_data; fetch_all_data()"

pipeline:
	python run_pipeline.py

pipeline-cached:
	python run_pipeline.py --skip-fetch

pipeline-backtest:
	python run_pipeline.py --skip-fetch --backtest

dashboard:
	streamlit run dashboard/app.py

clean:
	rm -rf data/cache/*.parquet
	rm -rf .pytest_cache __pycache__ .mypy_cache
	rm -rf htmlcov .coverage
