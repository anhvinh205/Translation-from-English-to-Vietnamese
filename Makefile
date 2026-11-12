.PHONY: install dev-install test test-unit test-integration lint format check api ui train-sft train-dpo eval up down clean

install:
	pip install -r requirements.txt -r requirements-ui.txt

dev-install:
	pip install -r requirements-dev.txt

test:
	MOCK_LLM=1 pytest tests/ -v --tb=short

test-unit:
	MOCK_LLM=1 pytest tests/unit/ -v --tb=short

test-integration:
	MOCK_LLM=1 pytest tests/integration/ -v --tb=short

lint:
	ruff check src/ tests/ ui/

format:
	ruff format src/ tests/ ui/

check: lint test

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run ui/streamlit_app.py

train-sft:
	python -m src.training.sft_qlora

train-dpo:
	python -m src.training.dpo_align

eval:
	python -m src.evaluation.benchmark

up:
	docker compose up --build -d

down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache .coverage htmlcov
