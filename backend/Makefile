.PHONY: help install up down ingest api demo test fmt

help:
	@echo "install   Install Python dependencies"
	@echo "up        Start Neo4j (docker compose)"
	@echo "down      Stop Neo4j"
	@echo "ingest    Run the Reddit -> graph + vector ingestion pipeline"
	@echo "api       Launch the FastAPI server"
	@echo "demo      Run the 4-query demo"
	@echo "test      Run unit tests (pure-logic, no external services)"

install:
	pip install -r requirements.txt

up:
	docker compose up -d

down:
	docker compose down

ingest:
	python -m ingestion.pipeline

api:
	uvicorn app.main:app --host $${API_HOST:-0.0.0.0} --port $${API_PORT:-8000} --reload

demo:
	python demo.py

test:
	pytest -q
