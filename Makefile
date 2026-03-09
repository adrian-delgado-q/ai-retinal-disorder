SHELL := /bin/bash

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install config-check train download-articles build-index inspect-index api frontend dev test

install:
	$(PIP) install -r requirements.txt

config-check:
	$(PYTHON) scripts/config_check.py

train:
	$(PYTHON) -m training.train_colab $(ARGS)

download-articles:
	$(PYTHON) scripts/download_article_files.py $(ARGS)

build-index:
	$(PYTHON) scripts/index_articles.py build-index $(ARGS)

inspect-index:
	$(PYTHON) scripts/index_articles.py inspect-index $(ARGS)

api:
	@API_HOST="$${API_HOST:-$$( $(PYTHON) -c 'from app_config import load_config; print(load_config().demo.api_host)' )}"; \
	API_PORT="$${API_PORT:-$$( $(PYTHON) -c 'from app_config import load_config; print(load_config().demo.api_port)' )}"; \
	uvicorn demo_app.main:create_app --factory --host "$$API_HOST" --port "$$API_PORT" --reload

frontend:
	@DEMO_API_URL="$${DEMO_API_URL:-$$( $(PYTHON) -c 'from app_config import load_config; print(load_config().effective_frontend_api_url)' )}"; \
	export DEMO_API_URL; \
	reflex run

dev:
	$(PYTHON) scripts/run_dev.py

test:
	pytest
