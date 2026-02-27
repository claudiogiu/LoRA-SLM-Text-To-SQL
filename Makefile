.PHONY: all setup venv train evaluate

all: setup train evaluate
	@echo "[ALL] End-to-end execution completed."

setup: venv
	@echo "[SETUP] Environment initialization completed. Dependency management and resource provisioning finalized."

venv:
	@if [ ! -f pyproject.toml ]; then \
		echo "[ERROR]  Required file 'pyproject.toml' not found."; \
		exit 1; \
	fi
	@if [ ! -f uv.lock ]; then \
		echo "[ERROR]  Required file 'uv.lock' not found"; \
		exit 1; \
	fi
	uv sync
	@echo "[VENV] Virtual environment '.venv' created and essential packages installed."

ifneq (,$(wildcard .env))
    include .env
    export $(shell sed 's/=.*//' .env)
else
$(error "[ERROR] Required environment file '.env' not found.")
endif

ifndef AZURE_OPENAI_ENDPOINT
$(error "[ERROR] Mandatory variable 'AZURE_OPENAI_ENDPOINT' is not defined in .env.")
endif

ifndef AZURE_OPENAI_API_KEY
$(error "[ERROR] Mandatory variable 'AZURE_OPENAI_API_KEY' is not defined in .env.")
endif

ifndef HUGGINGFACE_HUB_TOKEN
$(error "[ERROR] Mandatory variable 'HUGGINGFACE_HUB_TOKEN' is not defined in .env.")
endif

train:
	uv run python src/train_model.py
	@sleep 20
	@echo "[TRAIN] Training phase successfully completed."


evaluate:
	uv run python src/evaluate_model.py
	@echo "[EVALUATE] Evaluation phase successfully completed."