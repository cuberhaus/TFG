.PHONY: all run backend frontend install clean help mlops-up mlops-down mlops-logs skills-list skills-update skills-restore

# ── MLOps overlay (MLflow + Evidently) ─────────────────────────────────
# Auto-source the .env.mlops file when present so the host ports /
# image pins / credentials defined there flow into docker-compose.
#
# IMPORTANT: the overlay runs under its OWN Compose project (`tfg-mlops`)
# rather than sharing the default `tfg` project with docker-compose.yml.
# Reason: PersonalPortfolio's dev-all-demos.sh calls
#   docker compose -f docker-compose.yml down --remove-orphans
# on TFG before bringing the app up. If MLOps containers shared the `tfg`
# project, that one-liner would treat them as orphans and tear them down
# right after `make all` brought them up. Keeping the MLOps stack in a
# separate project name makes the two compose lifecycles fully independent.
MLOPS_ENV_FILE := observability/.env.mlops
MLOPS_COMPOSE := -p tfg-mlops -f docker-compose.mlops.yml
ifneq ($(wildcard $(MLOPS_ENV_FILE)),)
  MLOPS_COMPOSE_ENV := --env-file $(MLOPS_ENV_FILE)
else
  MLOPS_COMPOSE_ENV :=
endif


# Default target
all: run

# Run both backend and frontend concurrently
run:
	@echo "Starting backend and frontend..."
	@make -j 2 backend frontend

# Run the FastAPI backend
backend:
	@echo "Starting FastAPI backend..."
	cd backend && ./run.sh

# Run the React/Vite frontend
frontend:
	@echo "Starting React frontend..."
	cd frontend && npm run dev

# Install dependencies for both ends
install:
	@echo "Installing backend dependencies..."
	cd backend && pip install -r requirements.txt
	@echo "Installing ML/code dependencies..."
	cd code && pip install -r requirements.txt
	@echo "Installing frontend dependencies..."
	cd frontend && npm install

# Docker
.PHONY: docker-build docker-up docker-down docker-logs

docker-build:
	docker compose build

docker-up:
	docker compose up -d
	@echo ""
	@echo "  TFG Polyp Detection is running at:"
	@echo "    ➜  http://localhost:8082"
	@echo ""

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

# Stop running processes on the specific ports
clean:
	@echo "Stopping processes on ports 8082 (backend) and 5173 (frontend)..."
	-fuser -k 8082/tcp
	-fuser -k 5173/tcp
	@echo "Cleaned up."

mlops-up:
	docker compose $(MLOPS_COMPOSE_ENV) $(MLOPS_COMPOSE) up -d \
		mlflow-postgres mlflow-minio mlflow-minio-init mlflow-server \
		prediction-log-postgres evidently-ui evidently-scheduler
	@echo ""
	@echo "  MLOps observability stack is running:"
	@echo "    MLflow tracking UI  →  http://localhost:15000"
	@echo "    Evidently dashboard →  http://localhost:15001"
	@echo "    prediction-log DB   →  postgresql://mlops:mlops@localhost:15432/prediction_log"
	@echo ""
	@echo "  Enable prediction logging on the host-running FastAPI app:"
	@echo "    export MLOPS_PREDICTION_LOG_DSN=postgresql://mlops:mlops@localhost:15432/prediction_log"
	@echo "  Or for a Docker app, use host.docker.internal:15432 instead of localhost."
	@echo ""

mlops-down:
	docker compose $(MLOPS_COMPOSE_ENV) $(MLOPS_COMPOSE) down

mlops-logs:
	docker compose $(MLOPS_COMPOSE_ENV) $(MLOPS_COMPOSE) logs -f \
		mlflow-server evidently-ui evidently-scheduler

skills-list:
	@npx skills list -p

skills-update:
	@npx skills update -p -y
	@echo ""
	@echo "Changed skill files:"
	@git diff --name-only -- .agents/skills skills-lock.json || true

skills-restore:
	@npx skills experimental_install

help:
	@echo "Usage:"
	@echo "  make run            Start backend (:8082) and frontend (:5173) concurrently"
	@echo "  make backend        Start FastAPI backend only"
	@echo "  make frontend       Start React/Vite frontend only"
	@echo "  make install        Install all dependencies (backend + ML + frontend)"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-up      Start Docker container on :8082"
	@echo "  make docker-down    Stop Docker container"
	@echo "  make docker-logs    Tail Docker container logs"
	@echo "  make clean          Stop running dev processes"
	@echo ""
	@echo "  MLOps observability (additive, opt-in):"
	@echo "  make mlops-up       Start MLflow + Evidently (:15000, :15001, :15432)"
	@echo "  make mlops-down     Stop MLOps stack (preserves volumes)"
	@echo "  make mlops-logs     Tail MLflow + Evidently logs"
	@echo ""
	@echo "  Agent skills:"
	@echo "  make skills-list    List installed agent skills"
	@echo "  make skills-update  Update skills and show diff"
	@echo "  make skills-restore Restore pinned skills from skills-lock.json"