.PHONY: all run backend frontend install clean help

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