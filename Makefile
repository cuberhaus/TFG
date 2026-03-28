.PHONY: all run backend frontend install clean

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

# Stop running processes on the specific ports
clean:
	@echo "Stopping processes on ports 8082 (backend) and 5173 (frontend)..."
	-fuser -k 8082/tcp
	-fuser -k 5173/tcp
	@echo "Cleaned up."