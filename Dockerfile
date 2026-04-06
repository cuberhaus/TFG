# Stage 1: Build React frontend
FROM node:22-slim AS frontend

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
ENV VITE_API_URL=
RUN npm run build

# Stage 2: Python runtime with FastAPI + PyTorch
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY backend/ ./backend/
COPY code/src/ ./code/src/
COPY --from=frontend /app/dist/ ./frontend/dist/

RUN mkdir -p code/out code/data

ENV PORT=8082
EXPOSE 8082

WORKDIR /app/backend

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8082"]
