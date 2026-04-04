.PHONY: help install dev backend frontend build start clean docker-up docker-down lint test

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install       Install all dependencies"
	@echo "  start         Build frontend and start backend (serves both)"
	@echo "  dev           Run backend and frontend in dev mode"
	@echo "  backend       Run backend only"
	@echo "  frontend      Run frontend dev server"
	@echo "  build         Build frontend for production"
	@echo "  docker-up     Start all services with docker-compose"
	@echo "  docker-down   Stop all services"
	@echo "  lint          Run linters"
	@echo "  test          Run tests"
	@echo "  clean         Remove build artifacts"

install:
	cd backend && pip install -r requirements.txt
	cd frontend && npm install

start: build
	cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

backend:
	cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

frontend:
	cd frontend && npm run dev

dev:
	@echo "Starting backend and frontend..."
	@make -j2 backend frontend

build:
	cd frontend && npm run build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

lint:
	cd frontend && npm run lint

test:
	cd backend && python -m pytest test/ -v

clean:
	rm -rf frontend/dist
	rm -rf backend/__pycache__
	find backend -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find backend -type f -name "*.pyc" -delete 2>/dev/null || true
