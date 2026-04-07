.PHONY: help build up down dev logs clean init-data ingest-docs list-docs clear-docs

COMPOSE_DEV = docker compose -f docker-compose.yml -f docker-compose.dev.yml

help:
	@echo "TaraChat - RAG Chatbot"
	@echo ""
	@echo "Available commands:"
	@echo "  make build       - Build Docker images (production)"
	@echo "  make up          - Start the application (production)"
	@echo "  make dev         - Start the application (development with hot-reload)"
	@echo "  make down        - Stop the application"
	@echo "  make logs        - View logs"
	@echo "  make clean       - Clean up containers and volumes"
	@echo "  make init-data   - Initialize vector store with sample data"
	@echo ""
	@echo "Document management:"
	@echo "  make ingest-docs - Ingest documents from data/documents/"
	@echo "  make list-docs   - List all documents in vector store"
	@echo "  make clear-docs  - Clear all documents from vector store"
	@echo ""
	@echo "Local development (requires poetry):"
	@echo "  cd backend && poetry run python scripts/ingest_documents.py --help"

build:
	docker compose build

up:
	docker compose up -d
	@echo ""
	@echo "Application started! (production)"
	@echo "Frontend: http://localhost"
	@echo "Backend: http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"

dev:
	$(COMPOSE_DEV) up -d
	@echo ""
	@echo "Application started! (development)"
	@echo "Frontend: http://localhost:5173"
	@echo "Backend: http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"

down:
	docker compose down

logs:
	docker compose logs -f

clean:
	docker compose down -v
	rm -rf backend/vector_store

init-data:
	docker compose exec backend python scripts/init_data.py

ingest-docs:
	@echo "Ingesting documents from backend/data/documents/"
	docker compose exec backend python scripts/ingest_documents.py add --dir data/documents/

list-docs:
	docker compose exec backend python scripts/ingest_documents.py list

clear-docs:
	docker compose exec backend python scripts/ingest_documents.py clear
