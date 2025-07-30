# Makefile for Agentic API Docker operations

.PHONY: build run dev generate clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  make build     - Build the Docker image"
	@echo "  make run       - Run the container (shows help)"
	@echo "  make dev       - Run development container with mounted code"
	@echo "  make generate  - Generate API client (requires URL parameter)"
	@echo "  make shell     - Open shell in container"
	@echo "  make clean     - Remove containers and images"
	@echo "  make logs      - Show container logs"
	@echo ""
	@echo "Example usage:"
	@echo "  make build"
	@echo "  make generate URL=https://petstore.swagger.io/v2/swagger.json"
	@echo "  make dev"

# Build the Docker image
build:
	docker-compose build agentic-api

# Run the container (shows help by default)
run:
	docker-compose run --rm agentic-api

# Run development container with live code mounting
dev:
	docker-compose run --rm agentic-api-dev

# Generate API client from URL
generate:
ifndef URL
	@echo "Error: URL parameter is required"
	@echo "Usage: make generate URL=https://petstore.swagger.io/v2/swagger.json"
	@exit 1
endif
	docker-compose run --rm agentic-api agentic-api generate "$(URL)"

# Open shell in container
shell:
	docker-compose run --rm agentic-api bash

# Show container logs
logs:
	docker-compose logs -f

# Clean up containers and images
clean:
	docker-compose down --rmi all --volumes --remove-orphans
	docker system prune -f

# Example generation commands
example-petstore:
	docker-compose run --rm agentic-api agentic-api generate "https://petstore.swagger.io/v2/swagger.json" --client-name PetStoreClient

example-cve:
	docker-compose run --rm agentic-api agentic-api generate "https://cve.circl.lu/api/swagger.json" --client-name CVEClient
