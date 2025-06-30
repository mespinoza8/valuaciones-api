# Variables
IMAGE_NAME := valuaciones-api
CONTAINER_NAME := valuaciones-api-container
PORT := 8080

.PHONY: all build run stop clean rebuild

all: build

build:
	@echo "Building Docker image: $(IMAGE_NAME)"
	docker build -t $(IMAGE_NAME) .

run:
	@echo "Running Docker container: $(CONTAINER_NAME) on port $(PORT)"
	@echo "Access the application at http://localhost:$(PORT)"
	docker run -d --name $(CONTAINER_NAME) -p $(PORT):$(PORT) \
		-v "$(PWD):/app" \
		$(IMAGE_NAME)
	# Opcional: Ver logs después de iniciar (puedes comentarlo si no lo necesitas inmediatamente)
	# docker logs -f $(CONTAINER_NAME)

stop:
	@echo "Stopping Docker container: $(CONTAINER_NAME)"
	-docker stop $(CONTAINER_NAME) 2>/dev/null || true
	-docker rm $(CONTAINER_NAME) 2>/dev/null || true

clean: stop
	@echo "Cleaning up Docker image: $(IMAGE_NAME)"
	-docker rmi $(IMAGE_NAME) 2>/dev/null || true

rebuild: clean build

status:
	@echo "Docker container status for $(CONTAINER_NAME):"
	docker ps -a --filter "name=$(CONTAINER_NAME)" --format "table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Ports}}"

logs:
	@echo "Showing logs for container: $(CONTAINER_NAME)"
	docker logs -f $(CONTAINER_NAME)

