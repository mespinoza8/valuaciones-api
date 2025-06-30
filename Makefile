# Variables
IMAGE_NAME := valuaciones-api
CONTAINER_NAME := valuaciones-api-container
PORT := 8080
CONTAINER_PORT   := 8000


.PHONY: all build run stop clean rebuild

all: build

build:
	@echo "Building Docker image: $(IMAGE_NAME)"
	docker build -t $(IMAGE_NAME):latest .

run: build
	@echo "🏃‍♂️ Starting container $(CONTAINER_NAME) on host port $(HOST_PORT)"
	# Si ya había uno levantado, lo paramos y borramos
	-docker stop $(CONTAINER_NAME) 2>/dev/null || true
	-docker rm   $(CONTAINER_NAME) 2>/dev/null || true
	docker run -d \
	  --name $(CONTAINER_NAME) \
	  -p $(HOST_PORT):$(CONTAINER_PORT) \
	  $(IMAGE_NAME):latest

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

