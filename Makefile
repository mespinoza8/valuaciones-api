# Variables
IMAGE_NAME := valuaciones-api
CONTAINER_NAME := valuaciones-api-container
PORT := 8080


.PHONY: all build run stop clean rebuild

all: build

build:
	@echo "Building Docker image: $(IMAGE_NAME)"
	docker build -t $(IMAGE_NAME):latest .

run:
	@echo "Running $(CONTAINER_NAME) on http://localhost:$(PORT)"
	# elimina previo contenedor si existe
	-docker rm -f $(CONTAINER_NAME)
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(PORT):$(PORT) \
		$(IMAGE_NAME):latest

stop:
	docker rm -f $(CONTAINER_NAME) || true

clean: stop
	docker rmi $(IMAGE_NAME):latest || true

status:
	docker ps -a --filter "name=$(CONTAINER_NAME)"

logs:
	docker logs -f $(CONTAINER_NAME)