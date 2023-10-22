.PHONY: help
help:
	@echo "Usage:"
	@echo "    virtualenv       Set up virtual environment"
	@echo "    activate         Activate virtual environment"
	@echo "    source           Source the environment variable"
	@echo "    operate          Operate on local"
	@echo "    build            Build Docker image"
	@echo "    run              Run Docker image"
	@echo "    clean            Stop Docker image"

# Define the image name and tag
IMAGE_NAME = ML-grpc-service
IMAGE_TAG = latest

.PHONY: virtualenv
virtualenv:
	virtualenv venv -p python3

.PHONY: activate
activate:
	. venv/bin/activate

.PHONY: source
source:
	source .env

.PHONY: operate
operate:
	python3 main.py

.PHONY: build
build:
	docker-compose build

.PHONY: run
run:
	docker-compose up

.PHONY: stop
stop:
	docker-compose down

.PHONY: clean
clean:
	docker-compose down --volumes