.PHONY: help
help:
	@echo "Usage:"
	@echo "		virtualenv       Set up virtual environment"
	@echo "		activate         Activate virtual environment"
	@echo "		source           Source the environment variable"
	@echo "		install          Install the requirements.txt"
	@echo "		protoc           Generate the protoc file for GPRC Protocol"
	@echo "		operate          Operate on local"
	@echo "		build            Build Docker image"
	@echo "		run              Run Docker image"
	@echo "		clean            Stop Docker image"

# Define the image name and tag
IMAGE_NAME = ML-grpc-service
IMAGE_TAG = latest
PROTOC_INPUT = ./lib/proto
PROTOCOL_OUT_PY = ./lib/proto/py
GENERATE_FILE = ./lib/proto/linear_expression.proto ./lib/proto/nearest_neighbors.proto ./lib/proto/polynomial_features.proto ./lib/proto/svm_expression.proto


.PHONY: virtualenv
virtualenv:
	virtualenv venv -p python3

.PHONY: activate
activate:
	. venv/bin/activate

.PHONY: source
source:
	source .env

.PHONY: install
install:
	pip3 install --trusted-host pypi.python.org -r requirements.txt

.PHONY: protoc
protoc:
	python -m grpc_tools.protoc -I ${PROTOC_INPUT} --python_out=${PROTOCOL_OUT_PY} --grpc_python_out=${PROTOCOL_OUT_PY} ${GENERATE_FILE}

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