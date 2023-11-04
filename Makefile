.PHONY: help
help:
	@echo "Usage:"
	@echo "		virtualenv       Set up virtual environment"
	@echo "		activate         Activate virtual environment"
	@echo "		source           Source environment variable"
	@echo "		install          Install requirements.txt"
	@echo "		protoc           Generate protoc file for GPRC Protocol"
	@echo "		operate          Operate on local"
	@echo "		build            Build Docker image"
	@echo "		run              Run Docker image"
	@echo "		clean            Stop Docker image"

# Define the image name and tag
IMAGE_NAME = ML-grpc-service
IMAGE_TAG = latest
PROTOC_INPUT = ../../AIProto
PROTOCOL_OUT_PY = ./lib/proto/py
PROTOCOL_OUT_PYI = ./lib/proto/pyi
GENERATE_FILE = ../../AIProto/general_expression.proto ../../AIProto/lib/np_library.proto ../../AIProto/lib/enum_expression.proto ../../AIProto/linear_expression.proto ../../AIProto/nearest_neighbors.proto ../../AIProto/polynomial_features.proto ../../AIProto/svm_expression.proto

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
	python -m grpc_tools.protoc -I ${PROTOC_INPUT} --python_out=${PROTOCOL_OUT_PY} --pyi_out=${PROTOCOL_OUT_PYI} --grpc_python_out=${PROTOCOL_OUT_PY} ${GENERATE_FILE}

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