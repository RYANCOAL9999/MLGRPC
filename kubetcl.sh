#!/bin/bash

echo "Use Kubernetes to deploy the GRPC Services."

echo "There are three deployment models available: batch, single, and multi."

# Prompt the user for input
echo "Choose the model you wish to deploy, enter a key:"
read key

# deploy the mapping for the Port
kubectl apply -f ./deploy/configMapPort.yaml

# deploy the mapping for the GRPC Key
kubectl apply -f ./deploy/configMapGRPCKey.yaml

# deploy the container with the selected key
if [ "$key" == "multi" ]; then
    kubectl apply -f ./deploy/multiContainerDeployment.yaml
elif [ "$key" == "batch" ]; then
    kubectl apply -f ./deploy/batchDeployment.yaml
else
    kubectl apply -f ./deploy/deployment.yaml
fi