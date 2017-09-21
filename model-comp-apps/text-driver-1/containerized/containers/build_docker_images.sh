#!/usr/bin/env bash

set -e
set -u
set -o pipefail

# Build RPC base images for python/anaconda and deep learning
# models
time docker build -t model-comp/theano-rpc -f TheanoRpcDockerfile ./

# Build model-specific images
time docker build -t model-comp/theano-lstm -f TheanoSentimentDockerfile ./
