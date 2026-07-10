#!/usr/bin/env sh
# buf generate rewrites inference_pb2_grpc.py with a top-level import.
# Run this after every `buf generate` when Python stubs change.
set -eu

GRPC_FILE="python_backend/gen/inference_pb2_grpc.py"

if [ ! -f "$GRPC_FILE" ]; then
  echo "missing $GRPC_FILE" >&2
  exit 1
fi

sed -i 's/^import inference_pb2 as inference__pb2$/from . import inference_pb2 as inference__pb2/' "$GRPC_FILE"
echo "patched $GRPC_FILE"
