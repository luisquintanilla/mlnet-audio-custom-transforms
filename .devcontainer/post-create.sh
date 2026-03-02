#!/bin/bash
set -e

echo "=== Restoring dependencies ==="
dotnet restore

echo "=== Building solution ==="
dotnet build --no-restore

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To download models, install huggingface-cli:"
echo "  pip install huggingface-hub"
echo ""
echo "Then download a model, e.g.:"
echo "  huggingface-cli download onnx-community/ast-finetuned-audioset-10-10-0.4593 --include 'onnx/*' --local-dir models/ast"
echo ""
