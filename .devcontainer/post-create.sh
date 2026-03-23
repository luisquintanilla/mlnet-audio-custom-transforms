#!/bin/bash
set -e

echo "=== Installing espeak-ng (required for KittenTTS phonemization) ==="
sudo apt-get update && sudo apt-get install -y espeak-ng

echo "=== Installing huggingface-hub ==="
pip install --quiet huggingface-hub

echo "=== Restoring dependencies ==="
dotnet restore

echo "=== Building solution ==="
dotnet build --no-restore

echo ""
echo "=== Setup complete! ==="
echo ""
echo "espeak-ng is installed (required for KittenTTS phonemization)."
echo "huggingface-cli is installed. Download a model, e.g.:"
echo "  huggingface-cli download onnx-community/ast-finetuned-audioset-10-10-0.4593 --include 'onnx/*' --local-dir models/ast"
echo ""
