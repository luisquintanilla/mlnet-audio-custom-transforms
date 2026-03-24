#!/bin/bash
set -e

echo "=== Installing espeak-ng (required for KittenTTS phonemization) ==="
sudo apt-get update && sudo apt-get install -y espeak-ng

echo "=== Installing huggingface-hub ==="
pip install --quiet huggingface-hub

echo "=== Downloading KittenTTS model (if not already present) ==="
if [ ! -d "models/kittentts" ]; then
  git clone https://huggingface.co/KittenML/kitten-tts-mini-0.8 models/kittentts
fi
# Ensure LFS files are fetched (not just pointers)
cd models/kittentts && git lfs pull && cd ../..

echo "=== Restoring dependencies ==="
dotnet restore

echo "=== Building solution ==="
dotnet build --no-restore

echo ""
echo "=== Setup complete! ==="
echo ""
echo "espeak-ng is installed (required for KittenTTS phonemization)."
echo "KittenTTS model downloaded to models/kittentts/."
echo "huggingface-cli is installed. Download additional models, e.g.:"
echo "  huggingface-cli download onnx-community/ast-finetuned-audioset-10-10-0.4593 --include 'onnx/*' --local-dir models/ast"
echo ""
