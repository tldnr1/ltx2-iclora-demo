#!/bin/bash
# =============================================================================
# LTX-2 IC-LoRA Pose Control Demo - Model Download Script
# =============================================================================
#
# Usage: ./download_models.sh /path/to/models
#
# =============================================================================

set -e

# Default models directory
MODELS_DIR="${1:-./models}"

echo "=============================================="
echo "LTX-2 IC-LoRA Pose Demo - Model Downloader"
echo "=============================================="
echo ""
echo "Models will be downloaded to: $MODELS_DIR"
echo ""

# Create directory
mkdir -p "$MODELS_DIR"

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install --quiet huggingface_hub
fi

echo ""
echo "[1/3] Downloading LTX-2 models..."
echo "----------------------------------------------"

# Main checkpoint (distilled version - faster inference)
echo "  -> ltx-2-19b-distilled.safetensors (~38GB)"
huggingface-cli download Lightricks/LTX-2 \
    ltx-2-19b-distilled.safetensors \
    --local-dir "$MODELS_DIR" \
    --local-dir-use-symlinks False

# Spatial upscaler
echo "  -> ltx-2-spatial-upscaler-x2-1.0.safetensors (~2GB)"
huggingface-cli download Lightricks/LTX-2 \
    ltx-2-spatial-upscaler-x2-1.0.safetensors \
    --local-dir "$MODELS_DIR" \
    --local-dir-use-symlinks False

# IC-LoRA pose weight
echo "  -> ltx-2-19b-ic-lora-pose-control.safetensors"
huggingface-cli download Lightricks/LTX-2-19b-IC-LoRA-Pose-Control \
    ltx-2-19b-ic-lora-pose-control.safetensors \
    --local-dir "$MODELS_DIR" \
    --local-dir-use-symlinks False

echo ""
echo "[2/3] Downloading Gemma text encoder..."
echo "----------------------------------------------"
echo "  -> google/gemma-3-12b-it-qat-q4_0-unquantized (~12GB)"

huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized \
    --local-dir "$MODELS_DIR/gemma-3-12b-it-qat-q4_0-unquantized" \
    --local-dir-use-symlinks False

echo ""
echo "[3/3] Verifying downloads..."
echo "----------------------------------------------"

# Check files
check_file() {
    if [ -f "$1" ]; then
        size=$(du -h "$1" | cut -f1)
        echo "  [OK] $2 ($size)"
    else
        echo "  [MISSING] $2 - NOT FOUND"
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        count=$(ls -1 "$1" | wc -l)
        echo "  [OK] $2 ($count files)"
    else
        echo "  [MISSING] $2 - NOT FOUND"
    fi
}

echo ""
check_file "$MODELS_DIR/ltx-2-19b-distilled.safetensors" "Main checkpoint"
check_file "$MODELS_DIR/ltx-2-spatial-upscaler-x2-1.0.safetensors" "Spatial upscaler"
check_dir "$MODELS_DIR/gemma-3-12b-it-qat-q4_0-unquantized" "Gemma text encoder"
check_file "$MODELS_DIR/ltx-2-19b-ic-lora-pose-control.safetensors" "Pose IC-LoRA"

echo ""
echo "=============================================="
echo "Download complete!"
echo ""
echo "To run the demo:"
echo "  export MODELS_PATH=$MODELS_DIR"
echo "  docker compose up --build"
echo "=============================================="
