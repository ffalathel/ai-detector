#!/bin/bash

# AI Detector Training Mode Script
# This script sets up the environment for training and prevents API conflicts

echo "🚀 Starting AI Detector Training Mode"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "app/main.py" ]; then
    echo "❌ Error: Please run this script from the backend directory"
    exit 1
fi

# Set training mode environment variable
export TRAINING_MODE=true
echo "✅ TRAINING_MODE environment variable set to 'true'"

# Check if FastAPI app is running
if pgrep -f "uvicorn.*main:app" > /dev/null; then
    echo "⚠️  Warning: FastAPI app is currently running"
    echo "   It will automatically switch to training mode"
    echo "   (models will not be loaded, endpoints will return 503)"
fi

# Set GPU configuration if available
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected"
    export CUDA_VISIBLE_DEVICES=0
    echo "✅ CUDA_VISIBLE_DEVICES set to 0"
else
    echo "💻 No NVIDIA GPU detected, using CPU"
fi

# Set PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_CUDNN_V8_API_ENABLED=1
echo "⚙️  PyTorch memory management configured"

echo ""
echo "🎯 Now you can run training:"
echo "   python train/train_model.py"
echo ""
echo "📝 Or use the wrapper script:"
echo "   python train_with_api.py"
echo ""
echo "🔄 To exit training mode, run:"
echo "   export TRAINING_MODE=false"
echo ""

# Keep the environment active
echo "🔒 Training mode environment is now active"
echo "   Press Ctrl+C to exit"
echo ""

# Wait for user input or signal
trap 'echo ""; echo "🔄 Exiting training mode..."; export TRAINING_MODE=false; echo "✅ TRAINING_MODE reset to false"; exit 0' INT TERM

while true; do
    sleep 1
done
