#!/bin/bash
# ===========================================
# GCP GPU VM Setup Script (No Docker)
# Run this script ON the GCP VM after creation
# ===========================================

set -e

echo "=========================================="
echo "Installing NVIDIA Drivers & CUDA..."
echo "=========================================="

# Install NVIDIA driver
sudo apt-get update
sudo apt-get install -y linux-headers-$(uname -r)
sudo apt-get install -y nvidia-driver-535

# Verify GPU
nvidia-smi

echo "=========================================="
echo "Installing Python & Dependencies..."
echo "=========================================="

# Install Python 3.10 and pip
sudo apt-get install -y python3.10 python3.10-venv python3-pip

# Install system dependencies for OpenCV
sudo apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

# Create app directory
mkdir -p ~/yolo-backend
cd ~/yolo-backend

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install flask flask-cors ultralytics opencv-python-headless numpy gunicorn Pillow

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload your app.py and best.pt model to ~/yolo-backend/"
echo "2. Activate venv: source ~/yolo-backend/venv/bin/activate"
echo "3. Run: python app.py"
echo "   Or with gunicorn: gunicorn --bind 0.0.0.0:8080 -w 1 -t 300 app:app"
