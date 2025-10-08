#!/bin/bash
# Jetbot deployment environment setup script
# Run this on your Jetbot after transferring the files

set -e

echo "================================================"
echo "Jetbot Policy Deployment Environment Setup"
echo "================================================"

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "Warning: This doesn't appear to be a Jetson device"
    echo "Continuing anyway..."
fi

# Update system
echo ""
echo "Updating system packages..."
sudo apt-get update

# Install system dependencies
echo ""
echo "Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    i2c-tools \
    libi2c-dev

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip3 install --upgrade pip

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Check PyTorch installation
echo ""
echo "Checking PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check if Jetbot library is working
echo ""
echo "Checking Jetbot library..."
python3 -c "from jetbot import Robot; print('Jetbot library OK')" || {
    echo "Warning: Jetbot library not working. Installing jetbot..."
    pip3 install jetbot
}

# Make run script executable
echo ""
echo "Making run_policy.py executable..."
chmod +x run_policy.py

# Create logs directory
echo ""
echo "Creating logs directory..."
mkdir -p logs

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Edit run_policy.py to match your observation/action space"
echo "2. Test with: python3 run_policy.py --max-speed 0.2 --duration 10"
echo "3. Check DEPLOYMENT.md for detailed instructions"
echo ""
echo "Quick start: python3 run_policy.py --help"
echo ""
