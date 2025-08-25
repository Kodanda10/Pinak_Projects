#!/bin/zsh

# This script sets up the entire local memory environment.
# It checks for dependencies, installs them, and runs a verification demo.

# --- Helper Functions ---
print_step() {
  echo "\n=================================================================="
  echo "STEP: $1"
  echo "=================================================================="
}

# --- Main Setup Logic ---

# 1. Check for Homebrew
print_step "Checking for Homebrew"
if ! command -v brew &> /dev/null; then
  echo "Homebrew not found. Please install it to continue."
  echo "See: https://brew.sh/"
  # Uncomment the line below to auto-install Homebrew
  # /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  exit 1
fi
echo "Homebrew is installed."

# 2. Check for and Install Redis
print_step "Checking for Redis"
if ! brew services list | grep -q "redis"; then
  echo "Redis not found. Installing with Homebrew..."
  brew install redis
else
  echo "Redis is already installed."
fi

# 3. Start Redis Service
print_step "Starting Redis Service"
if ! brew services list | grep -q "redis.*started"; then
  echo "Starting Redis service..."
  brew services start redis
  sleep 2 # Give it a moment to start
else
  echo "Redis service is already running."
fi

# 4. Create Python Virtual Environment
print_step "Setting up Python Virtual Environment"
if [ ! -d "venv" ]; then
  echo "Creating Python virtual environment at ./venv/ ..."
  python3 -m venv venv
else
  echo "Virtual environment already exists."
fi

# 5. Activate Virtual Environment and Install Dependencies
print_step "Installing Python Dependencies"
source venv/bin/activate

if pip install -r requirements.txt; then
  echo "Python dependencies installed successfully."
else
  echo "[ERROR] Failed to install Python dependencies. Aborting."
  exit 1
fi

# 6. Run the Demonstration Script
print_step "Running Verification & Seeding Demo"
if python run_demo.py; then
  echo "\n[SUCCESS] The local memory environment is set up and verified!"
  echo "You can now activate the environment with 'source venv/bin/activate' and build your own scripts."
else
  echo "\n[ERROR] The demo script failed. Please check the output above for errors."
  exit 1
fi
