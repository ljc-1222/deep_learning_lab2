# Do not run as an executable. You must run this using: source setup.sh

ENV_DIR="dl_lab2_venv"

# Safely locate requirements.txt whether sourced in Bash or Zsh
if [ -n "${BASH_SOURCE[0]:-}" ]; then
  REQ_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/requirements.txt"
else
  REQ_FILE="$(pwd)/requirements.txt"
fi

# Check if python3 is installed
if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is not installed or not in PATH."
  return 1
fi

# Check if requirements.txt exists
if [[ ! -f "$REQ_FILE" ]]; then
  echo "Error: requirements.txt not found at $REQ_FILE"
  return 1
fi

# Create virtual environment if it does not exist
if [ -d "$ENV_DIR" ]; then
  echo "Virtual environment '$ENV_DIR' already exists. Skipping creation."
else
  echo "Creating virtual environment: $ENV_DIR"
  python3 -m venv "$ENV_DIR"
fi

# Activate the environment
echo "Activating the virtual environment..."
source "$ENV_DIR/bin/activate"

# Install dependencies
echo "Upgrading pip and installing dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r "$REQ_FILE"

echo "----------------------------------------"
echo "Setup complete. The environment is now active."
echo "Current Python: $(python3 --version)"
echo "----------------------------------------"