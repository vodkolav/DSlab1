#!/usr/bin/env bash

# Exit immediately if any command fails
set -e

VENV_DIR=".venv"

echo "=== 1. Creating virtual environment in ${VENV_DIR} ==="
python3 -m venv "${VENV_DIR}"

echo "=== 2. Upgrading pip ==="
"${VENV_DIR}/bin/pip" install --upgrade pip

echo "=== 3. Installing project in editable mode + dependencies ==="
# The dot (.) means "this project". 
# [dev] tells pip to also install the optional dev dependencies (like ipykernel).
"${VENV_DIR}/bin/pip" install -e ".[dev]"

echo "=== 4. Registering venv with VS Code ==="
# Create .vscode directory if it doesn't exist
mkdir -p .vscode

# Point VS Code's Python extension directly to our new venv
cat <<EOF > .vscode/settings.json
{
    "python.defaultInterpreterPath": "\${workspaceFolder}/${VENV_DIR}/bin/python"
}
EOF

echo "=== 5. Registering Jupyter Kernel ==="
# This ensures the venv explicitly shows up in VS Code's Notebook kernel picker
"${VENV_DIR}/bin/python" -m ipykernel install --user --name=my_package_dev --display-name "Python (Rockets dev)"

echo "=== Setup Complete! ==="
echo "To activate manually in your terminal, run: source ${VENV_DIR}/bin/activate"