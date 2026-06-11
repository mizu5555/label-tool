#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
VENV=".venv-build"

"$PYTHON_BIN" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r requirements.txt
"$VENV/bin/python" -m PyInstaller --clean --noconfirm label_tool.spec

echo
if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "Build complete: dist/label_tool"
else
  echo "Build complete: dist/label_tool"
fi
