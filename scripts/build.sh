#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
VENV=".venv-build"
OS_NAME="$(uname -s)"
DEBUG="${DEBUG:-0}"

if [[ "$OS_NAME" == "Darwin" ]]; then
  PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$PY_VERSION" != "3.12" ]]; then
    echo "macOS packaging requires Python 3.12 for the PyQt5/PyInstaller frozen build." >&2
    echo "Run with: PYTHON=/opt/homebrew/bin/python3.12 ./scripts/build.sh" >&2
    exit 1
  fi
fi

rm -rf "$VENV" build dist
"$PYTHON_BIN" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r requirements.txt

if [[ "$DEBUG" == "1" ]]; then
  export LABEL_TOOL_DEBUG=1
  echo "Building debug artifact with PyInstaller console/debug enabled."
else
  unset LABEL_TOOL_DEBUG
fi

"$VENV/bin/python" -m PyInstaller --clean --noconfirm label_tool.spec

echo
if [[ "$OS_NAME" == "Darwin" ]]; then
  if [[ "$DEBUG" == "1" ]]; then
    if [[ ! -f "dist/label_tool" ]]; then
      echo "Build failed: dist/label_tool was not created." >&2
      exit 1
    fi
    echo "Build complete: dist/label_tool"
    echo "Run with: ./dist/label_tool"
  elif [[ ! -d "dist/Label Tool.app" ]]; then
    echo "Build failed: dist/Label Tool.app was not created." >&2
    exit 1
  else
    echo "Build complete: dist/Label Tool.app"
    echo 'Run with: open "dist/Label Tool.app"'
  fi
else
  if [[ ! -f "dist/label_tool" ]]; then
    echo "Build failed: dist/label_tool was not created." >&2
    exit 1
  fi
  echo "Build complete: dist/label_tool"
  echo "Run with: ./dist/label_tool"
fi
