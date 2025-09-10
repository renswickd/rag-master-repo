#!/usr/bin/env bash
set -euo pipefail

echo "==> RAG project setup starting"

# Determine repo root (directory of this script's parent)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# 1) Create data directories
echo "==> Creating data directories"
mkdir -p data/source_data/basic-rag \
         data/source_data/multi-modal \
         data/source_data/langgraph \
         data/source_data/rag-ubac \
         data/source_data/agentic_rag \
         chroma_db

# 2) Choose Python
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  echo "Error: python not found on PATH" >&2
  exit 1
fi

# 3) Create virtual environment (.venv)
VENV_DIR="${REPO_ROOT}/.venv"
if [[ -d "$VENV_DIR" ]]; then
  echo "==> Reusing existing virtualenv at .venv"
else
  echo "==> Creating virtualenv at .venv"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PY="${VENV_DIR}/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "Error: virtualenv python not found at $VENV_PY" >&2
  exit 1
fi

# 4) Upgrade pip and install in editable mode
echo "==> Upgrading pip and installing package (editable)"
"$VENV_PY" -m pip install --upgrade pip setuptools wheel
"$VENV_PY" -m pip install -e .

echo "\n==> Setup complete"
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo "Then run, for example:"
echo "  python main.py --rag_type agentic-rag -v"
echo "  python main.py --rag_type agentic-rag"

