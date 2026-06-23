#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

echo "Starting vocabulary estimator at http://${HOST}:${PORT}"
echo "API docs: http://${HOST}:${PORT}/docs"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "${PYTHON_BIN}" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi

"${PYTHON_BIN}" -m uvicorn server.main:app --host "${HOST}" --port "${PORT}" --reload
