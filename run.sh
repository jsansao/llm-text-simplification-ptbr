#!/usr/bin/env bash
set -euo pipefail

# Activate virtual environment and run evaluation
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if [ ! -d .venv ]; then
    echo "Criando ambiente virtual..."
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
fi

echo "Carregando .env se existir..."
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

exec .venv/bin/python src/run.py "$@"
