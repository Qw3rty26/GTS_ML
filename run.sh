#!/usr/bin/env bash
set -e

MAIN="py/run.py"

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python "$MAIN" "$@"
