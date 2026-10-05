#!/usr/bin/env bash

MAIN="py/run.py"

nix-shell --run "
if [ ! -d .venv ]; then
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
fi

source .venv/bin/activate
python $MAIN $*
"
