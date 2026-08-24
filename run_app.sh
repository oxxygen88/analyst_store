#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  ./setup_linux.sh
fi
exec .venv/bin/python -m streamlit run app.py
