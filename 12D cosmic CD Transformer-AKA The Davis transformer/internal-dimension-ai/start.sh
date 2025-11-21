#!/bin/bash
# Quick start script - activates environment and runs the easy interface

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running setup first..."
    ./setup.sh
fi

source venv/bin/activate
python run_easy.py
