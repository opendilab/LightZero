#!/usr/bin/env bash
set -euo pipefail

# 1) system‐level deps (run once under WSL/Ubuntu):
sudo apt update
sudo apt install -y            \
    build-essential            \
    python3.11-venv            \
    python3.11-dev             \
    libgl1-mesa-glx            \
    libglib2.0-0               \
    libssl-dev                 \
    zlib1g-dev                 \
    libbz2-dev                 \
    libreadline-dev            \
    libsqlite3-dev             \
    libncursesw5-dev           \
    libxml2-dev                \
    libxmlsec1-dev             \
    libffi-dev                 \
    liblzma-dev

# 2) create & activate venv
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate

# 3) upgrade packaging tools & install locked deps
pip install --upgrade pip setuptools wheel
pip install -r requirements.lock.txt

# 4) compile C‐trees in‐place
python setup.py build_ext --inplace

echo ""
echo "✅  Bootstrap complete!"
echo "   * Activate via:  source .venv/bin/activate"
echo "   * Then run UniZero:  python -m zoo.atari.config.atari_unizero_config --env BoxingNoFrameskip-v4 --seed 0"
