#!/bin/zsh
# Launch Alis Studio with the local Krea 2 base (encoder/VAE/tokenizer) — no gated HF download needed.
cd "${0:A:h}"
export KREA2_BASE_DIR=/Users/hk/ComfyUI-Shared/models/krea2/base
exec .venv/bin/python app.py "$@"
