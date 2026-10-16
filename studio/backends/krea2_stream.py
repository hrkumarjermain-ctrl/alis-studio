"""Krea 2 Turbo · q8 4-step, block-streaming — for 16 GB Macs.

Runs mh73772/krea-2-turbo-mflux-q8-4step through the mflux-for-16gb fork
(github.com/he-be/mflux-for-16gb, branch feat/krea2-block-streaming), whose --block-streaming mode
reads one transformer block at a time from disk: ~7–8 GB peak at 1024² instead of ~20 GB.

The fork pins its own mflux, which conflicts with the mflux this app imports, so it runs as a
separate process from its own venv. That also hands every byte of memory back after each render.
Paths: KREA2_STREAM_BIN (fork's mflux-generate-krea2) and KREA2_STREAM_MODEL (assembled snapshot).
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import tempfile

from PIL import Image

from .base import Backend
from .mflux_common import _lora_args, _lora_params

_BIN = os.environ.get("KREA2_STREAM_BIN",
                      os.path.expanduser("~/ComfyUI-Shared/tools/mflux-16gb/.venv/bin/mflux-generate-krea2"))
_MODEL = os.environ.get("KREA2_STREAM_MODEL",
                        os.path.expanduser("~/ComfyUI-Shared/models/mlx/mh73772__krea-2-turbo-mflux-q8-4step"))
# variant id -> snapshot dir; baked-LoRA variants are hard-linked siblings made by tools/bake_krea2_lora.py
_VARIANT_DIRS = {"q8-4step": _MODEL, "q8-4step-refusal-v2": _MODEL + "__refusal-v2",
                 "q8-4step-refusal-v2-075": _MODEL + "__refusal-v2-0.75"}
_PROGRESS = re.compile(r"(\d+)/(\d+) \[")   # tqdm: " 50%|█████     | 2/4 [00:20<00:20, ...]"


class Krea2StreamBackend(Backend):
    id = "krea2-stream"
    label = "Krea 2 Turbo · q8 4-step (16 GB)"
    info = "Streams the transformer from disk block by block — 1024² fits in 16 GB. Runs in its own process."
    prompt_note = "Understands Korean and other languages natively (Qwen3 text encoder)."
    min_ram_gib = 16
    variants = [{"id": "q8-4step", "label": "q8 · 4-step · streaming"},
                {"id": "q8-4step-refusal-v2-075", "label": "q8 · 4-step · refusal-reduction v2 @ 0.75 baked"},
                {"id": "q8-4step-refusal-v2", "label": "q8 · 4-step · refusal-reduction v2 @ 1.00 baked"}]
    params = [
        {"key": "resolution", "label": "Resolution", "type": "resolution", "group": "Output",
         "sizes": [512, 768, 1024, 1280], "default_size": 1024,
         "aspects": ["1:1", "3:2", "2:3", "16:9", "9:16"], "default_aspect": "1:1",
         "min": 256, "max": 1280, "multiple": 16},
        {"key": "steps", "label": "Steps", "type": "int", "group": "Output", "min": 1, "max": 12, "default": 4,
         "hint": "The 4-step distill LoRA is merged in — 4 is the intended count."},
        {"key": "num_images", "label": "Images", "type": "int", "group": "Output", "min": 1, "max": 4, "default": 1},
        {"key": "seed", "label": "Seed", "type": "seed", "group": "Sampling", "default": 0},
        {"key": "guidance", "label": "Guidance (CFG)", "type": "float", "group": "Sampling",
         "min": 1, "max": 1, "step": 0.5, "default": 1.0, "fixed": True,
         "hint": "Distilled — CFG off (mflux expresses that as guidance 1.0)."},
        {"key": "sampler", "label": "Sampler", "type": "select", "group": "Sampling",
         "options": [{"value": "er_sde", "label": "ER-SDE (Turbo default, stochastic)"},
                     {"value": "euler", "label": "Euler (flow-match, deterministic)"}], "default": "er_sde",
         "hint": "The two samplers this engine implements; both use Krea's dynamic-shift flow schedule."},
        *_lora_params(),   # applied at runtime (--no-bake-lora): the only LoRA mode compatible with streaming
    ]

    @classmethod
    def is_available(cls) -> bool:
        return os.path.exists(_BIN) and os.path.isdir(os.path.join(_MODEL, "transformer"))

    def is_installed(self, variant):
        return os.path.isdir(os.path.join(_VARIANT_DIRS.get(variant, ""), "transformer"))

    def generate(self, *, prompt, variant, params, step_callback):
        n = int(params.get("num_images", 1))
        seed = int(params.get("seed", 0))
        steps = int(params.get("steps", 4))
        with tempfile.TemporaryDirectory(prefix="krea2stream-") as out:
            model = _VARIANT_DIRS.get(variant, _MODEL)
            if not self.is_installed(variant):
                raise ValueError(f"Variant '{variant}' isn't on disk ({model}).")
            sampler = params.get("sampler") if params.get("sampler") in ("er_sde", "euler") else "er_sde"
            cmd = [_BIN, "--model", model, "--base-model", "krea-2", "--block-streaming",
                   "--steps", str(steps), "--scheduler", sampler, "--guidance", "1.0",
                   "--width", str(int(params.get("width", 1024))), "--height", str(int(params.get("height", 1024))),
                   "--prompt", prompt, "--output", os.path.join(out, "image.png"),
                   "--seed", *[str(seed + i) for i in range(n)]]
            lora_paths, lora_scales = _lora_args(params)
            for path, scale in zip(lora_paths or [], lora_scales or []):
                cmd += ["--lora", path, str(scale)]
            if lora_paths:
                cmd.append("--no-bake-lora")

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log, buf, last = [], b"", None
            try:
                while chunk := proc.stdout.read1(4096):
                    buf += chunk
                    *lines, buf = re.split(rb"[\r\n]", buf)   # tqdm redraws with \r
                    for line in (x.decode(errors="ignore") for x in lines):
                        log.append(line)
                        m = _PROGRESS.search(line)
                        if m and int(m.group(1)) and m.groups() != last:   # tqdm repeats a step on redraw
                            last = m.groups()
                            step_callback(int(m.group(1)), int(m.group(2)))   # raises when the user hits Stop
            except BaseException:
                proc.kill()
                proc.wait()
                raise
            if proc.wait() != 0:
                tail = "\n".join(x for x in log if x.strip())[-800:]
                raise RuntimeError(f"Krea 2 streaming run failed (exit {proc.returncode}):\n{tail}")
            files = sorted(glob.glob(os.path.join(out, "*.png")))
            if not files:
                raise RuntimeError("Krea 2 streaming run finished but wrote no image.")
            return [Image.open(f).copy() for f in files]
