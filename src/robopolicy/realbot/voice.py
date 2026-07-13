"""Voice front-end: mic capture -> Whisper -> canonical instruction string.

Apple-Silicon-native: transcription uses MLX-Whisper (Metal) on the M3. A
small/base model is plenty for short "pick up the {color} block" commands.

The parsing (`normalize_instruction`) is pure Python and dependency-free, so it
is unit-testable without a mic or a model. Audio capture and transcription
lazy-import sounddevice / mlx_whisper so importing this module stays cheap.

CLI (loop: speak -> transcription -> normalized instruction):
    python -m robopolicy.realbot.voice
    python -m robopolicy.realbot.voice --once
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

from .config import load_config

# MLX-Whisper HF repo. base(.en) is a good speed/accuracy point for short commands.
DEFAULT_MODEL = "mlx-community/whisper-base.en-mlx"
SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono


# ─────────────────────────────────────────────────────────────────────────────
# Parsing — pure Python, no deps.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Instruction:
    """A parsed command."""
    canonical: str      # e.g. "pick up the red block" — what SmolVLA is conditioned on
    target: str         # e.g. "red block"
    raw: str            # the raw transcript


def normalize_instruction(
    text: str,
    objects: list[str],
    verb_synonyms: list[str],
    template: str = "pick up the {object}",
) -> Instruction | None:
    """Map a free-form transcript onto a canonical instruction, or None.

    Accepts any of the verb synonyms ("grab the red block", "get the duck") and
    normalizes to the canonical template so training and inference stay aligned.
    Matching is case-insensitive and requires a known object to be named.
    """
    if not text:
        return None
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)          # drop punctuation
    t = re.sub(r"\s+", " ", t).strip()

    # Require an intent verb so we don't fire on incidental object mentions.
    verbs = [v.lower() for v in verb_synonyms] or ["pick up"]
    if not any(v in t for v in verbs):
        return None

    # Longest object name first so "red block" wins over a bare "block".
    for obj in sorted(objects, key=len, reverse=True):
        if obj.lower() in t:
            return Instruction(
                canonical=template.format(object=obj),
                target=obj,
                raw=text,
            )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Audio capture (lazy deps).
# ─────────────────────────────────────────────────────────────────────────────
def record_utterance(
    max_seconds: float = 8.0,
    silence_seconds: float = 0.8,
    silence_rms: float = 0.012,
    sample_rate: int = SAMPLE_RATE,
):
    """Push-to-talk capture: press Enter, speak, auto-stop on trailing silence.

    Returns a float32 mono numpy array in [-1, 1]. Requires sounddevice + numpy.
    """
    import numpy as np
    import sounddevice as sd

    input(">> Press Enter, then speak…")
    block = int(sample_rate * 0.05)                 # 50 ms blocks
    quiet_needed = int(silence_seconds / 0.05)
    max_blocks = int(max_seconds / 0.05)

    chunks, quiet_run, spoke = [], 0, False
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32",
                        blocksize=block) as stream:
        for _ in range(max_blocks):
            data, _overflow = stream.read(block)
            mono = data[:, 0].copy()
            chunks.append(mono)
            rms = float(np.sqrt(np.mean(mono ** 2)) + 1e-9)
            if rms >= silence_rms:
                spoke, quiet_run = True, 0
            elif spoke:
                quiet_run += 1
                if quiet_run >= quiet_needed:
                    break
    audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype="float32")
    return audio


def transcribe(audio, model: str = DEFAULT_MODEL) -> str:
    """Transcribe a float32 mono@16k array with MLX-Whisper. Returns text."""
    import mlx_whisper

    result = mlx_whisper.transcribe(audio, path_or_hf_repo=model)
    return (result.get("text") or "").strip()


# ─────────────────────────────────────────────────────────────────────────────
# One-shot listen: record -> transcribe -> normalize.
# ─────────────────────────────────────────────────────────────────────────────
def listen_once(
    objects: list[str],
    verb_synonyms: list[str],
    template: str = "pick up the {object}",
    model: str = DEFAULT_MODEL,
) -> Instruction | None:
    """Capture one spoken command and parse it. None if nothing usable was heard."""
    audio = record_utterance()
    text = transcribe(audio, model=model)
    print(f"   heard: {text!r}")
    instr = normalize_instruction(text, objects, verb_synonyms, template)
    if instr is None:
        known = ", ".join(objects)
        print(f"   (no known object heard — say a verb + one of: {known})")
    return instr


def _cli(argv: list[str]) -> int:
    once = "--once" in argv
    cfg = load_config()
    task = cfg["task"]
    objects = task["objects"]
    synonyms = task.get("verb_synonyms", ["pick up"])
    template = task.get("instruction_template", "pick up the {object}")

    print(f"Voice test. Objects: {objects}. Ctrl-C to quit.")
    try:
        while True:
            instr = listen_once(objects, synonyms, template)
            if instr:
                print(f"   -> instruction: {instr.canonical!r}")
            if once:
                return 0
    except KeyboardInterrupt:
        print("\nbye")
        return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
