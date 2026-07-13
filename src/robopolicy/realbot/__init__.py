"""Path B — physical SO-101 track.

Voice- & vision-commanded object selection with a fine-tuned SmolVLA. This
subpackage is the genuinely new, repo-specific code; the heavy lifting
(teleop / record / train) is LeRobot's own CLI (see the record runbook and
scripts/runpod_smolvla.sh).

Modules
-------
config          load configs/smolvla_so101.yaml as a plain dict
voice           mic capture -> Whisper transcription -> canonical instruction
run_policy      load a fine-tuned SmolVLA and run it on the SO-101 from a string
agent           glue: voice/typed instruction -> rollout loop -> await next
detect_cameras  one-shot helper to discover stable camera ids + serial ports
"""

from .config import load_config

__all__ = ["load_config"]
