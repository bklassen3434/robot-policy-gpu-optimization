"""Glue: instruction (typed or spoken) -> SmolVLA rollout -> await next command.

Two modes, matching the plan's phasing:
  --typed   Phase 3 — type "pick up the red block", run one rollout. No mic/Whisper.
  (voice)   Phase 4 — speak the command; Whisper transcribes, then run the rollout.

The robot connects and the policy loads ONCE; each command is a fresh conditioned
rollout on the same live process.

    python -m robopolicy.realbot.agent --typed      # Phase 3
    python -m robopolicy.realbot.agent              # Phase 4 (voice)
"""

from __future__ import annotations

import argparse
import sys

from .config import DEFAULT_CONFIG, load_config
from .run_policy import SO101Runner
from .voice import Instruction, listen_once, normalize_instruction


def _typed_loop(runner: SO101Runner, objects, synonyms, template) -> None:
    print(f"Typed mode. Objects: {objects}. Type a command, or 'q' to quit.")
    while True:
        try:
            raw = input(">> command: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if raw.lower() in {"q", "quit", "exit"}:
            return
        instr = normalize_instruction(raw, objects, synonyms, template)
        if instr is None:
            print(f"   (need a verb + one of: {', '.join(objects)})")
            continue
        _run(runner, instr)


def _voice_loop(runner: SO101Runner, objects, synonyms, template, model) -> None:
    print(f"Voice mode. Objects: {objects}. Ctrl-C to quit.")
    while True:
        try:
            instr = listen_once(objects, synonyms, template, model=model)
        except KeyboardInterrupt:
            print("\nbye")
            return
        if instr is not None:
            _run(runner, instr)


def _run(runner: SO101Runner, instr: Instruction) -> None:
    print(f"   -> {instr.canonical!r}")
    runner.run_instruction(instr.canonical)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Voice/typed SO-101 object-selection agent.")
    ap.add_argument("--typed", action="store_true", help="typed commands (Phase 3, no mic)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--model", default=None, help="override MLX-Whisper model (voice mode)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    task = cfg["task"]
    objects = task["objects"]
    synonyms = task.get("verb_synonyms", ["pick up"])
    template = task.get("instruction_template", "pick up the {object}")

    runner = SO101Runner(args.config)
    try:
        if args.typed:
            _typed_loop(runner, objects, synonyms, template)
        else:
            from .voice import DEFAULT_MODEL
            _voice_loop(runner, objects, synonyms, template, args.model or DEFAULT_MODEL)
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
