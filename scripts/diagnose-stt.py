"""Diagnóstico STT: corre whisper-cli sobre el último WAV guardado."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Hacemos que se pueda ejecutar tanto con `python scripts/diagnose-stt.py`
# como con `python -m scripts.diagnose_stt`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis.config import Config


async def main() -> int:
    cfg = Config().stt
    wav = Path(os.environ.get("JARVIS_DEBUG_DIR", "debug-audio")) / "last-utt.wav"
    if not wav.is_file():
        print(f"no encuentro {wav}. Corré antes `jarvis run` con JARVIS_DEBUG_DIR")
        return 1
    binary = cfg.resolve_binary()
    if not binary:
        print(f"no encuentro {cfg.binary}")
        return 1
    args = [
        binary,
        "-m", str(Path(cfg.model).expanduser()),
        "-f", str(wav),
        "-l", cfg.language,
        "-t", str(cfg.threads),
        "--no-prints",
        "-nt",
    ]
    print("ejecutando:", " ".join(args))
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    print("--- stdout ---")
    print(out.decode("utf-8", "replace"))
    print("--- stderr ---")
    print(err.decode("utf-8", "replace"))
    print("returncode:", proc.returncode)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
