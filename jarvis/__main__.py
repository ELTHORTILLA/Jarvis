"""Punto de entrada: python -m jarvis [comando]

Comandos:
  run       ciclo de voz completo (por defecto)
  doctor    revisa dependencias y credenciales
  devices   lista dispositivos de audio
  say       sintetiza y reproduce un texto (prueba de TTS)
  listen    graba y transcribe una vez (prueba de STT)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .audio import Player, Recorder, describe_devices
from .config import Config
from .executor import build_executor
from .orchestrator import Assistant
from .stt import WhisperCppTranscriber
from .tts import MiniMaxTTS


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


async def cmd_doctor(cfg: Config) -> int:
    executor = build_executor(cfg.executor)
    checks: list[tuple[str, list[str]]] = [
        ("TTS (MiniMax)", MiniMaxTTS(cfg.tts).preflight()),
        ("STT (whisper.cpp)", WhisperCppTranscriber(cfg.stt).preflight()),
        (f"Ejecutor ({executor.name})", executor.preflight()),
    ]

    failed = False
    for label, problems in checks:
        if problems:
            failed = True
            print(f"✗ {label}")
            for p in problems:
                print(f"    - {p}")
        else:
            print(f"✓ {label}")

    try:
        import sounddevice as sd

        default_in = sd.query_devices(kind="input")
        print(f"✓ Micrófono: {default_in['name']}")
    except Exception as exc:
        failed = True
        print(f"✗ Micrófono: {exc}")

    if cfg.executor.backend == "dry-run":
        print("\nℹ️  Modo simulación activo: no se ejecutarán acciones reales.")
        print("   Para dar control real: JARVIS_BACKEND=llm (ver README).")
    elif cfg.executor.backend == "openclaw":
        print("\n⚠️  Modo OpenClaw: las órdenes se ejecutarán en este equipo.")
    else:
        print("\n🤖 Modo LLM (MiniMax M3 + tools locales).")
        print("   Shell y open_app deshabilitados. Para habilitarlos: JARVIS_ALLOW_DANGEROUS=1")
    return 1 if failed else 0


async def cmd_say(cfg: Config, text: str) -> int:
    async with MiniMaxTTS(cfg.tts) as tts:
        if problems := tts.preflight():
            print("\n".join(problems), file=sys.stderr)
            return 1
        wav = await tts.synthesize(text)
        print(f"{len(wav)} bytes de audio; reproduciendo…")
        await Player().play_wav(wav)
    return 0


async def cmd_listen(cfg: Config) -> int:
    transcriber = WhisperCppTranscriber(cfg.stt)
    if problems := transcriber.preflight():
        print("\n".join(problems), file=sys.stderr)
        return 1
    print("🎙️  Habla ahora…")
    samples = await Recorder(cfg.audio).record()
    if samples is None:
        print("no se capturó audio", file=sys.stderr)
        return 1
    print(f"capturados {len(samples) / cfg.audio.sample_rate:.1f}s; transcribiendo…")
    print(f"→ {await transcriber.transcribe(samples, cfg.audio.sample_rate)!r}")
    return 0


async def main_async(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis", description="Asistente de voz local")
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run", "doctor", "devices", "say", "listen"],
    )
    parser.add_argument("text", nargs="*", help="texto para el comando 'say'")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)
    cfg = Config()

    if args.command == "devices":
        print(describe_devices())
        return 0
    if args.command == "doctor":
        return await cmd_doctor(cfg)
    if args.command == "say":
        text = " ".join(args.text) or "Hola, soy tu asistente."
        return await cmd_say(cfg, text)
    if args.command == "listen":
        return await cmd_listen(cfg)

    # run: exige que las dependencias estén sanas antes de empezar.
    if await cmd_doctor(cfg) != 0:
        print("\nCorrige lo anterior antes de continuar.", file=sys.stderr)
        return 1
    await Assistant(cfg, build_executor(cfg.executor)).run()
    return 0


def main() -> int:
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
