"""Diagnóstico: graba 3s del micrófono activo y reporta RMS máximo.

Útil para distinguir entre:
  * dispositivo equivocado (RMS muy bajo, ~0)
  * umbral mal calibrado (RMS > 0.05)
  * AGC o compresión de Windows (RMS saturado en ~0.1)
"""
from __future__ import annotations

import sys

import numpy as np
import sounddevice as sd

from jarvis.config import AudioConfig, Config


def main() -> int:
    cfg = Config().audio
    idx = cfg.input_device
    info = sd.query_devices(idx, kind="input") if idx is not None else sd.query_devices(kind="input")
    name = info["name"]
    samplerate = int(info["default_samplerate"])
    print(f"dispositivo: [{idx}] {name}")
    print(f"sample rate del device: {samplerate}")
    print(f"configurado input_device: {idx}")
    print(f"samplerate de Jarvis: {cfg.sample_rate}")
    print(f"channels: {cfg.channels}")

    # Si el device abre a 48k y Jarvis pide 16k, sounddevice hace resample
    # automático pero a veces se cae el stream. Avisamos.
    if samplerate != cfg.sample_rate:
        print(f"⚠ el device trabaja a {samplerate} Hz, Jarvis pide {cfg.sample_rate} Hz — habrá resample")

    blocks: list[np.ndarray] = []
    def cb(indata, _frames, _time, status):
        if status:
            print(f"status: {status}", file=sys.stderr)
        blocks.append(indata[:, 0].copy())

    # Probamos WASAPI exclusivo si está disponible
    try:
        extra = sd.WasapiSettings(exclusive=True)
        stream = sd.InputStream(
            samplerate=cfg.sample_rate,
            channels=cfg.channels,
            blocksize=cfg.block_size,
            dtype="float32",
            device=idx,
            callback=cb,
            extra_settings=extra,
        )
        print("intentando WASAPI exclusivo...")
    except Exception as exc:
        print(f"no se pudo WASAPI exclusivo ({exc}); abriendo compartido")
        stream = sd.InputStream(
            samplerate=cfg.sample_rate,
            channels=cfg.channels,
            blocksize=cfg.block_size,
            dtype="float32",
            device=idx,
            callback=cb,
        )

    print("grabando 3 segundos... HABLA AHORA")
    with stream:
        sd.sleep(3000)

    if not blocks:
        print("no se capturó nada")
        return 1

    all_audio = np.concatenate(blocks)
    rms = float(np.sqrt(np.mean(np.square(all_audio))))
    peak = float(np.max(np.abs(all_audio)))
    print(f"RMS total:  {rms:.5f}")
    print(f"peak:       {peak:.5f}")
    if rms < 0.01:
        print("→ RMS muy bajo. Posible problema: device equivocado o AGC agresivo")
    elif rms < 0.05:
        print("→ RMS bajo. Subí el volumen de la Snowball o hablá más fuerte")
    else:
        print("→ RMS sano. Si Jarvis no te escucha, el problema está en el orquestador")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
