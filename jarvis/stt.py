"""Transcripción (STT) mediante whisper.cpp local.

MiniMax no expone API de ASR (POST /v1/audio/transcriptions responde 404), así
que la transcripción se resuelve en local. whisper.cpp funciona igual en Linux
y Windows, lo que mantiene el esqueleto portable.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path

import numpy as np

from .audio import to_wav_bytes
from .config import STTConfig

log = logging.getLogger(__name__)

# whisper.cpp emite líneas "[00:00:00.000 --> 00:00:02.000]   texto".
_TIMESTAMP = re.compile(r"^\s*\[[\d:.]+\s*-->\s*[\d:.]+\]\s*")
# Anotaciones no verbales que Whisper inventa en los silencios.
_NOISE = re.compile(r"^[\(\[\*].*[\)\]\*]$")


class STTError(RuntimeError):
    pass


class WhisperCppTranscriber:
    def __init__(self, cfg: STTConfig) -> None:
        self.cfg = cfg

    def preflight(self) -> list[str]:
        """Devuelve los problemas que impedirían transcribir."""
        problems: list[str] = []
        if not self.cfg.resolve_binary():
            problems.append(
                f"no se encontró el binario '{self.cfg.binary}' en PATH "
                "(instalar: sudo pacman -S whisper-cpp)"
            )
        if not self.cfg.model:
            problems.append("JARVIS_WHISPER_MODEL no está definido (ruta a un ggml-*.bin)")
        elif not Path(self.cfg.model).expanduser().is_file():
            problems.append(f"no existe el modelo: {self.cfg.model}")
        return problems

    async def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        problems = self.preflight()
        if problems:
            raise STTError("; ".join(problems))

        binary = self.cfg.resolve_binary()
        assert binary  # garantizado por preflight
        wav = to_wav_bytes(samples, sample_rate)

        # whisper.cpp lee de fichero. Usamos `.cache/stt/` en lugar de
        # `tempfile.mkstemp` porque algunos antivirus (Defender sobre todo)
        # bloquean o corrompen lecturas desde %TEMP% mientras los escanean,
        # y eso hace que whisper-cli devuelva stdout vacío aunque el archivo
        # exista y tenga el contenido correcto.
        cache_dir = Path(".cache") / "stt"
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_dir / f"jarvis-stt-{os.getpid()}-{id(self)}.wav"
        try:
            tmp.write_bytes(wav)
            args = [
                binary,
                "-m",
                str(Path(self.cfg.model).expanduser()),
                "-f",
                str(tmp),
                "-l",
                self.cfg.language,
                "-t",
                str(self.cfg.threads),
                "--no-prints",  # silencia el banner de carga del modelo
                "-nt",  # sin marcas de tiempo
            ]
            log.debug("ejecutando: %s", " ".join(args))
            # Debug: dejamos una copia del WAV enviado a Whisper en
            # JARVIS_DEBUG_DIR (si está definida) para poder reproducirlo y
            # entender qué está escuchando realmente el STT.
            debug_dir = os.environ.get("JARVIS_DEBUG_DIR")
            if debug_dir:
                debug_path = Path(debug_dir) / "last-utt.wav"
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                debug_path.write_bytes(wav)
                rms = float(np.sqrt(np.mean(np.square(samples))))
                log.debug("WAV de debug copiado a %s (rms=%.5f, sr=%d, len=%d)",
                          debug_path, rms, sample_rate, len(samples))
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            # Borramos antes de parsear, pero después de que el subproceso
            # haya soltado el handle. En Linux da igual, en Windows era
            # crítico.
            await asyncio.to_thread(_safe_unlink, tmp)
            if proc.returncode != 0:
                detail = stderr.decode("utf-8", "replace").strip() or "sin detalle"
                raise STTError(f"whisper.cpp falló (código {proc.returncode}): {detail}")
            raw = stdout.decode("utf-8", "replace")
            return self._clean(raw)
        except BaseException:
            # Si algo falla antes de communicate, intentamos borrar igual.
            await asyncio.to_thread(_safe_unlink, tmp)
            raise

    @staticmethod
    def _clean(raw: str) -> str:
        lines: list[str] = []
        for line in raw.splitlines():
            line = _TIMESTAMP.sub("", line).strip()
            if not line or _NOISE.match(line):
                continue
            lines.append(line)
        return " ".join(lines).strip()


def _safe_unlink(path: Path) -> None:
    """Borra `path` ignorando errores; útil en Windows donde el handle puede
    no estar liberado al instante."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
