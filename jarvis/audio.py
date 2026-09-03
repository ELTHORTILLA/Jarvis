"""Captura y reproducción de audio.

Usa sounddevice (PortAudio) en lugar de pyaudio: la misma librería nativa por
debajo, pero con ruedas precompiladas para Windows y Linux y una API que no
bloquea el event loop.
"""

from __future__ import annotations

import asyncio
import io
import logging
import queue
import wave

import numpy as np
import sounddevice as sd

from .config import AudioConfig

log = logging.getLogger(__name__)


class MicrophoneError(RuntimeError):
    pass


def _rms(block: np.ndarray) -> float:
    if block.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))


class Recorder:
    """Graba una intervención y corta al detectar silencio.

    El umbral se calibra contra el ruido de fondo al inicio de cada escucha,
    de modo que un ventilador o un aire acondicionado no mantengan la
    grabación abierta indefinidamente.
    """

    def __init__(self, cfg: AudioConfig) -> None:
        self.cfg = cfg

    def _record_blocking(self) -> np.ndarray | None:
        cfg = self.cfg
        blocks: queue.Queue[np.ndarray] = queue.Queue()

        def callback(indata, _frames, _time, status) -> None:
            if status:
                log.debug("estado del stream: %s", status)
            blocks.put(indata[:, 0].copy())

        collected: list[np.ndarray] = []
        speech: list[np.ndarray] = []
        block_secs = cfg.block_ms / 1000
        try:
            stream = sd.InputStream(
                samplerate=cfg.sample_rate,
                channels=cfg.channels,
                blocksize=cfg.block_size,
                dtype="float32",
                callback=callback,
            )
        except Exception as exc:  # PortAudio lanza tipos variados
            raise MicrophoneError(f"no se pudo abrir el micrófono: {exc}") from exc

        with stream:
            # 1. Calibración: medimos el ruido ambiente.
            noise: list[float] = []
            deadline = cfg.calibration_time
            elapsed = 0.0
            while elapsed < deadline:
                try:
                    block = blocks.get(timeout=1.0)
                except queue.Empty:
                    break
                noise.append(_rms(block))
                elapsed += block_secs

            if cfg.silence_threshold > 0:
                threshold = cfg.silence_threshold
            else:
                floor = float(np.median(noise)) if noise else 0.0
                # 3x el ruido de fondo, con un piso mínimo para micros muy limpios.
                threshold = max(floor * 3.0, 0.012)
            log.debug("umbral de voz: %.5f", threshold)

            # 2. Esperamos a que empiece a hablar.
            waited = 0.0
            while waited < cfg.start_timeout:
                try:
                    block = blocks.get(timeout=1.0)
                except queue.Empty:
                    continue
                waited += block_secs
                if _rms(block) >= threshold:
                    speech.append(block)
                    break
            else:
                return None
            if not speech:
                return None

            # 3. Grabamos hasta que el silencio se sostenga.
            collected.extend(speech)
            silence = 0.0
            total = block_secs * len(collected)
            while silence < cfg.silence_duration and total < cfg.max_utterance:
                try:
                    block = blocks.get(timeout=1.0)
                except queue.Empty:
                    break
                collected.append(block)
                total += block_secs
                silence = 0.0 if _rms(block) >= threshold else silence + block_secs

        if not collected:
            return None
        return np.concatenate(collected)

    async def record(self) -> np.ndarray | None:
        """Graba sin bloquear el event loop."""
        return await asyncio.to_thread(self._record_blocking)


def to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Empaqueta float32 [-1, 1] como WAV PCM 16-bit, que es lo que come Whisper."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as wav:
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
        width = wav.getsampwidth()
        channels = wav.getnchannels()
    if width != 2:
        raise ValueError(f"se esperaba WAV de 16 bits, se recibió {width * 8}")
    audio = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, rate


class Player:
    """Reproduce audio y permite cortarlo (para interrumpir un filler)."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def play_wav(self, data: bytes) -> None:
        samples, rate = decode_wav(data)
        await self.play(samples, rate)

    async def play(self, samples: np.ndarray, sample_rate: int) -> None:
        async with self._lock:
            await asyncio.to_thread(self._play_blocking, samples, sample_rate)

    @staticmethod
    def _play_blocking(samples: np.ndarray, sample_rate: int) -> None:
        sd.play(samples, samplerate=sample_rate)
        sd.wait()

    @staticmethod
    def stop() -> None:
        sd.stop()


def describe_devices() -> str:
    try:
        return str(sd.query_devices())
    except Exception as exc:
        return f"no se pudieron listar los dispositivos: {exc}"
