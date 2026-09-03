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

        # Si el usuario fijó un device, preguntamos su sample rate nativo:
        # sounddevice en modo compartido puede fallar si le pedimos uno
        # distinto. Si el device es el default de Windows (None), respetamos
        # el de la config directamente.
        device_sr = cfg.sample_rate
        if cfg.input_device is not None:
            try:
                device_sr = int(sd.query_devices(cfg.input_device, kind="input")["default_samplerate"])
            except Exception as exc:
                log.debug("no pude leer el sample rate del device, uso %d: %s", cfg.sample_rate, exc)

        def _open(sr: int):
            return sd.InputStream(
                samplerate=sr,
                channels=cfg.channels,
                blocksize=int(sr * cfg.block_ms / 1000),
                dtype="float32",
                device=cfg.input_device,
                callback=callback,
            )

        try:
            stream = _open(device_sr)
            actual_sr = device_sr
        except sd.PortAudioError as exc:
            log.debug("abriendo a %d Hz falló (%s); uso %d", device_sr, exc, cfg.sample_rate)
            try:
                stream = _open(cfg.sample_rate)
                actual_sr = cfg.sample_rate
            except Exception as exc2:
                raise MicrophoneError(f"no se pudo abrir el micrófono: {exc2}") from exc2
        except Exception as exc:
            raise MicrophoneError(f"no se pudo abrir el micrófono: {exc}") from exc

        with stream:
            # 0. Warm-up: descartamos el primer segundo de captura. Algunos
            #    dispositivos USB (Blue Snowball entre ellos) arrancan mudos
            #    y luego sueltan un transitorio fuerte cuando se "despiertan";
            #    si calibramos contra ese transitorio, el umbral queda
            #    inflado y la voz real queda por debajo.
            warmup = 1.0
            warmup_elapsed = 0.0
            while warmup_elapsed < warmup:
                try:
                    blocks.get(timeout=1.0)
                except queue.Empty:
                    break
                warmup_elapsed += block_secs

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
                # 0.005 evita disparar con picos espurios del USB. Si tu
                # ambiente es muy ruidoso, sube JARVIS_SILENCE_THRESHOLD.
                threshold = max(floor * 3.0, 0.005)
            log.debug(
                "umbral de voz: %.5f (floor=%.5f, max=%.5f)",
                threshold, floor, max(noise) if noise else 0.0,
            )

            # 2. Esperamos a que empiece a hablar.
            waited = 0.0
            while waited < cfg.start_timeout:
                try:
                    block = blocks.get(timeout=1.0)
                except queue.Empty:
                    continue
                waited += block_secs
                rms = _rms(block)
                if rms >= threshold:
                    log.debug("voz detectada (rms=%.5f tras %.2fs)", rms, waited)
                    speech.append(block)
                    break
            else:
                log.debug("start_timeout sin voz (umbral=%.5f)", threshold)
                return None
            if not speech:
                return None

            # 3. Grabamos hasta que el silencio se sostenga.
            #    Usamos histéresis: una vez detectado un pico claramente de
            #    voz (rms > 5x el umbral), mantenemos ese nivel como nuevo
            #    piso para no cerrar al medio de la frase cuando RMS baja
            #    entre palabras.
            collected.extend(speech)
            silence = 0.0
            total = block_secs * len(collected)
            active_threshold = threshold
            while silence < cfg.silence_duration and total < cfg.max_utterance:
                try:
                    block = blocks.get(timeout=1.0)
                except queue.Empty:
                    break
                collected.append(block)
                total += block_secs
                rms = _rms(block)
                if rms >= threshold * 5:
                    # Voz clara: subimos el piso activo para que el RMS de
                    # las pausas entre palabras (que cae al ruido de fondo)
                    # no acumule silencio.
                    active_threshold = max(active_threshold, rms * 0.4)
                    silence = 0.0
                elif rms >= active_threshold:
                    silence = 0.0
                else:
                    silence += block_secs

        if not collected:
            return None
        audio = np.concatenate(collected)
        # Si el device abrió a un SR distinto al que Whisper espera, remuestreamos.
        # ratio < 1 cuando bajamos (44.1k → 16k); new_len es el número de
        # muestras destino, que debe ser MENOR que el original.
        if actual_sr != cfg.sample_rate:
            log.debug("remuestreando de %d Hz a %d Hz", actual_sr, cfg.sample_rate)
            ratio = cfg.sample_rate / actual_sr
            new_len = int(round(len(audio) * ratio))
            if new_len > 0:
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, new_len),
                    np.arange(len(audio)),
                    audio,
                ).astype(np.float32)
        return audio, cfg.sample_rate

    async def record(self) -> tuple[np.ndarray, int] | None:
        """Graba sin bloquear el event loop. Devuelve (samples, sample_rate)."""
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
