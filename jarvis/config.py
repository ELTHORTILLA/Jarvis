"""Configuración central, cargada desde .env."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache"

load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


@dataclass
class AudioConfig:
    """Parámetros de captura. 16 kHz mono es lo que espera Whisper."""

    sample_rate: int = 16_000
    channels: int = 1
    block_ms: int = 30

    # Detección de silencio. El umbral se autocalibra con el ruido ambiente,
    # pero se puede fijar a mano con JARVIS_SILENCE_THRESHOLD.
    silence_threshold: float = field(default_factory=lambda: _env_float("JARVIS_SILENCE_THRESHOLD", 0.0))
    silence_duration: float = field(default_factory=lambda: _env_float("JARVIS_SILENCE_DURATION", 0.9))
    max_utterance: float = field(default_factory=lambda: _env_float("JARVIS_MAX_UTTERANCE", 20.0))
    start_timeout: float = field(default_factory=lambda: _env_float("JARVIS_START_TIMEOUT", 10.0))
    calibration_time: float = 0.6

    @property
    def block_size(self) -> int:
        return int(self.sample_rate * self.block_ms / 1000)


@dataclass
class STTConfig:
    """whisper.cpp local. `model` apunta a un fichero ggml-*.bin."""

    binary: str = field(default_factory=lambda: _env("JARVIS_WHISPER_BIN", "whisper-cli"))
    model: str = field(default_factory=lambda: _env("JARVIS_WHISPER_MODEL"))
    language: str = field(default_factory=lambda: _env("JARVIS_STT_LANGUAGE", "es"))
    threads: int = field(default_factory=lambda: _env_int("JARVIS_WHISPER_THREADS", max(1, (os.cpu_count() or 2))))

    def resolve_binary(self) -> str | None:
        return shutil.which(self.binary)


@dataclass
class TTSConfig:
    """MiniMax T2A v2. Pedimos WAV para reproducir sin decodificador externo."""

    api_key: str = field(default_factory=lambda: _env("MINIMAX_API_KEY") or _env("ANTHROPIC_AUTH_TOKEN"))
    base_url: str = field(default_factory=lambda: _env("MINIMAX_BASE_URL", "https://api.minimax.io"))
    model: str = field(default_factory=lambda: _env("MINIMAX_TTS_MODEL", "speech-2.6-turbo"))
    voice_id: str = field(default_factory=lambda: _env("MINIMAX_VOICE_ID", "Spanish_ConfidentWoman"))
    sample_rate: int = field(default_factory=lambda: _env_int("MINIMAX_TTS_SAMPLE_RATE", 24_000))
    speed: float = field(default_factory=lambda: _env_float("MINIMAX_TTS_SPEED", 1.0))
    timeout: float = field(default_factory=lambda: _env_float("MINIMAX_TTS_TIMEOUT", 30.0))

    @property
    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/t2a_v2"


@dataclass
class ExecutorConfig:
    """Motor de acciones.

    - `backend` = "llm" (por defecto) usa MiniMax M3 + tools locales.
    - `backend` = "dry-run" simula sin tocar nada.
    - `backend` = "openclaw" delega en `openclaw agent exec --json`.
    """

    backend: str = field(default_factory=lambda: _env("JARVIS_BACKEND", "llm"))
    binary: str = field(default_factory=lambda: _env("JARVIS_OPENCLAW_BIN", "openclaw"))
    cwd: str = field(default_factory=lambda: _env("JARVIS_OPENCLAW_CWD"))
    model: str = field(default_factory=lambda: _env("JARVIS_OPENCLAW_MODEL"))
    timeout: float = field(default_factory=lambda: _env_float("JARVIS_OPENCLAW_TIMEOUT", 180.0))

    def resolve_binary(self) -> str | None:
        return shutil.which(self.binary)


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    cache_dir: Path = CACHE_DIR

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Propagamos al entorno del proceso cualquier secreto del .env que
        # aún no esté allí. Así módulos que leen `os.environ` directamente
        # (MiniMaxChat) ven el token aunque `Config` no se lo pase.
        from dotenv import dotenv_values
        for key, value in dotenv_values(ROOT / ".env").items():
            if value and not os.environ.get(key):
                os.environ[key] = value
