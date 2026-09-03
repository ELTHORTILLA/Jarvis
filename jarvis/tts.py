"""Síntesis de voz (TTS) con MiniMax T2A v2.

Pedimos WAV en lugar de MP3 a propósito: el módulo `wave` de la stdlib lo
decodifica sin ffmpeg ni mpg123, así que el mismo código funciona en Windows.
"""

from __future__ import annotations

import asyncio
import logging

import aiohttp

from .config import TTSConfig

log = logging.getLogger(__name__)


class TTSError(RuntimeError):
    pass


class MiniMaxTTS:
    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "MiniMaxTTS":
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.cfg.timeout),
            headers={
                "Authorization": f"Bearer {self.cfg.api_key}",
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, *_exc) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    def preflight(self) -> list[str]:
        if not self.cfg.api_key:
            return ["falta MINIMAX_API_KEY (o ANTHROPIC_AUTH_TOKEN) en el entorno/.env"]
        return []

    async def synthesize(self, text: str) -> bytes:
        """Devuelve un WAV con la voz. Reintenta ante errores transitorios."""
        text = text.strip()
        if not text:
            raise TTSError("no hay texto que sintetizar")
        if self._session is None:
            raise TTSError("usar MiniMaxTTS como context manager (async with)")

        payload = {
            "model": self.cfg.model,
            "text": text,
            "stream": False,
            "voice_setting": {
                "voice_id": self.cfg.voice_id,
                "speed": self.cfg.speed,
                "vol": 1,
                "pitch": 0,
            },
            "audio_setting": {
                "sample_rate": self.cfg.sample_rate,
                "format": "wav",
                "channel": 1,
            },
            "output_format": "hex",
        }

        last_error: str = "desconocido"
        for attempt in range(3):
            if attempt:
                await asyncio.sleep(2**attempt)
            try:
                async with self._session.post(self.cfg.endpoint, json=payload) as resp:
                    if resp.status != 200:
                        last_error = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                        log.warning("TTS reintentando: %s", last_error)
                        continue
                    body = await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = f"error de red: {exc}"
                log.warning("TTS reintentando: %s", last_error)
                continue

            # La API responde 200 incluso en fallos lógicos; el código real
            # viene en base_resp.status_code (0 = éxito).
            status = body.get("base_resp", {})
            if status.get("status_code") not in (0, None):
                last_error = f"MiniMax {status.get('status_code')}: {status.get('status_msg')}"
                # 1004 = auth inválida; reintentar no ayuda.
                if status.get("status_code") == 1004:
                    raise TTSError(last_error)
                log.warning("TTS reintentando: %s", last_error)
                continue

            audio_hex = body.get("data", {}).get("audio")
            if not audio_hex:
                last_error = "la respuesta no incluyó audio"
                continue
            try:
                return bytes.fromhex(audio_hex)
            except ValueError as exc:
                raise TTSError(f"audio hex ilegible: {exc}") from exc

        raise TTSError(f"la síntesis falló tras 3 intentos: {last_error}")
