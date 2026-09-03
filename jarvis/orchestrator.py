"""Orquestador: encadena escuchar → transcribir → ejecutar → hablar.

El punto delicado es el paso 3: una acción de OpenClaw puede tardar segundos.
Para que no se perciba un congelamiento, la ejecución corre como tarea de
asyncio mientras el orquestador reproduce un acuse de recibo y, si hace falta,
fillers. El audio de los fillers se pre-sintetiza al arrancar, de modo que
suenen al instante y no cuesten una llamada a la API en caliente.
"""

from __future__ import annotations

import asyncio
import logging
import random

from .audio import Player, Recorder
from .config import Config
from .executor import Executor
from .stt import WhisperCppTranscriber
from .tts import MiniMaxTTS

log = logging.getLogger(__name__)

ACKS = ["Entendido, voy con eso.", "Perfecto, dame un momento.", "Vale, lo hago."]
FILLERS = ["Sigo en ello.", "Un momento más.", "Casi lo tengo."]

# Tras el acuse de recibo, cada cuántos segundos insistimos con un filler.
FILLER_INTERVAL = 7.0


class Assistant:
    def __init__(self, cfg: Config, executor: Executor) -> None:
        self.cfg = cfg
        self.executor = executor
        self.recorder = Recorder(cfg.audio)
        self.transcriber = WhisperCppTranscriber(cfg.stt)
        self.player = Player()
        self.tts: MiniMaxTTS | None = None
        self._clips: dict[str, bytes] = {}

    # ---------- utilidades de voz ----------

    async def speak(self, text: str) -> None:
        if not self.tts:
            log.warning("TTS no inicializado; solo texto: %s", text)
            return
        try:
            wav = await self.tts.synthesize(text)
        except Exception as exc:
            log.error("falló la síntesis (%s); texto: %s", exc, text)
            return
        await self.player.play_wav(wav)

    async def _prime_clips(self) -> None:
        """Pre-sintetiza acuses y fillers para que suenen sin latencia."""
        if not self.tts:
            return
        for phrase in [*ACKS, *FILLERS]:
            try:
                self._clips[phrase] = await self.tts.synthesize(phrase)
            except Exception as exc:
                log.debug("no se pudo pre-sintetizar %r: %s", phrase, exc)

    async def _play_clip(self, phrase: str) -> None:
        clip = self._clips.get(phrase)
        if clip:
            await self.player.play_wav(clip)
        else:
            await self.speak(phrase)

    # ---------- el ciclo ----------

    async def handle_turn(self) -> bool:
        """Un turno completo. Devuelve False si el usuario pidió terminar."""
        print("\n🎙️  Escuchando…", flush=True)
        recorded = await self.recorder.record()
        if recorded is None:
            print("   (no escuché nada)", flush=True)
            return True
        samples, sample_rate = recorded

        print("📝 Transcribiendo…", flush=True)
        try:
            text = await self.transcriber.transcribe(samples, sample_rate)
        except Exception as exc:
            log.error("falló la transcripción: %s", exc)
            await self.speak("No pude entender el audio.")
            return True

        if not text:
            print("   (silencio)", flush=True)
            return True
        print(f"👤 {text}", flush=True)
        print(f"👤 {text}", flush=True)

        if self._is_exit(text):
            await self.speak("Hasta luego.")
            return False

        # La ejecución arranca ya; el acuse de recibo suena en paralelo.
        task = asyncio.create_task(self.executor.run(text))
        await self._play_clip(random.choice(ACKS))

        # Mientras la tarea siga viva, mantenemos la conversación con fillers.
        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=FILLER_INTERVAL)
            except asyncio.TimeoutError:
                if not task.done():
                    await self._play_clip(random.choice(FILLERS))

        result = await task
        if result.detail:
            log.info("detalle: %s", result.detail)
        print(f"🤖 {result.reply}", flush=True)
        await self.speak(result.reply)
        return True

    @staticmethod
    def _is_exit(text: str) -> bool:
        lowered = text.lower().strip(" .!¡?¿")
        return any(k in lowered for k in ("adiós", "adios", "hasta luego", "termina", "salir"))

    async def run(self) -> None:
        async with MiniMaxTTS(self.cfg.tts) as tts:
            self.tts = tts
            print("🔊 Preparando voz…", flush=True)
            await self._prime_clips()
            await self.speak(
                f"Asistente listo, en modo {self.executor.name}."
                if self.executor.name == "dry-run"
                else f"Asistente listo. Ejecutando con {self.executor.name}."
            )
            print("\nListo. Habla cuando quieras (Ctrl+C para salir).", flush=True)
            try:
                while await self.handle_turn():
                    pass
            except (KeyboardInterrupt, asyncio.CancelledError):
                print("\nInterrumpido.", flush=True)
            finally:
                Player.stop()
