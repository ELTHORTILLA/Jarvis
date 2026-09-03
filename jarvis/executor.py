"""Ejecutor: convierte una orden en texto en una respuesta hablada.

Hay tres backends con la misma interfaz:

- DryRunExecutor: entiende y simula. Para validar el ciclo de audio.
- LLMExecutor: MiniMax M3 con tools locales. El modo principal.
- OpenClawExecutor: un turno headless de OpenClaw. Por si en el futuro
  prefieres delegar a su ecosistema de hooks y elevación.

Por defecto, `build_executor` devuelve LLMExecutor. Para simular o delegar:
    JARVIS_DRY_RUN=1   → DryRun
    JARVIS_BACKEND=openclaw → OpenClaw
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from .config import ExecutorConfig
from .llm import MiniMaxChat, ToolCall
from .tools import ALLOW_DANGEROUS, SYNC_EXECUTORS, get_tool_defs

log = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    ok: bool
    reply: str
    detail: str = ""


class Executor(Protocol):
    name: str

    def preflight(self) -> list[str]: ...

    async def run(self, command: str) -> ExecutionResult: ...


class DryRunExecutor:
    name = "dry-run"

    def preflight(self) -> list[str]:
        return []

    async def run(self, command: str) -> ExecutionResult:
        log.info("[dry-run] no se ejecuta: %s", command)
        return ExecutionResult(
            ok=True,
            reply=f"Modo simulación. Entendí: {command}. No ejecuté nada.",
            detail="dry-run: sin efectos",
        )


class LLMExecutor:
    """MiniMax M3 como cerebro, con tools locales.

    Mantiene una conversación de varios turnos en memoria: el modelo ve sus
    propios mensajes anteriores, así que puede encadenar herramientas (p. ej.
    listar apps y luego abrir una). Para mantener el contexto acotado, se
    recorta a las últimas N interacciones.
    """

    name = "llm"

    def __init__(self, cfg: ExecutorConfig, history_limit: int = 6) -> None:
        self.cfg = cfg
        self.history_limit = history_limit
        self._history: list[dict[str, Any]] = []

    def preflight(self) -> list[str]:
        # El LLM se inicializa lazy (async with); preflight mínimo aquí.
        return []

    async def run(self, command: str) -> ExecutionResult:
        try:
            async with MiniMaxChat() as llm:
                if problems := llm.preflight():
                    return ExecutionResult(False, problems[0], "llm preflight")

                from .tools import SYSTEM_PROMPT

                tools = get_tool_defs()

                async def execute(call: ToolCall) -> str:
                    fn = SYNC_EXECUTORS.get(call.name)
                    if not fn:
                        return f"tool {call.name!r} no registrada"
                    log.info("tool call: %s(%s)", call.name, call.input)
                    return await asyncio.to_thread(fn, call)

                # Cada turno parte de la historia anterior más el nuevo user.
                # El historial interno se reduce a un único turno completo
                # (user → assistant con tools → user con tool_results →
                # assistant final) para que MiniMax nunca vea tool_results
                # huérfanos de turnos previos.
                messages = list(self._history) + [{"role": "user", "content": command}]
                history, final = await llm.run_with_tools(
                    messages, tools, execute, system=SYSTEM_PROMPT
                )
                # Conservamos solo el último turno cerrado: desde el user
                # actual hasta el assistant final (incluye tool_results).
                self._history = [m for m in history if m.get("role") != "user" or m.get("content") != command]
                self._trim()

                return ExecutionResult(
                    ok=True,
                    reply=final.text or "(sin respuesta)",
                    detail=f"stop={final.stop_reason} usage={final.raw_usage.get('output_tokens')}tok",
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("falló el LLMExecutor")
            return ExecutionResult(False, f"No pude procesar la orden: {exc}", repr(exc)[:200])

    def _trim(self) -> None:
        # Mantenemos el último turno completo, no más.
        if len(self._history) > self.history_limit:
            self._history = self._history[-self.history_limit:]


class OpenClawExecutor:
    """Un turno headless de OpenClaw por orden.

    Se conserva por compatibilidad: si más adelante prefieres delegar el
    control de UI a OpenClaw, basta con `JARVIS_BACKEND=openclaw`.
    """

    name = "openclaw"

    def __init__(self, cfg: ExecutorConfig) -> None:
        self.cfg = cfg

    def preflight(self) -> list[str]:
        if not self.cfg.resolve_binary():
            return [
                f"no se encontró '{self.cfg.binary}' en PATH "
                "(instalar: curl -fsSL https://openclaw.ai/install.sh | bash)"
            ]
        return []

    async def run(self, command: str) -> ExecutionResult:
        binary = self.cfg.resolve_binary()
        if not binary:
            return ExecutionResult(False, "El motor de ejecución no está instalado.", "falta openclaw")
        args = [binary, "agent", "exec", command, "--json"]
        if self.cfg.cwd:
            args += ["--cwd", self.cfg.cwd]
        if self.cfg.model:
            args += ["--model", self.cfg.model]
        args += ["--timeout", str(int(self.cfg.timeout))]
        try:
            proc = await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        except OSError as exc:
            return ExecutionResult(False, "No pude lanzar el motor de ejecución.", str(exc))
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.cfg.timeout + 30
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ExecutionResult(False, "La tarea tardó demasiado y la cancelé.", "timeout")
        err = stderr.decode("utf-8", "replace").strip()
        raw = stdout.decode("utf-8", "replace").strip()
        if not raw:
            return ExecutionResult(False, "El motor no devolvió respuesta.", err[:400])
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError:
            return ExecutionResult(proc.returncode == 0, raw[:500], "respuesta no era JSON")
        final = (envelope.get("final") or "").strip()
        if envelope.get("ok") and final:
            return ExecutionResult(True, final, f"status={envelope.get('status')}")
        return ExecutionResult(False, final or "La tarea no se completó.", err[:200])


def build_executor(cfg: ExecutorConfig) -> Executor:
    backend = cfg.backend
    if backend == "dry-run":
        return DryRunExecutor()
    if backend == "openclaw":
        return OpenClawExecutor(cfg)
    return LLMExecutor(cfg)
