"""Cliente de chat para MiniMax M3 vía su gateway compatible con Anthropic.

Es un subconjunto deliberadamente pequeño del SDK de Anthropic: cubrimos
solo lo que el orquestador necesita — `messages.create` con tools, bucle
multi-turno y cancelación limpia.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import aiohttp

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.minimax.io/anthropic"
DEFAULT_MODEL = "MiniMax-M3.0"
ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class ToolDef:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class TextBlock:
    text: str


@dataclass
class AssistantTurn:
    blocks: list[TextBlock | ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    raw_usage: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.blocks if isinstance(b, TextBlock))

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [b for b in self.blocks if isinstance(b, ToolCall)]


class MiniMaxChat:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        # Si no nos pasan la API key, la buscamos en el entorno. El `.env`
        # lo carga Config; aquí somos agnósticos al origen.
        self.api_key = (
            api_key
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            or os.environ.get("MINIMAX_API_KEY")
            or ""
        )
        self.base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        if self.base_url.endswith("/anthropic"):
            self.endpoint = f"{self.base_url}/v1/messages"
        else:
            self.base_url = DEFAULT_BASE_URL
            self.endpoint = f"{self.base_url}/v1/messages"
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "MiniMaxChat":
        self._session = aiohttp.ClientSession(
            timeout=self.timeout,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
        )
        return self

    async def __aexit__(self, *_exc) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    def preflight(self) -> list[str]:
        if not self.api_key:
            return ["falta ANTHROPIC_AUTH_TOKEN en el entorno/.env"]
        return []

    @staticmethod
    def _parse_block(raw: dict[str, Any]) -> TextBlock | ToolCall:
        kind = raw.get("type")
        if kind == "text":
            return TextBlock(text=raw.get("text", ""))
        if kind == "tool_use":
            return ToolCall(
                id=raw["id"],
                name=raw["name"],
                input=raw.get("input", {}) or {},
            )
        # Bloque desconocido: lo representamos como texto vacío para no perder el turno.
        log.debug("bloque de respuesta ignorado: %s", kind)
        return TextBlock(text="")

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self._session is not None, "usar MiniMaxChat como async with"
        async with self._session.post(self.endpoint, json=payload) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"MiniMax HTTP {resp.status}: {text[:300]}")
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"respuesta no es JSON: {text[:300]}") from exc

    async def create(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDef] | None = None,
        system: str | None = None,
    ) -> AssistantTurn:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]

        log.debug("MiniMax request: %s", json.dumps(payload, ensure_ascii=False)[:400])
        body = await self._post(payload)

        base = body.get("base_resp", {})
        if base.get("status_code") not in (0, None):
            raise RuntimeError(f"MiniMax {base.get('status_code')}: {base.get('status_msg')}")

        turn = AssistantTurn(
            stop_reason=body.get("stop_reason", ""),
            raw_usage=body.get("usage", {}),
        )
        for raw in body.get("content", []):
            turn.blocks.append(self._parse_block(raw))
        return turn

    # ----- bucle de tools -----

    async def run_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDef],
        execute: Any,  # async (ToolCall) -> str
        system: str | None = None,
        max_steps: int = 8,
    ) -> tuple[list[dict[str, Any]], AssistantTurn]:
        """Itera hasta que el modelo deje de pedir tools.

        `execute` recibe un ToolCall y devuelve el string de salida. Los
        errores de la tool se devuelven al modelo como texto (no como
        excepción) para que pueda corregirse a sí mismo.
        """
        history = list(messages)
        last_turn: AssistantTurn | None = None

        for step in range(max_steps):
            turn = await self.create(history, tools=tools, system=system)
            last_turn = turn
            log.debug("paso %d stop_reason=%s tools=%d", step, turn.stop_reason, len(turn.tool_calls))

            if not turn.tool_calls:
                return history, turn

            # Empujamos el turno del asistente y, por cada tool, un tool_result.
            history.append({"role": "assistant", "content": [_block_to_raw(b) for b in turn.blocks]})
            tool_results: list[dict[str, Any]] = []
            for call in turn.tool_calls:
                try:
                    output = await execute(call)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    output = f"error ejecutando {call.name}: {exc!r}"
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": str(output)[:8000],  # el límite razonable para no inflar el contexto
                    }
                )
            history.append({"role": "user", "content": tool_results})

        # Si agotamos los pasos, devolvemos la última respuesta para que el
        # orquestador pueda al menos leer lo que el modelo alcanzó a decir.
        assert last_turn is not None
        return history, last_turn


def _block_to_raw(block: TextBlock | ToolCall) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
