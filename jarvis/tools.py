"""Tools locales: el vocabulario de acciones que el modelo puede invocar.

Las definiciones siguen el esquema de Anthropic (input_schema JSON-Schema) y
las implementaciones devuelven un string listo para que el modelo lo lea.
Cualquier excepción se reporta como texto en vez de propagarse, para que
el modelo pueda corregirse.
"""

from __future__ import annotations

import datetime
import logging
import os
import platform
import shlex
import subprocess
from typing import Any

from .llm import ToolDef, ToolCall

log = logging.getLogger(__name__)

ALLOWED_SHELL_COMMANDS: set[str] = {
    "date", "uname", "whoami", "pwd", "ls", "df -h", "uptime", "echo",
    "cal", "free -h", "nproc", "ip a", "ip route", "ss -tulpn",
}

# Por defecto NO permitimos abrir apps hasta que el usuario lo apruebe
# (el propio README lo recomienda; abrir apps significa control real).
ALLOW_DANGEROUS = os.environ.get("JARVIS_ALLOW_DANGEROUS", "0") not in {"0", "", "false", "no"}


SYSTEM_PROMPT = """Eres Jarvis, un asistente de voz local que controla este equipo.
Respondes SIEMPRE en español y en frases cortas (≤ 25 palabras) porque te
oyen por unos altavoces, no te leen.

Tienes exactamente cuatro herramientas: get_time, get_system_info,
shell_command, open_app. Antes de responder, mira si alguna resuelve la
pregunta; si la hay, úsala. Si no la hay, responde con lo que ya sabes.

Reglas duras:
- NO inventes resultados de tools. Si una tool devuelve un error o un
  aviso, dilo tal cual, sin adornarlo.
- NO confundas "esta tool concreta no está habilitada" con "no puedo
  contestar". Si shell_command está bloqueada pero get_system_info puede
  responder, usa get_system_info.
- Para "qué sistema operativo tengo", "datos del PC", "características
  del equipo" o "información del sistema", usa get_system_info.
- Para "qué hora es" o "qué día es", usa get_time.
- Para "abre X" o "ejecuta Y", usa open_app o shell_command según toque.
- Cuando el usuario diga "adiós" o "termina", no llames a tools: despídete.
"""


def _format_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def get_time(_call: ToolCall) -> str:
    return datetime.datetime.now().strftime("%A %d de %B de %Y, %H:%M:%S")


def get_system_info(_call: ToolCall) -> str:
    return (
        f"SO={platform.system()} {platform.release()}, "
        f"arquitectura={platform.machine()}, "
        f"python={platform.python_version()}"
    )


def shell_command(call: ToolCall) -> str:
    if not ALLOW_DANGEROUS:
        return (
            "shell_command no está habilitada en esta sesión. "
            "Para información del sistema puedes usar get_system_info o get_time."
        )
    cmd = call.input.get("cmd", "").strip()
    if not cmd:
        return "falta el parámetro 'cmd'"
    if cmd not in ALLOWED_SHELL_COMMANDS:
        return f"comando no permitido: {cmd!r}. Permitidos: {sorted(ALLOWED_SHELL_COMMANDS)}"
    try:
        result = subprocess.run(
            shlex.split(cmd), capture_output=True, text=True, timeout=5
        )
    except subprocess.TimeoutExpired:
        return "el comando tardó demasiado"
    except FileNotFoundError:
        return f"no se encontró el ejecutable: {shlex.split(cmd)[0]}"
    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    if result.returncode != 0:
        return f"exit {result.returncode}: {err or out}"
    return out or "(sin salida)"


def open_app(call: ToolCall) -> str:
    if not ALLOW_DANGEROUS:
        return (
            "open_app no está habilitada en esta sesión. "
            "Si quieres información del equipo, usa get_system_info."
        )
    name = call.input.get("name", "").strip()
    if not name:
        return "falta el parámetro 'name'"
    system = platform.system()
    if system == "Linux":
        try:
            subprocess.Popen(
                ["gtk-launch", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return f"ok, lanzando {name}"
        except FileNotFoundError:
            return f"no existe un .desktop llamado {name!r}"
        except OSError as exc:
            return f"no se pudo lanzar {name!r}: {exc}"
    if system == "Windows":
        # En Windows, `name` se interpreta como el nombre del ejecutable
        # (sin .exe) que esté en PATH. Para abrir archivos .lnk, .url o
        # URIs del shell, usa la tool shell_command con `start`.
        cmd = name if name.lower().endswith(".exe") else f"{name}.exe"
        try:
            subprocess.Popen(
                [cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return f"ok, lanzando {name}"
        except FileNotFoundError:
            return f"no se encontró {cmd!r} en PATH"
        except OSError as exc:
            return f"no se pudo lanzar {name!r}: {exc}"
    return f"open_app no está implementado en {system}"


def get_tool_defs() -> list[ToolDef]:
    return [
        ToolDef(
            name="get_time",
            description="Devuelve la fecha y hora actual del sistema.",
            input_schema={"type": "object", "properties": {}, "required": []},
        ),
        ToolDef(
            name="get_system_info",
            description=(
                "Devuelve datos del sistema: sistema operativo, versión del kernel, "
                "arquitectura y versión de Python. Úsala cuando el usuario pregunte "
                "por el sistema operativo, datos del PC, características del equipo, "
                "información del sistema o plataforma."
            ),
            input_schema={"type": "object", "properties": {}, "required": []},
        ),
        ToolDef(
            name="shell_command",
            description=(
                "Ejecuta un comando de shell de solo-lectura de una whitelist fija "
                "(date, uname, whoami, pwd, ls, df, uptime, free, nproc, ip, ss, cal, echo). "
                "Devuelve la salida cruda. NO la uses para preguntas sobre el sistema: "
                "para eso está get_system_info. Úsala solo si el usuario pide explícitamente "
                "ejecutar un comando concreto."
            ),
            input_schema={
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        ),
        ToolDef(
            name="open_app",
            description=(
                "Abre una aplicación por nombre. En Linux usa gtk-launch (.desktop). "
                "En Windows usa el nombre del ejecutable presente en PATH (con o sin .exe). "
                "Úsala solo cuando el usuario pida explícitamente abrir una aplicación. "
                "NO la uses para mostrar información del sistema: para eso está get_system_info."
            ),
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        ),
    ]


# Mapa: nombre de la tool -> función (síncrona, recibe ToolCall -> str)
SYNC_EXECUTORS: dict[str, Any] = {
    "get_time": get_time,
    "get_system_info": get_system_info,
    "shell_command": shell_command,
    "open_app": open_app,
}
