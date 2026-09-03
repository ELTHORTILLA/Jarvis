# Jarvis — asistente de voz local

Escucha por micrófono, transcribe, ejecuta la acción en el sistema y responde
hablando.

```
🎙️  micrófono → whisper.cpp (STT) → MiniMax M3 (LLM + tools) → MiniMax T2A (voz) → 🔊 altavoces
                                          ▲
                          acuses y fillers en paralelo, sin congelar el ciclo
```

## Estado actual

| Capa | Tecnología | Estado |
|---|---|---|
| Captura / reproducción | `sounddevice` (PortAudio) | ✅ verificado |
| STT | whisper.cpp local | ⚠️ requiere instalar binario + modelo |
| Cerebro (LLM) | MiniMax M3 vía `/anthropic/v1/messages` con tools | ✅ verificado end-to-end |
| TTS | MiniMax T2A v2 (WAV) | ✅ verificado end-to-end |
| Orquestación | Python asyncio | ✅ verificado con ejecutor simulado y LLM |

**Arranca con `JARVIS_BACKEND=llm`** (por defecto): MiniMax M3 razona y llama
a las tools locales (get_time, get_system_info, shell_command, open_app).
Las dos últimas requieren `JARVIS_ALLOW_DANGEROUS=1` explícito.

## Dos desviaciones respecto al plan inicial

1. **El STT no es MiniMax.** MiniMax no publica API de ASR — `POST
   /v1/audio/transcriptions` devuelve 404 en `api.minimax.io` y en
   `api.minimaxi.chat`. La transcripción se hace en local con whisper.cpp:
   gratis, offline y portable a Windows. El TTS sí es MiniMax.
2. **`sounddevice` en vez de `pyaudio`.** Misma PortAudio por debajo, pero con
   ruedas precompiladas (en Windows `pyaudio` suele exigir compilador) y una API
   que se integra con asyncio sin bloquear.

Nota: OpenClaw trae un *Talk mode* propio con bucle de voz, pero sus TTS son
elevenlabs/mlx/system y su STT depende de Apple/Android — no cubre Linux con voz
MiniMax, que es lo que este orquestador aporta.

## Puesta en marcha

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

### 1. STT

```bash
sudo pacman -S whisper-cpp          # Arch
./scripts/fetch-model.sh base       # ~148 MB
```

Copia la ruta que imprime el script a `JARVIS_WHISPER_MODEL` en `.env`.

> En este equipo (i7-5500U, sin GPU utilizable) el modelo `base` tarda unos
> segundos por frase. Si se hace pesado, prueba `tiny`; si falla en precisión,
> `small` a cambio de más tiempo.

### 2. TTS

Si `ANTHROPIC_AUTH_TOKEN` ya está en tu entorno, funciona sin tocar nada. Si no,
pon `MINIMAX_API_KEY` en `.env`.

### 3. Comprobar

```bash
.venv/bin/python -m jarvis doctor     # revisa dependencias y credenciales
.venv/bin/python -m jarvis say hola   # prueba solo el TTS
.venv/bin/python -m jarvis listen     # prueba solo el STT
.venv/bin/python -m jarvis devices    # lista dispositivos de audio
.venv/bin/python -m jarvis run        # ciclo completo
```

## Aislamiento de pruebas

Por defecto el asistente **no abre apps ni ejecuta comandos de shell**. Para
activarlos hace falta un opt-in explícito:

1. `JARVIS_BACKEND=dry-run` — valida el ciclo de audio sin razonar con LLM.
2. `JARVIS_BACKEND=llm` (por defecto) — MiniMax M3 razona y llama tools
   inofensivas (get_time, get_system_info).
3. `JARVIS_ALLOW_DANGEROUS=1` — habilita `shell_command` (whitelist) y
   `open_app`. Empieza con órdenes reversibles, no con procesos reales.

Si prefieres delegar el control de UI a OpenClaw, `JARVIS_BACKEND=openclaw`
vuelve al executor antiguo que llama a `openclaw agent exec --json`.

## Ruta a Windows

El código ya evita lo específico de Linux: audio vía PortAudio, WAV decodificado
con la stdlib (sin ffmpeg/mpg123) y rutas con `pathlib`. Para la guía paso a
paso, incluyendo la whitelist de `shell_command` adaptada a PowerShell y los
traspiés típicos de Defender/firewall, mira **[docs/PORT-WINDOWS.md](docs/PORT-WINDOWS.md)**.

Resumen rápido:
- `pip install -r requirements.txt` — `sounddevice` trae PortAudio incluido.
- whisper.cpp: binarios de las releases de `ggml-org/whisper.cpp`. Apunta
  `JARVIS_WHISPER_BIN` a `whisper-cli.exe`.
- `open_app` ya está portada (`gtk-launch` en Linux, `subprocess.Popen` en
  Windows). `shell_command` requiere que adaptes la whitelist a comandos de
  PowerShell antes de activarla.

## Estructura

```
jarvis/
  config.py        configuración desde .env
  audio.py         captura con detección de silencio autocalibrada, reproducción
  stt.py           whisper.cpp por subproceso
  tts.py           MiniMax T2A v2 con reintentos
  executor.py      dry-run | OpenClaw
  orchestrator.py  el ciclo y los fillers
  __main__.py      CLI
scripts/
  fetch-model.sh   descarga del modelo ggml
```
