# Port a Windows

Guía para ejecutar Jarvis en Windows 10/11. El código está escrito contra
**PortAudio** (vía `sounddevice`) y la **stdlib** de Python, así que el
núcleo no necesita cambios. Hay tres cosas a instalar aparte de Python:
el propio STT (`whisper.cpp`), el modelo y un editor con el que abrir
los archivos de configuración (cualquiera sirve: VS Code, Notepad++,
incluso Notepad).

## 1. Prerrequisitos

- **Windows 10 1809+ o Windows 11** (cualquier edición sirve).
- **Python 3.11, 3.12 o 3.13** (3.14 también funciona; el dev probó en él).
  Desde [python.org](https://www.python.org/downloads/windows/). En el
  instalador, marca **"Add Python to PATH"** — si no lo haces, el resto
  falla en silencio.
- **PowerShell 5+** (ya viene con Windows 10/11). Úsalo como shell para
  los comandos de esta guía; CMD también funciona pero los ejemplos asumen
  PowerShell.
- **Git para Windows** (opcional, pero recomendado para clonar el repo y
  seguir recibiendo cambios).
- **Acceso a internet** para descargar el binario de `whisper.cpp`, el
  modelo (~140 MB) y el `pip install` de las dependencias.

## 2. Clonar y crear el entorno

```powershell
cd $HOME\Documents
git clone https://github.com/ELTHORTILLA/Jarvis.git
cd Jarvis

# Si el instalador de Python marcó "Add Python to PATH" usas `python` directamente.
# Si marcó también "Install py launcher", puedes usar `py -3.12` o `py -3.13` para
# elegir versión. Lo más portable es:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Si `python` no se reconoce**, abre el instalador de Python otra vez y
> marca "Add Python to PATH" (modificar instalación → modificar). Después
> cierra y abre PowerShell.

Si PowerShell se queja de la activación:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Es un one-shot; solo afecta a tu usuario.

## 3. STT (whisper.cpp)

`whisper-cpp` no es un paquete pip. Tienes dos opciones:

### Opción A: binarios precompilados (recomendado)

Pega cada línea tal cual en PowerShell. **No las partas en dos** — PowerShell
se atraganta con comandos multilínea cuando se pegan desde un chat.

```powershell
$zip = "$env:TEMP\whisper.zip"
Invoke-WebRequest -Uri "https://github.com/ggml-org/whisper.cpp/releases/download/b4938/whisper-bin-x64.zip" -OutFile $zip
Expand-Archive -Path $zip -DestinationPath "C:\tools\whisper" -Force
& "C:\tools\whisper\whisper-cli.exe" --help | Select-Object -First 5
```

Si la URL `b4938` no funciona (porque suben una release más nueva), ve a
[github.com/ggml-org/whisper.cpp/releases](https://github.com/ggml-org/whisper.cpp/releases),
copia la URL del asset `whisper-bin-x64.zip` y reemplaza la URL en el comando.

### Opción B: compilar

Solo si la opción A no funciona en tu CPU. Necesitas Visual Studio
(Build Tools) + CMake. La guía oficial está en
[github.com/ggml-org/whisper.cpp#windows](https://github.com/ggml-org/whisper.cpp#windows).

### Modelo

Independiente del binario:

```powershell
.\scripts\fetch-model.ps1 base
```

> **PowerShell no ejecuta `.sh` por defecto.** Convertí el script o
> descárgalo a mano desde
> [huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin)
> y colócalo en `models\ggml-base.bin`. La ruta absoluta va al `.env`.

El modelo `base` (~140 MB) es el equilibrio razonable en CPU sin GPU
dedicada. Si tienes GPU NVIDIA, `small` rinde mejor con la misma
latencia.

## 4. Configurar `.env`

Si prefieres no tocar Notepad, este bloque PowerShell escribe el `.env`
directamente. Pégalo entero, te pedirá el token, y listo:

```powershell
$envFile = "D:\GitHub\Jarvis\.env"
$token = Read-Host "Pega tu token de MiniMax (sk-cp-...)"
$lines = @(
    "# --- MiniMax (TTS) ---"
    "MINIMAX_API_KEY=`"$token`""
    "MINIMAX_BASE_URL=https://api.minimax.io"
    "MINIMAX_TTS_MODEL=speech-2.6-turbo"
    "MINIMAX_VOICE_ID=Spanish_ConfidentWoman"
    "MINIMAX_TTS_SAMPLE_RATE=24000"
    "MINIMAX_TTS_SPEED=1.0"
    ""
    "# --- STT (whisper.cpp local) ---"
    "JARVIS_WHISPER_BIN=`"C:/tools/whisper/Release/whisper-cli.exe`""
    "JARVIS_WHISPER_STT_LANGUAGE=es"
    "JARVIS_WHISPER_MODEL=`"D:/GitHub/Jarvis/models/ggml-base.bin`""
    ""
    "# --- Captura de audio ---"
    "JARVIS_SILENCE_THRESHOLD=0"
    "JARVIS_SILENCE_DURATION=0.9"
    "JARVIS_MAX_UTTERANCE=20"
    "JARVIS_START_TIMEOUT=10"
    ""
    "# --- Ejecución ---"
    "JARVIS_BACKEND=llm"
    "JARVIS_ALLOW_DANGEROUS=0"
    "JARVIS_OPENCLAW_BIN=openclaw"
    "JARVIS_OPENCLAW_CWD="
    "JARVIS_OPENCLAW_MODEL="
    "JARVIS_OPENCLAW_TIMEOUT=180"
)
$lines -join "`n" | Set-Content -Path $envFile -Encoding UTF8
Write-Host "escrito: $envFile"
```

Si prefieres editar a mano con Notepad:

```powershell
copy .env.example .env
notepad .env
```

Edita **solo** estas tres líneas:

```
MINIMAX_API_KEY=<tu-token-de-MiniMax>
JARVIS_WHISPER_BIN="C:/tools/whisper/Release/whisper-cli.exe"
JARVIS_WHISPER_MODEL="D:/GitHub/Jarvis/models/ggml-base.bin"
```

> **Usa barras normales (`/`) y comillas dobles.** Si pones barras
> invertidas (`\`) sin comillas, Notepad puede guardar `\t` como un
> tabulador, comiéndose letras, y `python-dotenv` interpreta el `=` de
> la primera línea como parte del valor. Las comillas y las barras
> normales eliminan ese problema.

> Las releases recientes de whisper.cpp meten un subdirectorio `Release\`
> dentro del ZIP. Si tu `.exe` aparece en otra ruta tras extraer, ajusta
> `JARVIS_WHISPER_BIN` a la ruta real (`Get-ChildItem -Path "C:\tools\whisper" -Recurse -Filter "whisper-cli.exe"` te la dice).

El resto déjalo como está.

## 5. Comprobar

```powershell
python -m jarvis doctor
```

Deberías ver cuatro `✓`. Si falta alguno, el doctor dice exactamente qué
revisar.

## 6. Primera ejecución

```powershell
python -m jarvis run
```

Di "qué hora es" para probar el bucle completo. Si la transcripción sale
rara (palabras sin sentido), probablemente es ruido ambiente: ajusta
`JARVIS_SILENCE_THRESHOLD` (0.02–0.05 suele funcionar bien) o acércate
más al micrófono.

## Diferencias clave con Linux

| Pieza | Linux | Windows | Notas |
|---|---|---|---|
| Shell | bash | PowerShell o CMD | Activa el venv con `.\.venv\Scripts\Activate.ps1` |
| Path de venv | `.venv/bin/python` | `.venv\Scripts\python.exe` | Todo el código usa `python -m jarvis` que funciona en ambos |
| `whisper-cli` | paquete de sistema | binario externo | Apunta `JARVIS_WHISPER_BIN` al `.exe` |
| `open_app` | `gtk-launch` (`.desktop`) | `subprocess.Popen` con `.exe` | El refactor ya cubre ambos |
| `shell_command` | comandos Unix | comandos Windows | La whitelist es Unix — **antes de activar `JARVIS_ALLOW_DANGEROUS=1` en Windows, reemplázala por comandos de Windows**: `Get-Date`, `Get-ComputerInfo`, `Get-Location`, `Get-ChildItem`, `Get-Process`, `Get-CimInstance Win32_OperatingSystem`. El código detecta el sistema pero la lista de comandos permitidos la tienes que mantener tú. |
| Micrófono | ALSA / PulseAudio / PipeWire | WASAPI | `sounddevice` ya usa WASAPI por defecto en Windows; nada que cambiar |
| Firewall | n/a | primer aviso | La primera vez que llame a la API de MiniMax, Windows preguntará si dejas salir a Python. Acepta para redes privadas. |
| Antivirus | n/a | puede ser paranoico | Si el `.exe` de `whisper-cli` desaparece, añade una exclusión en Defender para `C:\tools\whisper\` |

## Si algo no anda

- **`jarvis doctor` dice "no se encontró el binario"**: comprueba la
  ruta en `JARVIS_WHISPER_BIN` — en Windows las barras invertidas del
  `.env` hay que escaparlas o usar la barra normal: `C:/tools/whisper/...`.
- **El TTS suena metálico o entrecortado**: baja `MINIMAX_TTS_SAMPLE_RATE`
  a `16000` o `8000`. WAV de 24 kHz mono es lo que más le cuesta
  decodificar a algunos DACs.
- **El STT transcribe basura**: ruido de fondo. Sube
  `JARVIS_SILENCE_THRESHOLD` o usa auriculares con micro.
- **PowerShell no activa el venv**: como arriba, `Set-ExecutionPolicy`.
- **Defender borra el binario de whisper**: exclusión en
  *Seguridad de Windows → Protección contra virus → Exclusiones*.

## Próximos pasos cuando ya esté funcionando

- Activar `JARVIS_ALLOW_DANGEROUS=1` con una whitelist de Windows
  adaptada.
- Cambiar el `voice_id` a uno que te guste más — los IDs disponibles
  los lista `python -c "import requests; ..."` o desde la web de MiniMax.
- Sustituir el modelo `base` por `small` si notas que la transcripción
  falla mucho; costará más segundos por frase, pero en hardware moderno
  sigue siendo usable.
