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
git clone https://github.com/<tu-usuario>/Jarvis.git
cd Jarvis

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si PowerShell se queja de la activación:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Es un one-shot; solo afecta a tu usuario.

## 3. STT (whisper.cpp)

`whisper-cpp` no es un paquete pip. Tienes dos opciones:

### Opción A: binarios precompilados (recomendado)

1. Ve a la página de releases:
   [github.com/ggml-org/whisper.cpp/releases](https://github.com/ggml-org/whisper.cpp/releases).
2. Descarga el ZIP marcado como `whisper-bin-x64.zip` (o `-x64-avx2.zip`
   si tu CPU lo soporta — CPUs desde 2014 suelen tener AVX2).
3. Descomprímelo donde quieras, por ejemplo `C:\tools\whisper\`.
4. Verifica que `whisper-cli.exe` está accesible:

   ```powershell
   & "C:\tools\whisper\whisper-cli.exe" --help
   ```

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

```powershell
copy .env.example .env
notepad .env
```

Edita **solo** estas tres líneas:

```
MINIMAX_API_KEY=<tu-token-de-MiniMax>
JARVIS_WHISPER_BIN=C:\tools\whisper\whisper-cli.exe
JARVIS_WHISPER_MODEL=C:\Users\<tu-usuario>\Documents\Jarvis\models\ggml-base.bin
```

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
