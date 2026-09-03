# Descarga un modelo ggml de Whisper para STT local.
#
#   .\scripts\fetch-model.ps1 -Model base
#
# 'base' es el equilibrio razonable en CPU sin GPU. 'small' transcribe
# mejor el español pero tarda más.
[CmdletBinding()]
param(
    [ValidateSet("tiny", "base", "small", "medium")]
    [string]$Model = "base"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Dir = Join-Path $Root "models"
$File = "ggml-$Model.bin"
$Url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$File"

if (-not (Test-Path $Dir)) { New-Item -ItemType Directory -Path $Dir | Out-Null }

$Dest = Join-Path $Dir $File
if (Test-Path $Dest) {
    Write-Host "ya existe: $Dest"
}
else {
    Write-Host "descargando $File..."
    Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
}

Write-Host ""
Write-Host "Modelo en: $Dest"
Write-Host "Añade esta línea a tu .env:"
Write-Host "  JARVIS_WHISPER_MODEL=$Dest"
