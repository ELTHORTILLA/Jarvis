#!/usr/bin/env bash
# Descarga un modelo ggml de Whisper para STT local.
#
#   ./scripts/fetch-model.sh [base|small|tiny|medium]
#
# 'base' es el equilibrio razonable en CPU sin GPU. 'small' transcribe mejor
# el español pero cuesta bastante más tiempo por frase.
set -euo pipefail

MODEL="${1:-base}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
FILE="ggml-${MODEL}.bin"
URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/${FILE}"

mkdir -p "$DIR"
if [[ -f "$DIR/$FILE" ]]; then
  echo "ya existe: $DIR/$FILE"
else
  echo "descargando $FILE…"
  curl -fL --progress-bar "$URL" -o "$DIR/$FILE.part"
  mv "$DIR/$FILE.part" "$DIR/$FILE"
fi

echo
echo "Modelo en: $DIR/$FILE"
echo "Añade esta línea a tu .env:"
echo "  JARVIS_WHISPER_MODEL=$DIR/$FILE"
