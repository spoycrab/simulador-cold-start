#!/usr/bin/env bash
# Reproduz todos os resultados do artigo, na ordem: auditoria e calibração dos dados (etapa 1),
# calibração e validação do simulador (etapa 2), experimento, análise e sensibilidade (etapa 3).
# Requer os arquivos do Inside Airbnb em Dados/ (ver Dados/LEIA-ME.txt).
# Uso: ./reproduzir.sh [--processos N]
set -euo pipefail
cd "$(dirname "$0")"
for comando in auditar calibrar calibrar-simulador validar experimento analisar sensibilidade; do
    echo "=== python -m coldstart $comando ==="
    python -m coldstart "$comando" "$@"
done
