#!/bin/bash
# Script de execução sequencial — Dia 1
# Roda cada etapa em ordem e para se alguma falhar.

set -e  # Para na primeira falha

echo "🚀 INICIANDO DIAGNÓSTICO SEQUENCIAL — AI WORLD ENGINE"
echo "======================================================="

echo ""
echo ">>> ETAPA 0: Ambiente"
python tests/test_env.py || { echo "❌ Etapa 0 falhou. Corrija o ambiente."; exit 1; }

echo ""
echo ">>> ETAPA 1: FLUX (Geração de Imagem)"
python tests/test_flux.py || { echo "❌ Etapa 1 falhou. FLUX não funcionou."; exit 1; }

echo ""
echo ">>> ETAPA 2: SAM 3 (Segmentação)"
python tests/test_sam3.py || { echo "❌ Etapa 2 falhou. SAM 3 não funcionou."; exit 1; }

echo ""
echo ">>> ETAPA 3: Trellis 2 (GLB)"
python tests/test_trellis.py || { echo "❌ Etapa 3 falhou. Trellis não funcionou."; exit 1; }

echo ""
echo ">>> ETAPA 4: Depth Anything (Profundidade)"
python tests/test_depth.py || { echo "❌ Etapa 4 falhou. Depth Anything não funcionou."; exit 1; }

echo ""
echo "======================================================="
echo "🏆 TODAS AS ETAPAS PASSARAM!"
echo "   tests/output_flux.png   → imagem gerada"
echo "   tests/output_trellis.glb → PRIMEIRO GLB REAL"
echo "   tests/output_depth.png  → mapa de profundidade"
echo ""
echo "Próximo passo: python tests/test_pipeline_minimal.py"
echo "======================================================="
