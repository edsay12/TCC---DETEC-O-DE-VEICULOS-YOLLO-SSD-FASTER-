# Roteiro Estratégico: TCC Monitoramento de Trânsito

Para que seu TCC seja robusto e não confuso, vamos dividir o projeto em **dois experimentos distintos**. Não tente misturá-los no mesmo código.

## Experimento 1: Detecção de Veículos em Ambientes Simulados
**Dataset:** UA-DETRAC
**Objetivo:** Comparar qual modelo de detecção é mais resiliente a condições adversas.

### Passos:
1.  **Organização**: Use o script `organizar_datasets.py` para separar as imagens em `diurno`, `noturno` e `baixa_qualidade`. (Já fizemos isso!).
2.  **Benchmark**: Rode o código de benchmark apenas para veículos.
3.  **Métrica**: Foque em **mAP (mean Average Precision)** e **FPS**.
4.  **O que escrever no TCC**: "O modelo YOLOv8 foi X% mais rápido, mas o Faster R-CNN foi mais estável no cenário noturno."

---

## Experimento 2: Reconhecimento Automático de Placas (ALPR)
**Dataset:** UFPR-ALPR
**Objetivo:** Avaliar a precisão do pipeline completo (Detecção + OCR).

### Passos:
1.  **Pipeline**: Use o fluxo de 3 estágios (Veículo -> Placa -> Texto).
2.  **Benchmark**: Rode o `run_alpr_benchmark.py`.
3.  **Métrica**: Foque em **Acurácia do OCR** (quantas letras a IA acertou).
4.  **O que escrever no TCC**: "O uso do EasyOCR em conjunto com o YOLOv8 atingiu uma taxa de sucesso de X%."

---

## O que ajustar IMEDIATAMENTE (Checklist):

1. [x] **Ambiente**: Já instalamos o Pip e criamos o `.venv_linux`.
2. [ ] **Limpeza de Caminhos**: Garanta que nenhum arquivo `.py` ou célula do Notebook tenha `C:/Users/...`. Use sempre `Path.cwd()`.
3. [ ] **Instalação Final**: Execute `!./.venv_linux/bin/python3 -m pip install -r requirements.txt` para garantir que o `ultralytics` e `easyocr` funcionem.
4. [ ] **Separação do Notebook**: Se possível, crie dois Notebooks ou seções bem separadas:
   - `01_Deteccao_Veiculos_UA.ipynb`
   - `02_Reconhecimento_Placas_UFPR.ipynb`

Isso vai deixar sua mente (e a do professor) muito mais tranquila!
