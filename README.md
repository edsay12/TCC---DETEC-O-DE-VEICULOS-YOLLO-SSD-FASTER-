# 🚗 Sistema de Monitoramento de Tráfego com Detecção de Veículos e Reconhecimento de Placas

> **Trabalho de Conclusão de Curso (TCC)**  
> Análise Comparativa de Modelos de Deep Learning para Detecção de Veículos e Reconhecimento Automático de Placas (ALPR) em Cenários de Monitoramento Rodoviário

---

## 📋 Resumo

Este projeto implementa um sistema completo de monitoramento de tráfego rodoviário, integrando:
1. **Detecção de Veículos** — Comparação entre YOLOv8, SSD e Faster R-CNN
2. **Reconhecimento de Placas (ALPR)** — Detecção de placas via YOLOv8 customizado + OCR com EasyOCR
3. **Pipeline Ponta a Ponta** — Processamento de vídeos em tempo real com anotação visual

Os modelos são avaliados em dois benchmarks públicos: **UA-DETRAC** (detecção de veículos) e **UFPR-ALPR** (reconhecimento de placas brasileiras).

---

## 🏗️ Arquitetura do Sistema

```
┌────────────────────────────────────────────────────┐
│                  Frame de Vídeo                    │
└──────────────────────┬─────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │   Detecção de Veículos   │
        │  YOLOv8 / SSD / FRCNN   │
        └────────────┬─────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │   Recorte do Veículo     │
        │      (Vehicle Crop)      │
        └────────────┬─────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │   Detecção de Placa      │
        │   YOLOv8 Customizado     │
        └────────────┬─────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │     OCR (EasyOCR)        │
        │  + Correção Posicional   │
        └────────────┬─────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │    Texto da Placa        │
        │    ex: "ABC1D23"         │
        └──────────────────────────┘
```

---

## 📁 Estrutura do Projeto

```
TCC1/
├── README.md                              # Este arquivo
├── requirements.txt                       # Dependências Python
│
├── notebooks/                             # Experimentos (Jupyter / Colab)
│   ├── 01_Deteccao_Veiculos_UA.ipynb      # Exp. 1: Benchmark UA-DETRAC
│   ├── 02_Reconhecimento_Placas_UFPR.ipynb# Exp. 2: Benchmark UFPR-ALPR
│   ├── 03_Pipeline_Final_Video.ipynb      # Exp. 3: Pipeline de vídeo
│   └── 04_Treinamento_YOLO_Placas.ipynb   # Treinamento YOLOv8 para placas
│
├── scripts/                               # Scripts executáveis
│   ├── organizar_datasets.py              # Pré-processamento de imagens
│   └── run_alpr_benchmark.py              # Benchmark ALPR via linha de comando
│
├── src/                                   # Código-fonte do sistema
│   ├── models/
│   │   └── detectors.py                   # YOLOv8, SSD, Faster R-CNN, PlateDetector
│   ├── ocr/
│   │   └── ocr_engine.py                  # EasyOCR + Tesseract com correção posicional
│   └── utils/
│       ├── evaluator.py                   # Métricas ALPR (IoU, CER, Exact Match)
│       ├── metrics.py                     # Métricas detecção (matching, AP)
│       ├── ua_detrac_runner.py            # Runner benchmark UA-DETRAC
│       └── ufpr_parser.py                 # Parser anotações UFPR-ALPR
│
├── models/                                # Pesos dos modelos (não versionado)
│   ├── yolov8n.pt                         # YOLOv8 Nano (COCO pré-treinado)
│   └── yolov8n-plate.pt                   # YOLOv8 Nano (treinado para placas)
│
└── data/                                  # Datasets (não versionado)
    ├── UA-DETRAC/                         # Dataset de detecção de veículos
    ├── UFPR-ALPR/                         # Dataset de placas brasileiras
    └── processado/                        # Imagens pré-processadas + resultados
```

---

## 🚀 Instalação

### Pré-requisitos
- Python 3.10+
- GPU NVIDIA com CUDA (recomendado, não obrigatório)

### Setup

```bash
# 1. Clonar o repositório
git clone https://github.com/edsay12/TCC---DETEC-O-DE-VEICULOS-YOLLO-SSD-FASTER-.git
cd TCC---DETEC-O-DE-VEICULOS-YOLLO-SSD-FASTER-

# 2. Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Baixar datasets (ver seção Datasets)
```

---

## 🧪 Experimentos

### TCC Experimento 1: Detecção de Veículos (UA-DETRAC)
**Notebook:** `notebooks/01_Deteccao_Veiculos_UA.ipynb`

**Objetivo**: Avaliar a resiliência e performance de YOLOv8, SSD e Faster R-CNN em diferentes cenários simulados.

---
**Métricas**: Precision, Recall, F1, mAP, FPS.

### Experimento 2 — Reconhecimento de Placas (UFPR-ALPR)
**Notebook:** `notebooks/02_Reconhecimento_Placas_UFPR.ipynb`

Avalia o pipeline completo de ALPR (detecção de veículo → detecção de placa → OCR) no dataset UFPR-ALPR.

**Métricas:** IoU, CER (Character Error Rate), Exact Match por frame e por track (votação majoritária)

### Experimento 3 — Pipeline de Vídeo
**Notebook:** `notebooks/03_Pipeline_Final_Video.ipynb`

Demonstração do sistema integrado processando vídeos de câmeras rodoviárias em tempo real.

### Treinamento — YOLOv8 para Placas
**Notebook:** `notebooks/04_Treinamento_YOLO_Placas.ipynb`

Fine-tuning do YOLOv8 Nano para detecção de placas brasileiras usando o UFPR-ALPR.

---

## 📊 Datasets

| Dataset | Descrição | Link |
|---------|-----------|------|
| **UA-DETRAC** | 140k frames de câmeras de tráfego com anotações de veículos | [Página oficial](https://detrac-db.rit.albany.edu/) |
| **UFPR-ALPR** | 4.500 imagens de veículos com placas brasileiras anotadas | [GitHub](https://github.com/raysonlaroca/ufpr-alpr-dataset) |

---

## 🛠️ Tecnologias

| Componente | Tecnologia |
|------------|-----------|
| Detecção de objetos | YOLOv8 (Ultralytics), Faster R-CNN, SSD (Torchvision) |
| OCR | EasyOCR, Tesseract |
| Visão computacional | OpenCV |
| Deep Learning | PyTorch |
| Análise de dados | Pandas, NumPy, Matplotlib, Seaborn |

---

## 📝 Licença

Projeto acadêmico desenvolvido como Trabalho de Conclusão de Curso.
