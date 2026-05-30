# Sistema de Monitoramento de Tráfego com Visão Computacional

Este projeto é parte integrante de um Trabalho de Conclusão de Curso (TCC) focado na análise comparativa de modelos de detecção de objetos e reconhecimento de caracteres (ALPR) em diferentes cenários ambientais.

## 🚀 Funcionalidades

- **Detecção de Objetos Multimodelo**: Suporte para YOLO, SSD e Faster R-CNN.
- **ALPR em Cascata**: Detecção de veículos seguida de segmentação e OCR de placas.
- **Análise Ambiental**: Classificação automática de cenários (Diurno, Noturno, Baixa Qualidade).
- **Métricas Científicas**: Coleta de tempo de processamento, FPS, precisão de detecção e taxas de falha.
- **Geração de Relatórios**: Exportação de resultados em CSV e relatórios consolidados em TXT.

## 📂 Estrutura do Projeto

```text
├── src/
│   ├── core/         # Orquestração do pipeline
│   ├── models/       # Wrappers para YOLO, SSD, Faster R-CNN
│   ├── ocr/          # Motores EasyOCR e Tesseract
│   └── utils/        # Logs, métricas e ferramentas de visualização
├── main.py           # Ponto de entrada para experimentos
├── organizar_datasets.py # Pré-processamento e classificação de cenários
├── requirements.txt  # Dependências do sistema
└── Analise_e_Treinamento_TCC.ipynb # Análise exploratória
```

## 🛠️ Instalação

1. Clone o repositório.
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Como Executar os Experimentos

Para comparar os modelos, você pode rodar:

```bash
# Executar com YOLO
python main.py --model yolo --ocr easyocr

# Executar com SSD
python main.py --model ssd --ocr easyocr

# Executar com Faster R-CNN
python main.py --model faster_rcnn --ocr easyocr
```

## 🎓 Metodologia Científica

O sistema foi desenhado para suportar a metodologia experimental, permitindo a extração de dados estatísticos que podem ser diretamente utilizados em tabelas e gráficos comparativos no corpo do TCC.
