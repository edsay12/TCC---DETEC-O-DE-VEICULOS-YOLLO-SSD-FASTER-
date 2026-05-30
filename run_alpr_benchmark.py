"""
Benchmark ALPR usando o dataset UFPR-ALPR com ground truth real.

Uso:
    python run_alpr_benchmark.py
    python run_alpr_benchmark.py --limit 20       # Teste rápido com 20 imagens
    python run_alpr_benchmark.py --split testing  # Split oficial para o TCC
"""

import cv2
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

from src.models.detectors import YOLODetector, TorchvisionDetector, PlateDetector
from src.ocr.ocr_engine import OCREngine
from src.utils.ufpr_parser import load_dataset_split
from src.utils.evaluator import evaluate_detection

# Detecta o diretório base dinamicamente
BASE_DIR      = Path(__file__).parent if "__file__" in locals() else Path.cwd()
DATASET_ROOT  = BASE_DIR / "UFPR-ALPR dataset"
OUTPUT_DIR    = BASE_DIR / "dataset_processado" / "resultados"


def run_benchmark(split: str = 'testing', limit: int = None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  BENCHMARK ALPR — UFPR-ALPR Dataset (Ground Truth Real)")
    print("=" * 60)

    # --- Carrega amostras do split ---
    print(f"\nCarregando split '{split}'...")
    samples = load_dataset_split(DATASET_ROOT, split=split, limit=limit)
    print(f"  Total de amostras carregadas: {len(samples)}")

    # --- Detectores de veículo ---
    models_config = [
        ('YOLOv8',       lambda: YOLODetector('yolov8n.pt')),
        ('SSD',          lambda: TorchvisionDetector(model_type='ssd')),
        ('Faster R-CNN', lambda: TorchvisionDetector(model_type='faster_rcnn')),
    ]

    # Tenta inicializar cada modelo, pula se falhar
    vehicle_detectors = []
    for name, factory in models_config:
        try:
            print(f"  Carregando {name}...", end=' ')
            det = factory()
            vehicle_detectors.append((name, det))
            print("OK")
        except Exception as e:
            print(f"FALHOU ({type(e).__name__}). Pulando.")

    if not vehicle_detectors:
        print("Nenhum detector carregou com sucesso. Abortando.")
        return None

    # --- Detector de placa (único, compartilhado) ---
    plate_detector = PlateDetector()

    # --- OCR ---
    ocr = OCREngine(engine_type='easyocr')

    all_results = []

    for det_name, vehicle_det in vehicle_detectors:
        print(f"\n>>> Testando: {det_name}")

        for sample in tqdm(samples, desc=det_name):
            img_path   = sample['image_path']
            annotation = sample['annotation']
            gt_text    = annotation['plate_text']
            gt_bbox    = annotation['plate_bbox']       # ground truth da placa
            gt_veh     = annotation['vehicle_bbox']     # ground truth do veículo

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h_img, w_img = img.shape[:2]

            # --- Estágio 1: Usa o detector real para encontrar o veículo ---
            vehicle_detections = vehicle_det.detect(img)
            vehicles = [d for d in vehicle_detections
                        if d['class'] in ['car', 'truck', 'bus', 'motorcycle']]

            if vehicles:
                # Pega o maior veículo detectado
                v = max(vehicles, key=lambda d: (d['bbox'][2]-d['bbox'][0]) * (d['bbox'][3]-d['bbox'][1]))
                x1, y1, x2, y2 = map(int, v['bbox'])
                # Clipa para dentro da imagem
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                vehicle_crop = img[y1:y2, x1:x2]
                # Ajusta gt_bbox para o sistema de coordenadas do recorte
                gt_bbox_crop = [
                    gt_bbox[0] - x1, gt_bbox[1] - y1,
                    gt_bbox[2] - x1, gt_bbox[3] - y1,
                ]
            else:
                # Fallback: usa veículo do ground truth se o detector falhar
                x1, y1 = max(0, gt_veh[0]), max(0, gt_veh[1])
                x2, y2 = min(w_img, gt_veh[2]), min(h_img, gt_veh[3])
                vehicle_crop = img[y1:y2, x1:x2]
                gt_bbox_crop = [
                    gt_bbox[0] - x1, gt_bbox[1] - y1,
                    gt_bbox[2] - x1, gt_bbox[3] - y1,
                ]

            if vehicle_crop.size == 0:
                continue

            # --- Estágio 2: Detecta a placa dentro do recorte ---
            plate_detections = plate_detector.detect(vehicle_crop)

            if plate_detections:
                best_plate = max(plate_detections, key=lambda p: p['conf'])
                pred_bbox  = best_plate['bbox']
                px1, py1, px2, py2 = map(int, pred_bbox)
                px1, py1 = max(0, px1), max(0, py1)
                plate_crop = vehicle_crop[py1:py2, px1:px2]
            else:
                # Fallback: usa bbox GT da placa para isolar contribuição do OCR
                pred_bbox  = None
                gx1, gy1, gx2, gy2 = [max(0, int(c)) for c in gt_bbox_crop]
                plate_crop = vehicle_crop[gy1:gy2, gx1:gx2]

            # --- Estágio 3: OCR ---
            pred_text = ''
            if plate_crop is not None and plate_crop.size > 0:
                pred_text = ocr.read_plate(plate_crop) or ''

            # --- Avaliação ---
            metrics = evaluate_detection(pred_bbox, gt_bbox_crop, pred_text, gt_text)

            all_results.append({
                'model':         det_name,
                'track':         img_path.parent.name,   # ex: track0091
                'image':         img_path.name,
                'gt_plate':      gt_text,
                'pred_plate':    pred_text,
                'vehicle_found': len(vehicles) > 0,
                'iou':           round(metrics['iou'], 4),
                'cer':           round(metrics['cer'], 4),
                'exact_match':   metrics['exact_match'],
                'plate_found':   metrics['plate_found'],
            })

    # --- Salva CSV ---
    df = pd.DataFrame(all_results)
    csv_path = OUTPUT_DIR / 'alpr_benchmark.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResultados salvos em: {csv_path}")

    # --- Gera Relatório ---
    _generate_report(df)

    return df


def _generate_report(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  RELATÓRIO FINAL — BENCHMARK UFPR-ALPR")
    print("=" * 60)

    # --- Métricas por frame ---
    summary = df.groupby('model').agg(
        Total_Frames=('image', 'count'),
        IoU_Medio=('iou', 'mean'),
        CER_Medio=('cer', 'mean'),
        Acerto_Frame=('exact_match', 'mean'),
        Taxa_Placa_Encontrada=('plate_found', 'mean'),
    ).round(4)

    # --- Métricas por track (votação por maioria) ---
    # Cada track = 1 veículo com a mesma placa em todos os frames.
    # A leitura OCR mais frequente no track é a "resposta" do sistema.
    def track_accuracy(group):
        """Para cada track, elege a leitura mais votada e compara com GT."""
        def best_vote(g):
            non_empty = g['pred_plate'].dropna()
            non_empty = non_empty[non_empty != '']
            if non_empty.empty:
                voted = ''
            else:
                voted = non_empty.mode().iloc[0]  # leitura mais frequente
            gt = g['gt_plate'].iloc[0]
            from src.utils.evaluator import is_exact_match
            return is_exact_match(voted, gt)

        return group.groupby('track').apply(best_vote).mean()

    track_acc = df.groupby('model').apply(track_accuracy).rename('Acerto_Track_Voto')
    summary = summary.join(track_acc)

    summary['Acerto_Frame']           = (summary['Acerto_Frame'] * 100).round(1).astype(str) + '%'
    summary['Taxa_Placa_Encontrada']  = (summary['Taxa_Placa_Encontrada'] * 100).round(1).astype(str) + '%'
    summary['Acerto_Track_Voto']      = (summary['Acerto_Track_Voto'] * 100).round(1).astype(str) + '%'

    print(summary.to_string())
    print()
    print("Nota: 'Acerto_Track_Voto' usa votacao por maioria entre os frames do mesmo track.")
    print("      Cada track corresponde a um veiculo especifico do dataset UFPR-ALPR.")

    # --- Gráficos ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Benchmark ALPR — UFPR-ALPR Dataset', fontsize=14, fontweight='bold')

    plot_df = df.groupby('model')[['iou', 'cer']].mean().reset_index()

    sns.barplot(data=plot_df, x='model', y='iou', hue='model', ax=axes[0], palette='viridis', legend=False)
    axes[0].set_title('IoU Médio (maior = melhor)')
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('IoU')

    sns.barplot(data=plot_df, x='model', y='cer', hue='model', ax=axes[1], palette='magma', legend=False)
    axes[1].set_title('CER Médio (menor = melhor)')
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('CER (Character Error Rate)')

    # Usa acerto por track (votação) no gráfico — métrica mais justa
    track_vote_df = df.groupby('model').apply(
        lambda g: g.groupby('track').apply(
            lambda t: (
                lambda voted, gt: voted == gt
            )(
                (t['pred_plate'][t['pred_plate'] != ''].mode().iloc[0]
                 if not t['pred_plate'][t['pred_plate'] != ''].empty else ''),
                t['gt_plate'].iloc[0]
            )
        ).mean()
    ).reset_index()
    track_vote_df.columns = ['model', 'acerto_track']

    sns.barplot(data=track_vote_df, x='model', y='acerto_track', hue='model', ax=axes[2], palette='rocket', legend=False)
    axes[2].set_title('Acerto por Track — Votação (maior = melhor)')
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel('Proporção de Veículos Corretos')

    plt.tight_layout()
    chart_path = OUTPUT_DIR / 'alpr_comparativo.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Gráficos salvos em: {chart_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark ALPR com UFPR-ALPR")
    parser.add_argument('--split',  type=str, default='testing',
                        choices=['testing', 'training', 'validation'])
    parser.add_argument('--limit',  type=int, default=None,
                        help='Limite de imagens por split (ex: 20 para teste rápido)')
    args = parser.parse_args()

    df = run_benchmark(split=args.split, limit=args.limit)
