import sys
import cv2
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# Garante que a raiz do projeto está no sys.path para imports de src/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.detectors import YOLODetector, TorchvisionDetector, PlateDetector
from src.ocr.ocr_engine import OCREngine
from src.utils.ufpr_parser import load_dataset_split
from src.utils.evaluator import evaluate_detection, compute_iou

BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path.cwd()
DATASET_ROOT = BASE_DIR / "data" / "UFPR-ALPR"
OUTPUT_DIR = BASE_DIR / "data" / "processado" / "resultados"


# =========================================================
# Utilitários
# =========================================================
VALID_VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle'}


def clip_box(box, w, h):
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    return [x1, y1, x2, y2]


def expand_box(box, w, h, margin=0.10):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    dx = int(bw * margin)
    dy = int(bh * margin)
    return clip_box([x1 - dx, y1 - dy, x2 + dx, y2 + dy], w, h)


def crop_image(img, box):
    x1, y1, x2, y2 = map(int, box)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None
    return crop


def shift_box_to_crop(box, crop_origin):
    """
    Converte bbox do sistema global da imagem para o sistema local do recorte.
    crop_origin = (x1_crop, y1_crop)
    """
    x1, y1, x2, y2 = box
    ox, oy = crop_origin
    return [x1 - ox, y1 - oy, x2 - ox, y2 - oy]


def choose_best_vehicle(vehicles, gt_vehicle_box):
    """
    Escolhe o veículo detectado com maior IoU com o GT do veículo.
    """
    if not vehicles:
        return None, 0.0

    best_det = None
    best_iou = -1.0

    for det in vehicles:
        det_box = det['bbox']
        iou = compute_iou(det_box, gt_vehicle_box)
        if iou > best_iou:
            best_iou = iou
            best_det = det

    return best_det, best_iou


def majority_vote(series):
    non_empty = series.dropna()
    non_empty = non_empty[non_empty != '']
    if non_empty.empty:
        return ''
    return non_empty.mode().iloc[0]


# =========================================================
# Benchmark principal
# =========================================================
def run_benchmark(
    split='testing',
    limit=None,
    mode='end_to_end',           # 'end_to_end' ou 'ocr_oracle'
    vehicle_margin=0.10,
    save_csv=True
):
    """
    mode='end_to_end':
        - usa detector de veículo
        - usa detector de placa
        - se a placa não for detectada => falha real (sem GT fallback da placa)

    mode='ocr_oracle':
        - usa GT da placa para recorte da placa
        - serve para medir o OCR isoladamente
    """
    assert mode in {'end_to_end', 'ocr_oracle'}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  BENCHMARK ALPR — UFPR-ALPR ({mode})")
    print("=" * 70)

    print(f"\nCarregando split '{split}'...")
    samples = load_dataset_split(DATASET_ROOT, split=split, limit=limit)
    print(f"  Total de amostras carregadas: {len(samples)}")

    # -------------------------
    # Detectores de veículo
    # -------------------------
    models_config = [
        ('YOLOv8',       lambda: YOLODetector(str(BASE_DIR / 'models' / 'yolov8n.pt'))),
        ('SSD',          lambda: TorchvisionDetector(model_type='ssd')),
        ('Faster R-CNN', lambda: TorchvisionDetector(model_type='faster_rcnn')),
    ]

    vehicle_detectors = []
    for name, factory in models_config:
        try:
            print(f"  Carregando {name}...", end=' ')
            det = factory()
            vehicle_detectors.append((name, det))
            print("OK")
        except Exception as e:
            print(f"FALHOU ({type(e).__name__}): {e}")

    if not vehicle_detectors:
        print("Nenhum detector carregou com sucesso.")
        return None

    # Detector de placa
    plate_detector = PlateDetector()

    # OCR
    ocr = OCREngine(engine_type='easyocr')

    all_results = []

    # ======================================================
    # Loop por modelo
    # ======================================================
    for det_name, vehicle_det in vehicle_detectors:
        print(f"\n>>> Testando: {det_name}")

        for sample in tqdm(samples, desc=det_name):
            img_path = sample['image_path']
            ann = sample['annotation']

            gt_text = ann['plate_text']
            gt_plate = ann['plate_bbox']       # bbox da placa na imagem inteira
            gt_vehicle = ann['vehicle_bbox']   # bbox do veículo na imagem inteira

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h_img, w_img = img.shape[:2]

            # -------------------------
            # ETAPA 1: detectar veículo
            # -------------------------
            detections = vehicle_det.detect(img)
            vehicles = [
                d for d in detections
                if str(d.get('class', '')).lower() in VALID_VEHICLE_CLASSES
            ]

            best_vehicle, vehicle_iou = choose_best_vehicle(vehicles, gt_vehicle)

            if best_vehicle is not None:
                vehicle_box = clip_box(best_vehicle['bbox'], w_img, h_img)
                vehicle_found = True
                vehicle_box_source = 'detected'
            else:
                # fallback apenas para não perder o frame no pipeline,
                # mas isso fica registrado no CSV
                vehicle_box = clip_box(gt_vehicle, w_img, h_img)
                vehicle_found = False
                vehicle_box_source = 'gt_fallback'

            # expande o recorte do veículo
            vehicle_box = expand_box(vehicle_box, w_img, h_img, margin=vehicle_margin)
            vx1, vy1, vx2, vy2 = vehicle_box

            vehicle_crop = crop_image(img, vehicle_box)
            if vehicle_crop is None:
                continue

            # GT da placa no sistema do crop do veículo
            gt_plate_crop = shift_box_to_crop(gt_plate, (vx1, vy1))

            # -------------------------
            # ETAPA 2: detectar placa
            # -------------------------
            pred_bbox = None
            plate_crop = None
            plate_found = False
            plate_bbox_source = 'none'

            if mode == 'ocr_oracle':
                # usa a placa GT diretamente
                plate_bbox_source = 'gt_oracle'
                plate_found = True
                pred_bbox = gt_plate_crop[:]  # para IoU = 1 no modo oracle
                plate_crop = crop_image(vehicle_crop, gt_plate_crop)

            else:
                # mode = end_to_end
                plate_detections = plate_detector.detect(vehicle_crop)

                if plate_detections:
                    best_plate = max(plate_detections, key=lambda p: p.get('conf', 0.0))
                    pred_bbox = best_plate['bbox']
                    plate_bbox_source = 'detected'
                    plate_found = True
                    plate_crop = crop_image(vehicle_crop, pred_bbox)
                else:
                    # SEM fallback GT no benchmark real
                    pred_bbox = None
                    plate_crop = None
                    plate_found = False
                    plate_bbox_source = 'none'

            # -------------------------
            # ETAPA 3: OCR
            # -------------------------
            pred_text = ''
            ocr_executed = False

            if plate_crop is not None and plate_crop.size > 0:
                ocr_executed = True
                pred_text = ocr.read_plate(plate_crop) or ''

            # -------------------------
            # AVALIAÇÃO
            # -------------------------
            metrics = evaluate_detection(
                pred_bbox,
                gt_plate_crop,
                pred_text,
                gt_text
            )

            all_results.append({
                'mode': mode,
                'model': det_name,
                'track': img_path.parent.name,
                'image': img_path.name,

                'gt_plate': gt_text,
                'pred_plate': pred_text,

                'vehicle_found': vehicle_found,
                'vehicle_iou': round(vehicle_iou, 4) if vehicle_iou is not None else 0.0,
                'vehicle_bbox_source': vehicle_box_source,

                'plate_found': plate_found,
                'plate_bbox_source': plate_bbox_source,

                'ocr_executed': ocr_executed,

                'iou': round(metrics['iou'], 4),
                'cer': round(metrics['cer'], 4),
                'exact_match': metrics['exact_match'],
            })

    df = pd.DataFrame(all_results)

    if save_csv:
        csv_name = f'alpr_benchmark_{mode}.csv'
        csv_path = OUTPUT_DIR / csv_name
        df.to_csv(csv_path, index=False)
        print(f"\nResultados salvos em: {csv_path}")

    _generate_report(df, mode=mode)
    return df


# =========================================================
# Relatório
# =========================================================
def _generate_report(df: pd.DataFrame, mode='end_to_end'):
    print("\n" + "=" * 70)
    print(f"  RELATÓRIO FINAL — UFPR-ALPR ({mode})")
    print("=" * 70)

    summary = df.groupby('model').agg(
        Total_Frames=('image', 'count'),
        Vehicle_Detected=('vehicle_found', 'mean'),
        Plate_Found=('plate_found', 'mean'),
        OCR_Executado=('ocr_executed', 'mean'),
        IoU_Medio=('iou', 'mean'),
        CER_Medio=('cer', 'mean'),
        Acerto_Frame=('exact_match', 'mean'),
    ).round(4)

    # acerto por track via votação
    track_acc = (
        df.groupby(['model', 'track'])
          .apply(lambda g: majority_vote(g['pred_plate']) == g['gt_plate'].iloc[0])
          .groupby(level=0)
          .mean()
          .rename('Acerto_Track_Voto')
    )

    summary = summary.join(track_acc)

    # formatação %
    for col in ['Vehicle_Detected', 'Plate_Found', 'OCR_Executado', 'Acerto_Frame', 'Acerto_Track_Voto']:
        summary[col] = (summary[col] * 100).round(1).astype(str) + '%'

    print(summary.to_string())
    print()

    # gráficos
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(f'Benchmark ALPR — UFPR-ALPR ({mode})', fontsize=14, fontweight='bold')

    plot_df = df.groupby('model').agg({
        'iou': 'mean',
        'cer': 'mean',
        'plate_found': 'mean',
        'exact_match': 'mean'
    }).reset_index()

    sns.barplot(data=plot_df, x='model', y='iou', hue='model', ax=axes[0], legend=False)
    axes[0].set_title('IoU médio')
    axes[0].set_ylim(0, 1)

    sns.barplot(data=plot_df, x='model', y='cer', hue='model', ax=axes[1], legend=False)
    axes[1].set_title('CER médio')
    axes[1].set_ylim(0, 1)

    sns.barplot(data=plot_df, x='model', y='plate_found', hue='model', ax=axes[2], legend=False)
    axes[2].set_title('Taxa de placa encontrada')
    axes[2].set_ylim(0, 1)

    sns.barplot(data=plot_df, x='model', y='exact_match', hue='model', ax=axes[3], legend=False)
    axes[3].set_title('Acerto por frame')
    axes[3].set_ylim(0, 1)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / f'alpr_comparativo_{mode}.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Gráficos salvos em: {chart_path}")


# =========================================================
# Execução
# =========================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark ALPR com UFPR-ALPR")
    parser.add_argument('--split', type=str, default='testing',
                        choices=['testing', 'training', 'validation'])
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--mode', type=str, default='end_to_end',
                        choices=['end_to_end', 'ocr_oracle'])
    parser.add_argument('--vehicle_margin', type=float, default=0.10)
    args = parser.parse_args()

    run_benchmark(
        split=args.split,
        limit=args.limit,
        mode=args.mode,
        vehicle_margin=args.vehicle_margin
    )
