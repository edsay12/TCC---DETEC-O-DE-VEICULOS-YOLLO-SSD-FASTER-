"""
Runner do benchmark UA-DETRAC para o TCC.

O dataset UA-DETRAC está organizado em formato YOLO:
  labels/train/<sequência>/<frame>.txt   — classe cx cy w h (normalizado)
  images/train/<sequência>/<frame>.jpg

Este módulo:
  1. Faz o matching entre imagens do dataset_processado (diurno/noturno/baixa_qualidade)
     e os labels originais do UA-DETRAC.
  2. Roda cada detector e mede Precision, Recall, F1, AP e FPS.
  3. Retorna um DataFrame pronto para análise.
"""

import cv2
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.utils.metrics import evaluate_batch, match_detections


# ─────────────────────────────────────────────────────────────────────────────
# Parser de labels YOLO → bboxes absolutas
# ─────────────────────────────────────────────────────────────────────────────

def _yolo_to_abs(cx: float, cy: float, w: float, h: float,
                 img_w: int, img_h: int) -> list:
    """Converte bbox YOLO normalizada para [x1, y1, x2, y2] absoluto."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def _load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> list:
    """
    Lê um arquivo de label YOLO e retorna a lista de bboxes absolutas.
    Classe 1 = veículo (UA-DETRAC usa classe 0 e 1 para veículos).
    """
    if not label_path.exists():
        return []
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # cls = int(parts[0])  # todas as classes são veículos no UA-DETRAC
            cx, cy, bw, bh = map(float, parts[1:5])
            boxes.append(_yolo_to_abs(cx, cy, bw, bh, img_w, img_h))
    return boxes


# ─────────────────────────────────────────────────────────────────────────────
# Indexador: mapeia nome de frame → caminho do label no UA-DETRAC
# ─────────────────────────────────────────────────────────────────────────────

def _build_label_index(ua_detrac_root: Path) -> dict:
    """
    Cria um índice {stem_do_frame: Path_do_label} percorrendo
    UA-DETRAC/DETRAC_Upload/labels/train e val.
    """
    index = {}
    labels_root = ua_detrac_root / 'DETRAC_Upload' / 'labels'
    for split_dir in labels_root.iterdir():
        if not split_dir.is_dir():
            continue
        # Tenta layout flat (arquivos de label diretamente sob o split)
        for lbl in split_dir.glob('*.txt'):
            index[lbl.stem] = lbl
        # Tenta layout nested (arquivos de label agrupados em subpastas de sequências)
        for seq_dir in split_dir.iterdir():
            if seq_dir.is_dir():
                for lbl in seq_dir.glob('*.txt'):
                    index[lbl.stem] = lbl
    return index


# ─────────────────────────────────────────────────────────────────────────────
# Função principal de benchmark
# ─────────────────────────────────────────────────────────────────────────────

VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'van'}


def run_ua_detrac_benchmark(
    scenario_dirs: dict,
    models_config: list,
    ua_detrac_root: Path,
    n_samples: int = 100,
    iou_threshold: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Executa o benchmark de detecção de veículos com o UA-DETRAC.

    Args:
        scenario_dirs : dict {'diurno': Path, 'noturno': Path, 'baixa_qualidade': Path}
        models_config : lista de (nome, detector_instance)
        ua_detrac_root: pasta raiz do UA-DETRAC (contém DETRAC_Upload/)
        n_samples     : número de imagens por cenário (para viabilizar execução)
        iou_threshold : limiar de IoU para TP
        seed          : semente para reprodutibilidade da amostragem

    Returns:
        DataFrame com uma linha por (modelo, cenário, frame)
    """
    random.seed(seed)

    print("Indexando labels do UA-DETRAC...")
    label_index = _build_label_index(ua_detrac_root)
    print(f"  {len(label_index)} labels indexados.")

    all_rows = []

    for model_name, detector in models_config:
        for scenario_name, scenario_dir in scenario_dirs.items():
            # Coleta imagens do cenário
            img_paths = list(scenario_dir.rglob('*.jpg')) + \
                        list(scenario_dir.rglob('*.png'))

            if not img_paths:
                print(f"  [AVISO] Nenhuma imagem em {scenario_dir}")
                continue

            # Amostragem aleatória
            sample = random.sample(img_paths, min(n_samples, len(img_paths)))

            results_batch = []

            for img_path in tqdm(sample,
                                 desc=f'{model_name} | {scenario_name}',
                                 leave=False):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h_img, w_img = img.shape[:2]

                # Ground truth via label index
                gt_boxes = label_index.get(img_path.stem, None)
                if gt_boxes is not None:
                    gt_boxes = _load_gt_boxes(gt_boxes, w_img, h_img)
                else:
                    gt_boxes = []

                # Inferência
                t0 = time.perf_counter()
                detections = detector.detect(img)
                elapsed = time.perf_counter() - t0

                # Filtra só veículos
                pred_boxes   = []
                conf_scores  = []
                for d in detections:
                    if d['class'].lower() in VEHICLE_CLASSES:
                        pred_boxes.append(d['bbox'])
                        conf_scores.append(d['conf'])

                results_batch.append({
                    'pred_boxes' : pred_boxes,
                    'gt_boxes'   : gt_boxes,
                    'time_s'     : elapsed,
                    'conf_scores': conf_scores,
                })

                # Linha por frame
                tp, fp, fn = match_detections(pred_boxes, gt_boxes, iou_threshold)
                fps  = 1.0 / elapsed if elapsed > 0 else 0.0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1   = (2 * prec * rec / (prec + rec)
                        if (prec + rec) > 0 else 0.0)

                all_rows.append({
                    'model'    : model_name,
                    'scenario' : scenario_name,
                    'image'    : img_path.name,
                    'n_pred'   : len(pred_boxes),
                    'n_gt'     : len(gt_boxes),
                    'tp'       : tp,
                    'fp'       : fp,
                    'fn'       : fn,
                    'precision': round(prec, 4),
                    'recall'   : round(rec,  4),
                    'f1'       : round(f1,   4),
                    'fps'      : round(fps,  2),
                    'confidence': round(float(np.mean(conf_scores))
                                        if conf_scores else 0.0, 4),
                    'time_ms'  : round(elapsed * 1000, 2),
                })

    return pd.DataFrame(all_rows)


def compute_map_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula mAP por (modelo, cenário) a partir do DataFrame de resultados frame a frame.

    Usa as colunas: tp, fp, n_gt por imagem.
    AP é calculado via AUC da curva Precision-Recall.
    """
    from src.utils.metrics import compute_ap_from_matches

    records = []
    for (model, scenario), grp in df.groupby(['model', 'scenario']):
        # Para AP: ordena por confiança desc e acumula TP/FP
        grp_sorted = grp.sort_values('confidence', ascending=False)
        tp_flags = []
        fp_flags = []
        for _, row in grp_sorted.iterrows():
            # Simplificação: cada frame com tp>0 contribui com tp TPs e fp FPs
            tp_flags.extend([1] * row['tp'] + [0] * row['fp'])
            fp_flags.extend([0] * row['tp'] + [1] * row['fp'])

        n_gt = grp['n_gt'].sum()
        ap   = compute_ap_from_matches(tp_flags, fp_flags, n_gt)

        records.append({
            'model'   : model,
            'scenario': scenario,
            'AP'      : round(ap, 4),
        })

    map_df = pd.DataFrame(records)
    overall = map_df.groupby('model')['AP'].mean().reset_index()
    overall['scenario'] = 'GERAL (mAP)'
    overall.rename(columns={'AP': 'AP'}, inplace=True)

    return pd.concat([map_df, overall], ignore_index=True)
