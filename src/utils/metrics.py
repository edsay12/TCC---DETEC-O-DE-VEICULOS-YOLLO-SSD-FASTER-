"""
Módulo de métricas para o benchmark do TCC.

Implementa:
  - Precision, Recall, F1-score por imagem/batch
  - AP (Average Precision) via interpolação de 11 pontos (PASCAL VOC)
  - mAP (mean Average Precision) por modelo
  - ExperimentLogger para registro frame a frame
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Funções de IoU e matching de bounding boxes
# ─────────────────────────────────────────────────────────────────────────────

def iou_single(box_a: list, box_b: list) -> float:
    """IoU entre dois boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def match_detections(pred_boxes: list, gt_boxes: list,
                     iou_threshold: float = 0.5) -> tuple:
    """
    Faz o matching entre predições e ground-truth usando IoU.

    Retorna:
        tp (int): true positives
        fp (int): false positives
        fn (int): false negatives
    """
    if not gt_boxes:
        return 0, len(pred_boxes), 0
    if not pred_boxes:
        return 0, 0, len(gt_boxes)

    matched_gt = set()
    tp = 0
    fp = 0

    for pb in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for gi, gb in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = iou_single(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gi

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


# ─────────────────────────────────────────────────────────────────────────────
# Precision / Recall / F1
# ─────────────────────────────────────────────────────────────────────────────

def compute_precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def compute_recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def compute_f1(precision: float, recall: float) -> float:
    return (2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# AP (Average Precision) — interpolação PASCAL VOC 11 pontos
# ─────────────────────────────────────────────────────────────────────────────

def compute_ap(precisions: list, recalls: list) -> float:
    """
    Calcula AP usando interpolação de 11 pontos (PASCAL VOC).

    Args:
        precisions: lista de valores de precision em ordem crescente de recall
        recalls   : lista de valores de recall correspondentes

    Returns:
        AP (float 0–1)
    """
    ap = 0.0
    for threshold in np.arange(0.0, 1.1, 0.1):
        prec_at_rec = [p for p, r in zip(precisions, recalls) if r >= threshold]
        ap += max(prec_at_rec) if prec_at_rec else 0.0
    return ap / 11.0


def compute_ap_from_matches(tp_list: list, fp_list: list, n_gt: int) -> float:
    """
    Calcula AP a partir de listas cumulativas de TP e FP (ordenadas por confiança).

    Args:
        tp_list : lista de 0/1 indicando TP para cada detecção (ordem conf desc)
        fp_list : lista de 0/1 indicando FP para cada detecção
        n_gt    : total de ground-truth boxes

    Returns:
        AP (float 0–1)
    """
    if n_gt == 0:
        return 0.0

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)

    recalls    = tp_cum / n_gt
    precisions = tp_cum / (tp_cum + fp_cum + 1e-9)

    # Adiciona pontos sentinela
    recalls    = np.concatenate(([0.0], recalls,    [recalls[-1]  if len(recalls)  else 0.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))

    # Torna a curva monotonicamente decrescente
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Área sob a curva (mudanças de recall)
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    ap  = np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1])
    return float(ap)


# ─────────────────────────────────────────────────────────────────────────────
# Métricas agregadas para um conjunto de imagens
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_batch(results: list, iou_threshold: float = 0.5) -> dict:
    """
    Avalia um lote de imagens com predições e ground-truth.

    Args:
        results: lista de dicts com chaves:
            'pred_boxes' : [[x1,y1,x2,y2], ...]  (coordenadas absolutas)
            'gt_boxes'   : [[x1,y1,x2,y2], ...]
            'time_s'     : float (tempo de inferência em segundos)
            'conf_scores': [float, ...]            (confiança de cada pred)
        iou_threshold: limiar de IoU para considerar TP

    Returns:
        dict com precision, recall, f1, ap, fps, mean_conf
    """
    total_tp = total_fp = total_fn = 0
    all_tp_flags = []
    all_fp_flags = []
    n_gt_total   = 0
    total_time   = 0.0
    all_confs    = []

    for r in results:
        pred = r.get('pred_boxes', [])
        gt   = r.get('gt_boxes',   [])
        tp, fp, fn = match_detections(pred, gt, iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        n_gt_total += len(gt)
        total_time += r.get('time_s', 0.0)
        all_confs.extend(r.get('conf_scores', []))

        # Para AP: 1 TP flag por predição
        for pb in pred:
            best = max((iou_single(pb, gb) for gb in gt), default=0.0)
            all_tp_flags.append(1 if best >= iou_threshold else 0)
            all_fp_flags.append(0 if best >= iou_threshold else 1)

    precision = compute_precision(total_tp, total_fp)
    recall    = compute_recall(total_tp, total_fn)
    f1        = compute_f1(precision, recall)
    ap        = compute_ap_from_matches(all_tp_flags, all_fp_flags, n_gt_total)
    fps       = len(results) / total_time if total_time > 0 else 0.0
    mean_conf = float(np.mean(all_confs)) if all_confs else 0.0

    return {
        'precision' : round(precision, 4),
        'recall'    : round(recall,    4),
        'f1'        : round(f1,        4),
        'ap'        : round(ap,        4),
        'fps'       : round(fps,       2),
        'mean_conf' : round(mean_conf, 4),
        'tp'        : total_tp,
        'fp'        : total_fp,
        'fn'        : total_fn,
        'n_frames'  : len(results),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentLogger — registro frame a frame
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentLogger:
    """Registra resultados frame a frame e gera CSV + relatório."""

    def __init__(self, output_dir: str | Path):
        self.output_dir  = Path(output_dir)
        self.logs_dir    = self.output_dir / "logs"
        self.results_dir = self.output_dir / "resultados"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.data = []

    def log_frame(self, filename: str, scenario: str, model_name: str,
                  pred_boxes: list, gt_boxes: list,
                  time_s: float, conf_scores: list = None):
        """Registra uma detecção de um único frame."""
        tp, fp, fn = match_detections(pred_boxes, gt_boxes)
        prec = compute_precision(tp, fp)
        rec  = compute_recall(tp, fn)
        f1   = compute_f1(prec, rec)
        fps  = 1.0 / time_s if time_s > 0 else 0.0
        avg_conf = float(np.mean(conf_scores)) if conf_scores else 0.0

        self.data.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'filename' : filename,
            'scenario' : scenario,
            'model'    : model_name,
            'n_pred'   : len(pred_boxes),
            'n_gt'     : len(gt_boxes),
            'tp'       : tp,
            'fp'       : fp,
            'fn'       : fn,
            'precision': round(prec, 4),
            'recall'   : round(rec,  4),
            'f1'       : round(f1,   4),
            'time_ms'  : round(time_s * 1000, 2),
            'fps'      : round(fps, 2),
            'confidence': round(avg_conf, 4),
        })

    # compat com código legado que usa log_detection
    def log_detection(self, filename, scenario, model_name,
                      vehicles_count, plates_count, processing_time, avg_conf=0.0):
        self.data.append({
            'timestamp' : time.strftime('%Y-%m-%d %H:%M:%S'),
            'filename'  : filename,
            'scenario'  : scenario,
            'model'     : model_name,
            'n_pred'    : vehicles_count,
            'n_gt'      : 0,
            'tp': 0, 'fp': 0, 'fn': 0,
            'precision' : 0.0, 'recall': 0.0, 'f1': 0.0,
            'time_ms'   : processing_time * 1000,
            'fps'       : 1.0 / processing_time if processing_time > 0 else 0.0,
            'confidence': avg_conf,
        })

    def save_results(self) -> Path:
        if not self.data:
            print("Nenhum dado para salvar.")
            return None
        df = pd.DataFrame(self.data)
        csv_path = self.results_dir / 'experimento_detalhado.csv'
        df.to_csv(csv_path, index=False)
        print(f"CSV salvo em: {csv_path}")
        return csv_path

    def generate_plots(self, df: pd.DataFrame = None) -> Path:
        """Gera gráficos comparativos FPS × Precisão × F1."""
        if df is None:
            df = pd.DataFrame(self.data)
        if df.empty:
            return None

        sns.set_theme(style='whitegrid')
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Benchmark UA-DETRAC — Comparativo de Modelos',
                     fontsize=14, fontweight='bold')

        sns.barplot(data=df, x='scenario', y='fps',
                    hue='model', ax=axes[0], palette='viridis')
        axes[0].set_title('FPS por Cenário')
        axes[0].set_ylabel('Frames por Segundo')

        sns.barplot(data=df, x='scenario', y='precision',
                    hue='model', ax=axes[1], palette='magma')
        axes[1].set_title('Precision por Cenário')
        axes[1].set_ylabel('Precision')
        axes[1].set_ylim(0, 1)

        sns.barplot(data=df, x='scenario', y='f1',
                    hue='model', ax=axes[2], palette='rocket')
        axes[2].set_title('F1-Score por Cenário')
        axes[2].set_ylabel('F1-Score')
        axes[2].set_ylim(0, 1)

        plt.tight_layout()
        plot_path = self.results_dir / 'comparativo_modelos.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        return plot_path
