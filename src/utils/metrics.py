"""
Módulo de métricas para o benchmark do TCC.

Implementa:
  - IoU (Intersection over Union) entre bounding boxes
  - Matching de detecções preditas vs ground-truth
  - AP (Average Precision) via área sob a curva Precision-Recall
"""
import numpy as np


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
# AP (Average Precision) — Área sob a curva Precision-Recall
# ─────────────────────────────────────────────────────────────────────────────

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
