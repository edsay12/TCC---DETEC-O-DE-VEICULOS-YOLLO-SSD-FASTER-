"""
Módulo de avaliação de métricas para o benchmark ALPR do TCC.

Métricas implementadas:
    - IoU  (Intersection over Union): precisão do recorte da placa
    - CER  (Character Error Rate)   : precisão do OCR caractere a caractere
    - Acerto Total (Exact Match)    : placa lida 100% correta
"""


def compute_iou(box_a: list, box_b: list) -> float:
    """
    Calcula o IoU entre duas bounding boxes no formato [x1, y1, x2, y2].

    IoU = Área de Interseção / Área de União

    Retorna um valor entre 0.0 (sem sobreposição) e 1.0 (sobreposição perfeita).
    """
    # Interseção
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    # Área de cada caixa
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def compute_cer(predicted: str, ground_truth: str) -> float:
    """
    Calcula o CER (Character Error Rate) entre dois textos.

    CER = (Substituições + Inserções + Deleções) / len(ground_truth)

    Baseado na distância de edição de Levenshtein.
    Retorna 0.0 para acerto perfeito, 1.0 para erro total.
    """
    pred = predicted.upper().replace(' ', '') if predicted else ''
    gt   = ground_truth.upper().replace(' ', '') if ground_truth else ''

    if not gt:
        return 0.0 if not pred else 1.0

    # Matriz de Levenshtein
    n, m = len(gt), len(pred)
    dp = list(range(m + 1))

    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if gt[i - 1] == pred[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])

    return min(dp[m] / n, 1.0)


def is_exact_match(predicted: str, ground_truth: str) -> bool:
    """Retorna True se a placa foi lida exatamente correta (ignorando maiúsculas)."""
    pred = predicted.upper().replace(' ', '').replace('-', '') if predicted else ''
    gt   = ground_truth.upper().replace(' ', '').replace('-', '') if ground_truth else ''
    return pred == gt


def evaluate_detection(predicted_bbox: list, ground_truth_bbox: list,
                        predicted_text: str, ground_truth_text: str) -> dict:
    """
    Avalia uma única detecção de placa contra o ground truth.

    Retorna dict com:
        - iou         : float (0-1)
        - cer         : float (0-1, menor é melhor)
        - exact_match : bool
        - plate_found : bool (IoU > 0.3)
    """
    iou = compute_iou(predicted_bbox, ground_truth_bbox) if predicted_bbox else 0.0
    cer = compute_cer(predicted_text, ground_truth_text)
    match = is_exact_match(predicted_text, ground_truth_text)

    return {
        'iou': iou,
        'cer': cer,
        'exact_match': match,
        'plate_found': iou > 0.3,  # Limiar padrão para "detectou corretamente"
    }
