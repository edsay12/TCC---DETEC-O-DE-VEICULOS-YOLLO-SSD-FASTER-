"""
Parser para o dataset UFPR-ALPR.

Cada imagem .png tem um arquivo .txt correspondente com as anotações:
    camera: GoPro Hero4 Silver
    position_vehicle: 803 367 281 245   (x, y, w, h)
    plate: AQY6388
    corners: 910,482 987,483 987,508 909,508   (x1,y1 x2,y2 x3,y3 x4,y4)
    char 1: 912 492 9 14   (x, y, w, h de cada caractere)
"""

from pathlib import Path


def parse_annotation(txt_path: Path) -> dict:
    """
    Lê um arquivo .txt do UFPR-ALPR e retorna um dicionário com:
        - plate_text   : str   — ex: 'AQY6388'
        - vehicle_bbox : list  — [x1, y1, x2, y2]
        - plate_bbox   : list  — [x1, y1, x2, y2] (bounding rect dos 4 cantos)
        - plate_corners: list  — [[x,y], [x,y], [x,y], [x,y]]
    Retorna None se o arquivo não puder ser lido.
    """
    try:
        data = {}
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            # Posição do veículo: x y w h
            if line.startswith('position_vehicle:'):
                parts = line.split(':')[1].strip().split()
                x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                data['vehicle_bbox'] = [x, y, x + w, y + h]

            # Texto da placa
            elif line.startswith('plate:'):
                data['plate_text'] = line.split(':')[1].strip()

            # Cantos da placa: x1,y1 x2,y2 x3,y3 x4,y4
            elif line.startswith('corners:'):
                corners_str = line.split(':')[1].strip().split()
                corners = []
                for c in corners_str:
                    cx, cy = c.split(',')
                    corners.append([int(cx), int(cy)])
                data['plate_corners'] = corners

                # Calcula bounding box retangular a partir dos 4 cantos
                xs = [p[0] for p in corners]
                ys = [p[1] for p in corners]
                data['plate_bbox'] = [min(xs), min(ys), max(xs), max(ys)]

        return data if 'plate_text' in data else None

    except Exception as e:
        print(f"Erro ao parsear {txt_path}: {e}")
        return None


def load_dataset_split(dataset_root: Path, split: str = 'testing', limit: int = None) -> list:
    """
    Carrega todos os pares (imagem, anotação) de um split do UFPR-ALPR.

    Args:
        dataset_root: caminho para a pasta 'UFPR-ALPR dataset'
        split: 'testing', 'training' ou 'validation'
        limit: número máximo de amostras a carregar (None = todas)

    Returns:
        Lista de dicts com chaves: 'image_path', 'annotation'
    """
    split_path = dataset_root / split
    samples = []

    # O dataset é organizado em track0001/, track0002/, etc.
    for track_dir in sorted(split_path.iterdir()):
        if not track_dir.is_dir():
            continue
        for img_path in sorted(track_dir.glob('*.png')):
            txt_path = img_path.with_suffix('.txt')
            if not txt_path.exists():
                continue
            annotation = parse_annotation(txt_path)
            if annotation:
                samples.append({
                    'image_path': img_path,
                    'annotation': annotation
                })
            if limit and len(samples) >= limit:
                return samples

    return samples
