import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

## Caminho (raiz do projeto = um nível acima de scripts/)
BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path.cwd()

DATASET_PROCESSADO = BASE_DIR / "data" / "processado"

SUBDIRS = ["diurno", "noturno", "baixa_qualidade", "ruido"]

METADATA_FILE = DATASET_PROCESSADO / "processamento_metadata.csv"

def setup_directories(base_path, subdirs):
    base_path.mkdir(parents=True, exist_ok=True)
    for subdir in subdirs:
        (base_path / subdir).mkdir(parents=True, exist_ok=True)
    
    if not METADATA_FILE.exists():
        df = pd.DataFrame(columns=["timestamp", 
        "original_file", 
        "scenario", 
        "brightness", 
        "blur", 
        "contrast", 
        "noise", 
        "has_label", 
        "target_path"
        ])

        df.to_csv(METADATA_FILE, index=False)

def get_image_metrics(image):
    """Calcula métricas detalhadas"""
    if image is None: return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Brilho Médio (Escala 0-255)
    brightness = np.mean(gray)
    
    # 2. Variância de Laplacian (Foco/Nitidez)
    blur_map = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 3. Contraste (Desvio padrão do brilho)
    contrast = gray.std()
    
    # 4. Estimativa de Ruído (Proxy simples via Laplaciano)
    noise = cv2.meanStdDev(cv2.Laplacian(gray, cv2.CV_64F))[1][0][0]
    
    return {
        "brightness": brightness,
        "blur": blur_map,
        "contrast": contrast,
        "noise": noise
    }

def classify_scenario(metrics):
    """Classificação lógica com thresholds ajustáveis."""
    if not metrics: return "erro"
    
    # Thresholds baseados em literatura e experimentação
    BRIGHTNESS_THRESHOLD_NIGHT = 75
    BRIGHTNESS_THRESHOLD_LOW_LIGHT = 105
    BLUR_THRESHOLD = 80
    NOISE_THRESHOLD = 60 
    
    if metrics['blur'] < BLUR_THRESHOLD:
        return "baixa_qualidade" # Borrada
    
    if metrics['noise'] > NOISE_THRESHOLD:
        return "ruido"
    
    if metrics['brightness'] < BRIGHTNESS_THRESHOLD_NIGHT:
        return "noturno"
    elif metrics['brightness'] < BRIGHTNESS_THRESHOLD_LOW_LIGHT:
        return "baixa_qualidade" # Baixa iluminação
    else:
        return "diurno"

def sync_label(original_img_path, target_folder, new_filename):
    """Busca o arquivo .txt (YOLO) correspondente e o copia para a nova pasta."""
    label_path = Path(str(original_img_path).replace("images", "labels")).with_suffix(".txt")
    
    if label_path.exists():
        target_label_path = target_folder / Path(new_filename).with_suffix(".txt")
        try:
            with open(label_path, 'r') as f:
                content = f.read()
            with open(target_label_path, 'w') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"Erro ao sincronizar label {label_path}: {e}")
            return False
    return False

def preprocess_image(image, target_size=(640, 640)):
    """Aplica melhorias automáticas na imagem (Resize + CLAHE + Blur suave)."""
    if image is None: return None
    
    # 1. Redimensionamento
    img = cv2.resize(image, target_size)
    
    # 2. Equalização de Histograma Adaptativa (CLAHE) - Melhora visibilidade em cenários escuros
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    limg = cv2.merge((clahe.apply(l), a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # 3. Remoção de Ruído (Filtro Gaussiano leve)
    final_img = cv2.GaussianBlur(final_img, (3, 3), 0)
    
    return final_img

def process_image_worker(args):
    """Função worker para processamento paralelo de uma única imagem."""
    img_path, output_base, target_size = args
    try:
        img = cv2.imread(str(img_path))
        if img is None: return None
        
        metrics = get_image_metrics(img)
        scenario = classify_scenario(metrics)
        processed_img = preprocess_image(img, target_size)
        
        # Caminhos de destino
        target_dir = output_base / scenario
        save_path = target_dir / img_path.name
        
        # Salva imagem processada
        cv2.imwrite(str(save_path), processed_img)
        
        # Sincroniza Label se existir
        has_label = sync_label(img_path, target_dir, img_path.name)
        
        # Prepara entrada para o CSV de metadados
        new_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "original_file": img_path.name,
            "scenario": scenario,
            "brightness": metrics['brightness'],
            "blur": metrics['blur'],
            "contrast": metrics['contrast'],
            "noise": metrics['noise'],
            "has_label": has_label,
            "target_path": str(save_path)
        }
        return new_entry
    except Exception as e:
        print(f"Erro ao processar {img_path}: {e}")
        return None

def run_pipeline(input_path, output_path, max_workers=None, chunk_size=1000):
    """Executa o processamento em massa e salva resultados em blocos (chunks)."""
    print(f"\n--- Iniciando Pipeline: {input_path.name} ---")
    extensions = {'.jpg', '.png', '.jpeg'}
    image_tasks = []
    
    # Coleta todas as imagens recursivamente
    for root, _, files in os.walk(input_path):
        if str(output_path) in root: continue
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in extensions:
                image_tasks.append((file_path, output_path, (640, 640)))

    total_images = len(image_tasks)
    if total_images == 0:
        print(f"Nenhuma imagem encontrada em {input_path}")
        return

    print(f"Encontradas {total_images} imagens. Processando com {max_workers or 'todos'} cores...")
    
    # Processa em pedaços (chunks) para não sobrecarregar a memória e salvar progresso
    for i in range(0, total_images, chunk_size):
        chunk = image_tasks[i:i + chunk_size]
        print(f"Processando bloco {i//chunk_size + 1}/{(total_images-1)//chunk_size + 1} ({i} a {min(i+chunk_size, total_images)})...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_image_worker, chunk))
        
        # Filtra falhas e salva bloco no CSV
        valid_results = [r for r in results if r is not None]
        if valid_results:
            df_chunk = pd.DataFrame(valid_results)
            df_chunk.to_csv(METADATA_FILE, mode='a', header=False, index=False)
            print(f"  -> {len(valid_results)} imagens salvas no metadados.")

    print(f"Concluído! Processamento de {input_path.name} finalizado.")

if __name__ == "__main__":
    # Garante estrutura de pastas
    setup_directories(DATASET_PROCESSADO, SUBDIRS)
    
    # Processar UA-DETRAC
    ua_path = BASE_DIR / "data" / "UA-DETRAC" / "DETRAC_Upload" / "images"
    if ua_path.exists():
        run_pipeline(ua_path, DATASET_PROCESSADO)
    else:
        print(f"Aviso: {ua_path} não encontrado.")
