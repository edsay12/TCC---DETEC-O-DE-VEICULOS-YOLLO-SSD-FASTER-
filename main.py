import argparse
from src.models.detectors import YOLODetector, TorchvisionDetector
from src.ocr.ocr_engine import OCREngine
from src.core.pipeline import TrafficPipeline
from pathlib import Path

def run_experiment(model_type, ocr_type, dataset_path, output_dir):
    print(f"--- Iniciando Experimento TCC ---")
    print(f"Modelo: {model_type} | OCR: {ocr_type}")
    
    # 1. Selecionar Detector
    if model_type == 'yolo':
        detector = YOLODetector()
    elif model_type == 'ssd':
        detector = TorchvisionDetector(model_type='ssd')
    else:
        detector = TorchvisionDetector(model_type='faster_rcnn')
        
    # 2. Selecionar OCR
    ocr = OCREngine(engine_type=ocr_type)
    
    # 3. Inicializar Pipeline
    pipeline = TrafficPipeline(detector, ocr, output_dir)
    
    # 4. Executar
    pipeline.run_on_dataset(dataset_path)
    
    print("\n[SUCESSO] Experimento concluído!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCC Traffic Monitoring Experiment")
    parser.add_argument("--model", type=str, default="yolo", choices=["yolo", "ssd", "faster_rcnn"])
    parser.add_argument("--ocr", type=str, default="easyocr", choices=["easyocr", "tesseract"])
    parser.add_argument("--input", type=str, default="dataset_processado")
    parser.add_argument("--output", type=str, default="dataset_processado")
    
    args = parser.parse_args()
    
    run_experiment(args.model, args.ocr, args.input, args.output)
