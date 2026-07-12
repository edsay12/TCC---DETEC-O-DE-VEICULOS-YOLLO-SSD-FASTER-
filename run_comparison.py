import argparse
from src.models.detectors import YOLODetector, TorchvisionDetector
from src.ocr.ocr_engine import OCREngine
from src.core.pipeline import TrafficPipeline
from pathlib import Path

def run_full_comparison(dataset_path, output_dir, limit=None):
    print("="*50)
    print("INICIANDO COMPARAÇÃO MULTI-MODELO TCC")
    print("="*50)
    
    models_to_test = [
        ('yolo', 'yolov8n.pt'),
        ('ssd', 'ssd'),
        ('faster_rcnn', 'faster_rcnn')
    ]
    
    # OCR padrão para comparação de detectores
    ocr = OCREngine(engine_type='easyocr')
    
    # Detector de placas para o segundo estágio
    from src.models.detectors import PlateDetector
    plate_detector = PlateDetector()
    
    # Logger único para consolidar resultados de todos os modelos
    from src.utils.metrics import ExperimentLogger
    logger = ExperimentLogger(output_dir)
    
    for model_type, model_key in models_to_test:
        print(f"\n>>> Testando Modelo: {model_type.upper()}")
        
        if model_type == 'yolo':
            detector = YOLODetector(model_key)
        else:
            detector = TorchvisionDetector(model_type=model_key)
            
        pipeline = TrafficPipeline(detector, ocr, output_dir, plate_detector=plate_detector)
        pipeline.logger = logger # Injeta o logger compartilhado
        
        # Modificamos o run_on_dataset para aceitar um limite opcional para testes rápidos
        base_path = Path(dataset_path)
        scenarios = ['diurno', 'noturno', 'baixa_qualidade']
        
        for scenario in scenarios:
            scenario_path = base_path / scenario
            if not scenario_path.exists(): continue
            
            files = list(scenario_path.glob("*.jpg"))
            if limit:
                files = files[:limit]
            
            print(f"Processando {len(files)} imagens no cenário: {scenario}")
            for file_path in files:
                import cv2
                frame = cv2.imread(str(file_path))
                if frame is not None:
                    pipeline.process_frame(frame, file_path.name, scenario)
                    
    # Salvar todos os resultados combinados
    final_report = logger.save_results()
    print(f"\n[SUCESSO] Comparação finalizada. Relatório: {final_report}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCC Multi-Model Comparison")
    parser.add_argument("--input", type=str, default="dataset_processado")
    parser.add_argument("--output", type=str, default="dataset_processado")
    parser.add_argument("--limit", type=int, default=None, help="Limite de imagens por cenário para teste rápido")
    
    args = parser.parse_args()
    run_full_comparison(args.input, args.output, args.limit)
