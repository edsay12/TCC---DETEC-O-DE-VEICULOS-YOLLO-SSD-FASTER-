import cv2
import time
from pathlib import Path
from tqdm import tqdm
from src.utils.metrics import ExperimentLogger

class TrafficPipeline:
    def __init__(self, detector, ocr_engine, output_dir, plate_detector=None):
        self.detector = detector
        self.ocr = ocr_engine
        self.plate_detector = plate_detector # Opcional: Detector de placas real
        self.logger = ExperimentLogger(output_dir)
        self.output_dir = Path(output_dir)
        
    def process_frame(self, frame, filename, scenario):
        start_time = time.time()
        
        # 1. Detecção de Veículo
        detections = self.detector.detect(frame)
        
        vehicles = [d for d in detections if d['class'] in ['car', 'truck', 'bus', 'motorcycle']]
        plates_found = 0
        confidences = [v['conf'] for v in vehicles]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        for i, vehicle in enumerate(vehicles):
            x1, y1, x2, y2 = map(int, vehicle['bbox'])
            vehicle_crop = frame[y1:y2, x1:x2]
            
            if vehicle_crop.size == 0: continue
            
            # 2. Detecção/Localização de Placa (ALPR)
            if self.plate_detector:
                # Usa modelo especializado para achar a placa dentro do carro
                plate_detections = self.plate_detector.detect(vehicle_crop)
                if plate_detections:
                    # Pega a placa com maior confiança
                    p = max(plate_detections, key=lambda x: x['conf'])
                    px1, py1, px2, py2 = map(int, p['bbox'])
                    plate_area = vehicle_crop[py1:py2, px1:px2]
                else:
                    plate_area = None
            else:
                # Fallback para Heurística se não houver detector de placas
                h, w = vehicle_crop.shape[:2]
                y_start, y_end = int(h * 0.7), int(h * 0.95)
                x_start, x_end = int(w * 0.15), int(w * 0.85)
                plate_area = vehicle_crop[y_start:y_end, x_start:x_end]
            
            if plate_area is None or plate_area.size == 0: continue
            
            # 3. OCR
            plate_text = self.ocr.read_plate(plate_area)
            
            if plate_text:
                plates_found += 1
                # Salvar recorte da placa
                plate_path = self.output_dir / "placas_detectadas" / f"{Path(filename).stem}_v{i}.jpg"
                plate_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(plate_path), plate_area)
        
        processing_time = time.time() - start_time
        
        # Logar resultados
        self.logger.log_detection(
            filename, scenario, 
            self.detector.model_name, 
            len(vehicles), plates_found, processing_time,
            avg_conf=avg_conf
        )
        
        return frame

    def run_on_dataset(self, processed_dataset_path, limit=None):
        """Percorre as pastas diurno/noturno do dataset já organizado."""
        base_path = Path(processed_dataset_path)
        scenarios = ['diurno', 'noturno', 'baixa_qualidade']
        
        for scenario in scenarios:
            scenario_path = base_path / scenario
            if not scenario_path.exists(): continue
            
            print(f"\nProcessando cenário: {scenario}")
            files = list(scenario_path.glob("*.jpg"))
            
            # Aplica o limite se especificado
            if limit:
                files = files[:limit]
            
            for file_path in tqdm(files, desc=f"Lendo {scenario}"):
                frame = cv2.imread(str(file_path))
                if frame is not None:
                    self.process_frame(frame, file_path.name, scenario)
                    
        self.logger.save_results()
