import torch
import cv2
from abc import ABC, abstractmethod
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssdlite320_mobilenet_v3_large
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, SSDLite320_MobileNet_V3_Large_Weights

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame):
        """Retorna lista de detecções: [{'bbox': [x1, y1, x2, y2], 'conf': 0.9, 'class': 'car'}]"""
        pass

class YOLODetector(BaseDetector):
    def __init__(self, model_path='models/yolov8n.pt'):
        self.model = YOLO(model_path)
        self.model_name = "YOLOv8"
        
    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            detections.append({
                'bbox': box.xyxy[0].tolist(),
                'conf': float(box.conf),
                'class': self.model.names[int(box.cls)]
            })
        return detections

class TorchvisionDetector(BaseDetector):
    def __init__(self, model_type='faster_rcnn', confidence_threshold=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = confidence_threshold
        self.model_name = "Faster R-CNN" if model_type == 'faster_rcnn' else "SSD"
        
        if model_type == 'faster_rcnn':
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn(weights=weights).to(self.device)
            self.classes = weights.meta["categories"]
        else: # SSD
            weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            self.model = ssdlite320_mobilenet_v3_large(weights=weights).to(self.device)
            self.classes = weights.meta["categories"]
            
        self.model.eval()

    def detect(self, frame):
        # Preprocess
        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(img_tensor)[0]
        
        detections = []
        for i in range(len(prediction['boxes'])):
            score = float(prediction['scores'][i])
            if score > self.threshold:
                detections.append({
                    'bbox': prediction['boxes'][i].tolist(),
                    'conf': score,
                    'class': self.classes[int(prediction['labels'][i])]
                })
        return detections

class PlateDetector(BaseDetector):
    def __init__(self, model_path='models/yolov8n-plate.pt'):
        try:
            # Tenta carregar o modelo YOLO especializado
            self.model = YOLO(model_path)
            self.has_model = True
        except Exception as e:
            print(f"Aviso: Modelo YOLO de placas não encontrado. Usando fallback OpenCV.")
            self.has_model = False
        self.model_name = "PlateDetector"
        
    def detect(self, frame):
        if self.has_model:
            results = self.model(frame, verbose=False)[0]
            detections = []
            for box in results.boxes:
                detections.append({
                    'bbox': box.xyxy[0].tolist(),
                    'conf': float(box.conf),
                    'class': 'plate'
                })
            return detections
        else:
            # --- FALLBACK: Visão Computacional Clássica ---
            # Ideal para TCC: Detecção baseada em bordas e contornos
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Filtro para reduzir ruído mantendo bordas
            bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
            # Detecção de bordas
            edged = cv2.Canny(bfilter, 30, 200)
            
            # Encontrar contornos
            keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(keypoints[0], key=cv2.contourArea, reverse=True)[:10]
            
            detections = []
            for res in contours:
                approx = cv2.approxPolyDP(res, 10, True)
                if len(approx) == 4: # Retângulos (possíveis placas)
                    x, y, w, h = cv2.boundingRect(res)
                    aspect_ratio = w / float(h)
                    # Placas brasileiras têm proporção de ~3:1 a ~4:1
                    if 2.0 < aspect_ratio < 5.0:
                        detections.append({
                            'bbox': [float(x), float(y), float(x+w), float(y+h)],
                            'conf': 0.8, # Confiança fixa para o fallback
                            'class': 'plate'
                        })
            return detections
