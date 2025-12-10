from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path):
        """
        model_path 예: '/home/.../models/yolov8s-worldv2.pt'
        """
        self.model = YOLO(model_path)

    def detect(self, image_bgr):
        """
        image_bgr: np.ndarray(BGR)
        return: [
            {
                'class_name': str,
                'confidence': float,
                'bbox': [x1, y1, x2, y2]   # normalized (0~1)
            },
        ]
        """

        results = self.model(image_bgr)[0]

        detections = []
        h, w, _ = image_bgr.shape

        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            x1, y1, x2, y2 = box.cpu().numpy()

            detections.append({
                "bbox": [x1 / w, y1 / h, x2 / w, y2 / h],   # normalized
                "confidence": float(conf.cpu().numpy()),
                "class_id": int(cls.cpu().numpy()),
                "class_name": self.model.names[int(cls.cpu().numpy())]
            })

        return detections
