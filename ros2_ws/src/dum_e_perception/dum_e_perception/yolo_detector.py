# yolo_detector.py
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path, device: str = "cuda"):
        """
        model_path 예: '/home/.../models/yolov8s-worldv2.pt'
        """
        self.model = YOLO(model_path)
        self.model.to(device)

        try:
            clip_model = self.model.model.clip_model
            clip_model.to(device)
        except AttributeError:
            pass

    def set_classes(self, classes: list[str] | None):
        if classes is None:
            self.model.set_classes(None)
        else:
            classes = [str(c) for c in classes]
            self.model.set_classes(classes)

    def detect(self, image_bgr, classes=None, conf_threshold: float = 0.15):
        """
        image_bgr: np.ndarray(BGR)
        classes: optional, ['blue scissors'] 처럼 프롬프트 리스트
        return: [
            {
                'class_name': str,
                'confidence': float,
                'bbox': [x1, y1, x2, y2]   # normalized (0~1)
            },
        ]
        """

        if classes is not None:
            self.set_classes(classes)

        results = self.model(image_bgr, conf=conf_threshold, verbose=False)[0]

        detections = []
        h, w, _ = image_bgr.shape

        # YOLO-World에서는 results.names가 현재 set_classes에 맞춰져 있음
        names = results.names if hasattr(results, "names") else self.model.names

        for box, cls, conf in zip(results.boxes.xyxy,
                                  results.boxes.cls,
                                  results.boxes.conf):
            x1, y1, x2, y2 = box.cpu().numpy()
            cls_id = int(cls.cpu().numpy())

            detections.append({
                "bbox": [x1 / w, y1 / h, x2 / w, y2 / h],   # normalized
                "confidence": float(conf.cpu().numpy()),
                "class_id": cls_id,
                "class_name": names[cls_id],
            })

        return detections
