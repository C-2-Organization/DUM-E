# yolo_detector.py
from ultralytics import YOLO
import torch

class YOLODetector:
    def __init__(self, model_path, device: str | None = None):
        """
        model_path 예: '/home/.../models/yolov8s-worldv2.pt'
        device:
          - None: 자동 선택 (cuda 가능하면 cuda, 아니면 cpu)
          - "cpu" / "cuda" 강제 가능
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = YOLO(model_path)
        self.model.to(device)  # 여기서 cuda 없으면 죽었었음 :contentReference[oaicite:3]{index=3}

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
                "bbox_xyxy_px": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(conf.cpu().numpy()),
                "class_id": cls_id,
                "class_name": names[cls_id],
            })

        return detections
