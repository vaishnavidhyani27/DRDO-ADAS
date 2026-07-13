from ultralytics import YOLO


class PotholeDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model.predict(
            source=frame,
            conf=0.40,
            iou=0.40,
            imgsz=960,
            device="cpu",
            verbose=False,
        )

        detections = []

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append(
                    {
                        "class": "Pothole",
                        "confidence": round(confidence, 2),
                        "bbox": [x1, y1, x2, y2],
                        "type": "pothole",
                    }
                )

        return {
            "detections": detections,
            "pothole_count": len(detections),
        }