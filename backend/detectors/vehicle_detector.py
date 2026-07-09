from ultralytics import YOLO
import cv2


class VehicleDetector:
    def __init__(self):
        self.model = YOLO("models/yolov8n.pt")

        self.target_classes = {
            0: "Person",
            1: "Bicycle",
            2: "Car",
            3: "Motorcycle",
            5: "Bus",
            7: "Truck"
        }

    def detect(self, frame):
        detections = []

        results = self.model.predict(
            source=frame,
            conf=0.25,
            iou=0.45,
            imgsz=640,
            device="cpu",
            verbose=False
        )

        for result in results:
            for box in result.boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls not in self.target_classes:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                label = self.target_classes[cls]

                detections.append({
                    "class": label,
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2]
                })

                color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                text = f"{label} {conf:.2f}"

                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        return frame, detections