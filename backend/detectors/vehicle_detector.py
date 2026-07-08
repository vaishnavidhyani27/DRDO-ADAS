from ultralytics import YOLO
import cv2


class VehicleDetector:

    def __init__(self):
        # Load YOLO model only once
        self.model = YOLO("models/yolov8n.pt")

        # COCO vehicle classes
        self.vehicle_classes = {
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
            conf=0.5,
            verbose=False
        )

        for result in results:
            for box in result.boxes:

                cls = int(box.cls[0])

                if cls not in self.vehicle_classes:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                confidence = float(box.conf[0])

                vehicle_name = self.vehicle_classes[cls]

                detections.append({
                    "vehicle": vehicle_name,
                    "confidence": confidence,
                    "box": (x1, y1, x2, y2)
                })

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{vehicle_name} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        return frame, detections