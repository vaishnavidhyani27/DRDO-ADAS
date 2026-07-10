from ultralytics import YOLO
from utils.distance_estimator import estimate_distance


TARGET_CLASSES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}


class VehicleDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model.predict(
            source=frame,
            conf=0.25,
            iou=0.45,
            imgsz=512,
            device="cpu",
            verbose=False,
        )

        detections = []
        vehicle_count = 0
        pedestrian_count = 0
        distances = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if class_id not in TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = TARGET_CLASSES[class_id]

                box_height = y2 - y1
                distance_m = estimate_distance(label, box_height)

                if label == "Person":
                    pedestrian_count += 1
                else:
                    vehicle_count += 1

                if distance_m is not None:
                    distances.append(distance_m)

                detections.append({
                    "class": label,
                    "confidence": round(confidence, 2),
                    "bbox": [x1, y1, x2, y2],
                    "distance_m": distance_m,
                    "type": "road_object",
                })

        nearest_distance = min(distances) if distances else None

        return {
            "detections": detections,
            "vehicle_count": vehicle_count,
            "pedestrian_count": pedestrian_count,
            "nearest_distance_m": nearest_distance,
        }