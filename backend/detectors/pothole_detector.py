from ultralytics import YOLO


class PotholeDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    @staticmethod
    def estimate_pothole_distance(y_bottom, frame_height):
        """
        Estimate pothole distance using road perspective.

        Potholes near the bottom of the frame are treated as closer.
        Potholes near the road horizon are treated as farther away.

        This is an approximate prototype estimate and should later
        be calibrated using the actual phone camera.
        """

        if frame_height <= 0:
            return None

        # Approximate road horizon at 45% of the image height.
        horizon_y = frame_height * 0.45

        if y_bottom <= horizon_y:
            return 20.0

        road_height = frame_height - horizon_y
        relative_position = (y_bottom - horizon_y) / road_height

        relative_position = max(0.0, min(relative_position, 1.0))

        # Approximate range:
        # near horizon -> around 20 m
        # near bottom  -> around 2 m
        distance_m = 2.0 + 18.0 * ((1.0 - relative_position) ** 1.6)

        return round(distance_m, 1)

    def detect(self, frame):
        frame_height, _ = frame.shape[:2]

        results = self.model.predict(
            source=frame,
            conf=0.35,
            iou=0.45,
            imgsz=640,
            device="cpu",
            verbose=False,
        )

        detections = []
        distances = []

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                distance_m = self.estimate_pothole_distance(
                    y_bottom=y2,
                    frame_height=frame_height,
                )

                if distance_m is not None:
                    distances.append(distance_m)

                detections.append(
                    {
                        "class": "Pothole",
                        "confidence": round(confidence, 2),
                        "bbox": [x1, y1, x2, y2],
                        "distance_m": distance_m,
                        "type": "pothole",
                    }
                )

        nearest_pothole_distance = min(distances) if distances else None

        return {
            "detections": detections,
            "pothole_count": len(detections),
            "nearest_pothole_distance_m": nearest_pothole_distance,
        }