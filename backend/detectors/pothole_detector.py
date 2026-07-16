from ultralytics import YOLO


class PotholeDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model.predict(
            source=frame,
            conf=0.55,
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
                frame_height, frame_width = frame.shape[:2]

                box_center_y = (y1 + y2) / 2
                box_center_x = (x1 + x2) / 2

                # Potholes must appear in the lower road region.
                if box_center_y < frame_height * 0.48:
                    continue

                # Ignore extreme left and right roadside regions.
                if (
                box_center_x < frame_width * 0.08
                or box_center_x > frame_width * 0.92
                ):
                    continue

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