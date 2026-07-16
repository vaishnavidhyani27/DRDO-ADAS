from ultralytics import YOLO


class PotholeDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_height * frame_width

        results = self.model.predict(
            source=frame,
            conf=0.65,
            iou=0.35,
            imgsz=960,
            device="cpu",
            verbose=False,
        )

        detections = []

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                box_width = max(1, x2 - x1)
                box_height = max(1, y2 - y1)
                box_area = box_width * box_height

                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2

                area_ratio = box_area / frame_area
                aspect_ratio = box_width / box_height

                # 1. Potholes must be in the lower road region.
                if box_center_y < frame_height * 0.58:
                    continue

                # 2. Ignore extreme roadside regions.
                if (
                    box_center_x < frame_width * 0.12
                    or box_center_x > frame_width * 0.88
                ):
                    continue

                # 3. Real potholes are usually wider than tall.
                if aspect_ratio < 1.15:
                    continue

                # 4. Reject tiny noise detections.
                if area_ratio < 0.0008:
                    continue

                # 5. Reject very large detections such as cars,
                # buildings, road regions, or shadows.
                if area_ratio > 0.10:
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