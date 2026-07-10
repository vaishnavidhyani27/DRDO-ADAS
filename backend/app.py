from flask import Flask, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO
import base64
import cv2
import numpy as np
import os


app = Flask(__name__)
CORS(app)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VEHICLE_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")
POTHOLE_MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")


vehicle_model = YOLO(VEHICLE_MODEL_PATH)
pothole_model = YOLO(POTHOLE_MODEL_PATH)


TARGET_CLASSES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}


REAL_OBJECT_HEIGHTS_CM = {
    "Person": 170,
    "Bicycle": 110,
    "Car": 150,
    "Motorcycle": 120,
    "Bus": 300,
    "Truck": 300,
}


FOCAL_LENGTH_PX = 700


def decode_image(image_data):
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    image_bytes = base64.b64decode(image_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Invalid image data")

    return frame


def estimate_distance(label, box_height):
    if box_height <= 0:
        return None

    real_height_cm = REAL_OBJECT_HEIGHTS_CM.get(label)

    if real_height_cm is None:
        return None

    distance_cm = (real_height_cm * FOCAL_LENGTH_PX) / box_height
    distance_m = distance_cm / 100

    return round(distance_m, 1)


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "service": "DRDO ADAS YOLOv8 API",
            "status": "running",
            "vehicle_model": "loaded",
            "pothole_model": "loaded",
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json(silent=True)

        if not data or "image" not in data:
            return jsonify({"error": "No image received"}), 400

        frame = decode_image(data["image"])

        detections = []
        vehicle_count = 0
        pedestrian_count = 0
        pothole_count = 0
        detected_distances = []

        vehicle_results = vehicle_model.predict(
            source=frame,
            conf=0.25,
            iou=0.45,
            imgsz=512,
            device="cpu",
            verbose=False,
        )

        for result in vehicle_results:
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
                    detected_distances.append(distance_m)

                detections.append(
                    {
                        "class": label,
                        "confidence": round(confidence, 2),
                        "bbox": [x1, y1, x2, y2],
                        "distance_m": distance_m,
                        "type": "road_object",
                    }
                )

        pothole_results = pothole_model.predict(
            source=frame,
            conf=0.25,
            iou=0.45,
            imgsz=512,
            device="cpu",
            verbose=False,
        )

        for result in pothole_results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                pothole_count += 1

                detections.append(
                    {
                        "class": "Pothole",
                        "confidence": round(confidence, 2),
                        "bbox": [x1, y1, x2, y2],
                        "distance_m": None,
                        "type": "pothole",
                    }
                )

        nearest_distance = (
            min(detected_distances) if detected_distances else None
        )

        if pedestrian_count > 0:
            alert = "Pedestrian Detected"
        elif pothole_count > 0:
            alert = "Pothole Detected"
        elif nearest_distance is not None and nearest_distance <= 4:
            alert = "Object Too Close"
        elif vehicle_count >= 3:
            alert = "Traffic Ahead"
        else:
            alert = "No Alert"

        return jsonify(
            {
                "detections": detections,
                "vehicle_count": vehicle_count,
                "pedestrian_count": pedestrian_count,
                "pothole_count": pothole_count,
                "nearest_distance_m": nearest_distance,
                "alert": alert,
            }
        )

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except Exception as error:
        app.logger.exception("Detection failed")
        return jsonify(
            {
                "error": "Detection failed",
                "details": str(error),
            }
        ), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)