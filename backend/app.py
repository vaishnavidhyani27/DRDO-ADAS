from flask import Flask, jsonify, request
from flask_cors import CORS
import base64
import cv2
import numpy as np
import os

from detectors.vehicle_detector import VehicleDetector
from detectors.pothole_detector import PotholeDetector


app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    allow_headers=["Content-Type"],
    methods=["GET", "POST", "OPTIONS"],
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VEHICLE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "yolov8n.pt",
)

POTHOLE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best.pt",
)


vehicle_detector = VehicleDetector(VEHICLE_MODEL_PATH)
pothole_detector = PotholeDetector(POTHOLE_MODEL_PATH)


def decode_image(image_data):
    if not isinstance(image_data, str) or not image_data:
        raise ValueError("Image data is missing or invalid")

    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_data)
    except Exception as error:
        raise ValueError("Invalid base64 image data") from error

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Unable to decode image")

    return frame


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
    return jsonify(
        {
            "status": "healthy",
            "vehicle_detector": "ready",
            "pothole_detector": "ready",
        }
    )


@app.route("/detect", methods=["POST", "OPTIONS"])
def detect():
    if request.method == "OPTIONS":
        return "", 204

    try:
        data = request.get_json(silent=True)

        if not data or "image" not in data:
            return jsonify({"error": "No image received"}), 400

        frame = decode_image(data["image"])

        vehicle_result = vehicle_detector.detect(frame)
        pothole_result = pothole_detector.detect(frame)

        detections = (
            vehicle_result["detections"]
            + pothole_result["detections"]
        )

        vehicle_count = vehicle_result["vehicle_count"]
        pedestrian_count = vehicle_result["pedestrian_count"]
        nearest_vehicle_distance = vehicle_result["nearest_distance_m"]

        pothole_count = pothole_result["pothole_count"]
        nearest_pothole_distance = pothole_result[
            "nearest_pothole_distance_m"
        ]

        if (
            nearest_pothole_distance is not None
            and nearest_pothole_distance <= 4
        ):
            alert = (
                f"Warning. Pothole detected at "
                f"{nearest_pothole_distance} meters"
            )

        elif nearest_pothole_distance is not None:
            alert = (
                f"Pothole detected at "
                f"{nearest_pothole_distance} meters"
            )

        elif pedestrian_count > 0:
            alert = "Pedestrian Detected"

        elif (
            nearest_vehicle_distance is not None
            and nearest_vehicle_distance <= 4
        ):
            alert = "Vehicle Too Close"

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
                "nearest_vehicle_distance_m": nearest_vehicle_distance,
                "nearest_pothole_distance_m": nearest_pothole_distance,
                "nearest_distance_m": nearest_vehicle_distance,
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
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )