from flask import Flask, jsonify, request
from flask_cors import CORS
import base64
import os
import cv2
import numpy as np

from detectors.vehicle_detector import VehicleDetector
from detectors.pothole_detector import PotholeDetector
from detectors.lane_detector import LaneDetector
from detectors.driver_detector import DriverDetector

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
vehicle_detector = VehicleDetector(os.path.join(BASE_DIR, "models", "yolov8n.pt"))
pothole_detector = PotholeDetector(os.path.join(BASE_DIR, "models", "best.pt"))
lane_detector = LaneDetector()
driver_detector = DriverDetector()
previous_vehicles = []


def decode_image(data):
    if not isinstance(data, str) or not data:
        raise ValueError("Image data is missing or invalid")
    try:
        encoded = data.split(",", 1)[1] if "," in data else data
        frame = cv2.imdecode(
            np.frombuffer(base64.b64decode(encoded), np.uint8),
            cv2.IMREAD_COLOR,
        )
    except Exception as error:
        raise ValueError("Invalid base64 image data") from error
    if frame is None:
        raise ValueError("Unable to decode image")
    return frame


def get_frame():
    data = request.get_json(silent=True) or {}
    if "image" not in data:
        raise ValueError("No image received")
    return decode_image(data["image"])


def detect_wrong_way(detections, lane_polygon):
    global previous_vehicles

    if len(lane_polygon or []) < 4:
        previous_vehicles = []
        return {"wrong_way": False, "wrong_way_vehicles": []}

    polygon = np.asarray(lane_polygon, np.int32)
    allowed = {"Bicycle", "Car", "Motorcycle", "Bus", "Truck"}
    current, wrong = [], []

    for item in detections:
        if item.get("class") not in allowed:
            continue

        x1, y1, x2, y2 = item["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        area = max(1, (x2 - x1) * (y2 - y1))
        current_item = {
            "class": item["class"],
            "bbox": item["bbox"],
            "cx": cx,
            "cy": cy,
            "area": area,
        }
        current.append(current_item)

        inside = cv2.pointPolygonTest(
            polygon, (int(cx), int(y2)), False
        ) >= 0
        if not inside:
            continue

        matches = [old for old in previous_vehicles if old["class"] == item["class"]]
        if not matches:
            continue

        match = min(
            matches,
            key=lambda old: (old["cx"] - cx) ** 2 + (old["cy"] - cy) ** 2,
        )
        movement = ((match["cx"] - cx) ** 2 + (match["cy"] - cy) ** 2) ** 0.5
        growth = (area - match["area"]) / max(match["area"], 1)

        if movement <= 140 and growth > 0.12 and area > 5000:
            wrong.append({
                "class": item["class"],
                "bbox": item["bbox"],
                "area_growth": round(growth, 2),
            })

    previous_vehicles = current
    return {"wrong_way": bool(wrong), "wrong_way_vehicles": wrong}


@app.get("/")
def home():
    return jsonify({
        "service": "DRDO Integrated ADAS API",
        "status": "running",
        "vehicle_model": "loaded",
        "pothole_model": "loaded",
        "lane_model": "loaded",
        "driver_model": "loaded",
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "vehicle_detector": "ready",
        "pothole_detector": "ready",
        "lane_detector": "ready",
        "driver_detector": "ready",
    })


@app.route("/detect", methods=["POST", "OPTIONS"])
def detect():
    if request.method == "OPTIONS":
        return "", 204

    try:
        frame = get_frame()
        vehicles = vehicle_detector.detect(frame)
        potholes = pothole_detector.detect(frame)
        lanes = lane_detector.detect(frame)

        detections = vehicles["detections"] + potholes["detections"]
        wrong = detect_wrong_way(
            vehicles["detections"], lanes.get("lane_polygon", [])
        )

        vehicle_count = vehicles.get("vehicle_count", 0)
        pedestrian_count = vehicles.get("pedestrian_count", 0)
        pothole_count = potholes.get("pothole_count", 0)
        nearest = vehicles.get("nearest_distance_m")
        departure = lanes.get("lane_departure", False)

        if wrong["wrong_way"]:
            alert = "Warning. Wrong way vehicle ahead"
        elif departure:
            direction = lanes.get("departure_direction") or ""
            alert = f"Warning. Lane departure {direction}".strip()
        elif pothole_count:
            alert = "Warning. Pothole ahead"
        elif pedestrian_count:
            alert = "Pedestrian detected"
        elif nearest is not None and nearest <= 4:
            alert = "Vehicle too close"
        elif vehicle_count >= 3:
            alert = "Traffic ahead"
        else:
            alert = "No Alert"

        return jsonify({
            "detections": detections,
            "vehicle_count": vehicle_count,
            "pedestrian_count": pedestrian_count,
            "pothole_count": pothole_count,
            "nearest_vehicle_distance_m": nearest,
            "nearest_distance_m": nearest,
            "left_lane": lanes.get("left_lane", []),
            "right_lane": lanes.get("right_lane", []),
            "lane_polygon": lanes.get("lane_polygon", []),
            "lane_detected": lanes.get("lane_detected", False),
            "lane_status": lanes.get("lane_status", "Lane Not Detected"),
            "lane_departure": departure,
            "departure_direction": lanes.get("departure_direction"),
            "lane_center": lanes.get("lane_center"),
            "lane_offset_ratio": lanes.get("offset_ratio"),
            **wrong,
            "alert": alert,
        })

    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        app.logger.exception("Road detection failed")
        return jsonify({"error": "Detection failed", "details": str(error)}), 500


@app.route("/detect-driver", methods=["POST", "OPTIONS"])
def detect_driver():
    if request.method == "OPTIONS":
        return "", 204

    try:
        return jsonify(driver_detector.detect(get_frame()))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        app.logger.exception("Driver detection failed")
        return jsonify({
            "error": "Driver detection failed",
            "details": str(error),
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)