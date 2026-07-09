from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)
CORS(app)

vehicle_model = YOLO("models/yolov8n.pt")
pothole_model = YOLO("models/best.pt")

TARGET_CLASSES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


def decode_image(image_data):
    image_data = image_data.split(",")[1]
    image_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


def estimate_distance(box_height):
    if box_height <= 0:
        return "N/A"
    distance = round(1200 / box_height, 1)
    return f"{distance} m"


@app.route("/")
def home():
    return "YOLOv8 ADAS Backend Running"


@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({"error": "No image received"}), 400

    frame = decode_image(data["image"])

    detections = []
    vehicle_count = 0
    pedestrian_count = 0
    pothole_count = 0
    nearest_distance = "N/A"

    vehicle_results = vehicle_model.predict(
        source=frame,
        conf=0.25,
        iou=0.45,
        imgsz=640,
        verbose=False
    )

    for result in vehicle_results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = TARGET_CLASSES[cls]
            box_height = y2 - y1
            distance = estimate_distance(box_height)

            if label == "Person":
                pedestrian_count += 1
            else:
                vehicle_count += 1

            nearest_distance = distance

            detections.append({
                "class": label,
                "confidence": round(conf, 2),
                "bbox": [x1, y1, x2, y2],
                "distance": distance,
                "type": "vehicle"
            })

    pothole_results = pothole_model.predict(
        source=frame,
        conf=0.25,
        iou=0.45,
        imgsz=640,
        verbose=False
    )

    for result in pothole_results:
        for box in result.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            pothole_count += 1

            detections.append({
                "class": "Pothole",
                "confidence": round(conf, 2),
                "bbox": [x1, y1, x2, y2],
                "distance": "Road Surface",
                "type": "pothole"
            })

    if pedestrian_count > 0:
        alert = "Pedestrian Detected"
    elif pothole_count > 0:
        alert = "Pothole Detected"
    elif vehicle_count >= 3:
        alert = "Traffic Ahead"
    else:
        alert = "No Alert"

    return jsonify({
        "detections": detections,
        "vehicle": f"{vehicle_count} Vehicles",
        "pedestrian": f"{pedestrian_count} Pedestrians",
        "pothole": f"{pothole_count} Potholes",
        "distance": nearest_distance,
        "lane": "Safe",
        "driver": "Awake",
        "phone": "Not Detected",
        "alert": alert
    })


if __name__ == "__main__":
    app.run(debug=True)