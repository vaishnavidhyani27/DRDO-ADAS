from flask import Flask, jsonify, Response
from flask_cors import CORS
import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from detectors.vehicle_detector import VehicleDetector

app = Flask(__name__)
CORS(app)

detector = VehicleDetector()
camera = cv2.VideoCapture(0)

latest_data = {
    "vehicle": "0 Vehicles",
    "distance": "Calculating...",
    "pedestrian": "Not Detected",
    "lane": "Safe",
    "pothole": "Not Detected",
    "driver": "Monitoring",
    "phone": "Not Detected",
    "alert": "No Alert"
}


def generate_frames():
    global latest_data

    while True:
        success, frame = camera.read()

        if not success:
            break

        frame, detections = detector.detect(frame)

        vehicle_count = 0
        pedestrian_detected = False

        for d in detections:
            if d["class"] == "Person":
                pedestrian_detected = True
            else:
                vehicle_count += 1

        latest_data = {
            "vehicle": f"{vehicle_count} Vehicles",
            "distance": "Approx. 5 m",
            "pedestrian": "Detected" if pedestrian_detected else "Not Detected",
            "lane": "Safe",
            "pothole": "Not Detected",
            "driver": "Monitoring",
            "phone": "Not Detected",
            "alert": "No Alert" if vehicle_count < 3 else "Traffic Ahead"
        }

        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/")
def home():
    return "ADAS Flask Backend Running"


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    return jsonify(latest_data)


if __name__ == "__main__":
    app.run(debug=True)