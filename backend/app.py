from flask import Flask, jsonify, request
from flask_cors import CORS

import base64
import os

import cv2
import numpy as np

from detectors.vehicle_detector import VehicleDetector
from detectors.pothole_detector import PotholeDetector
from detectors.lane_detector import LaneDetector


app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    allow_headers=["Content-Type"],
    methods=["GET", "POST", "OPTIONS"],
)


# --------------------------------------------------
# Model paths and detector initialization
# --------------------------------------------------

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

vehicle_detector = VehicleDetector(
    VEHICLE_MODEL_PATH
)

pothole_detector = PotholeDetector(
    POTHOLE_MODEL_PATH
)

lane_detector = LaneDetector()


# Previous vehicle information for basic tracking.
previous_vehicles = []


# --------------------------------------------------
# Image decoding
# --------------------------------------------------

def decode_image(image_data):
    if not isinstance(image_data, str) or not image_data:
        raise ValueError(
            "Image data is missing or invalid"
        )

    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(
            image_data
        )
    except Exception as error:
        raise ValueError(
            "Invalid base64 image data"
        ) from error

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if frame is None:
        raise ValueError(
            "Unable to decode image"
        )

    return frame


# --------------------------------------------------
# Wrong-way vehicle detection
# --------------------------------------------------

def box_center_and_area(box):
    x1, y1, x2, y2 = box

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    area = max(1, (x2 - x1) * (y2 - y1))

    return center_x, center_y, area


def point_inside_lane(point, polygon):
    if not polygon or len(polygon) < 4:
        return False

    polygon_array = np.asarray(
        polygon,
        dtype=np.int32,
    )

    return (
        cv2.pointPolygonTest(
            polygon_array,
            point,
            False,
        )
        >= 0
    )


def detect_wrong_way(
    detections,
    lane_polygon,
):
    """
    Prototype wrong-way logic:

    1. Vehicle must lie inside the current lane.
    2. It must match a vehicle from the previous frame.
    3. Its bounding-box area must grow noticeably,
       indicating that it is approaching the camera.
    """

    global previous_vehicles

    vehicle_classes = {
        "Bicycle",
        "Car",
        "Motorcycle",
        "Bus",
        "Truck",
    }

    current_vehicles = []
    wrong_way_detections = []

    for detection in detections:
        if detection["class"] not in vehicle_classes:
            continue

        box = detection["bbox"]
        center_x, center_y, area = (
            box_center_and_area(box)
        )

        # Use the bottom centre because it represents
        # where the vehicle touches the road.
        road_point = (
            int(center_x),
            int(box[3]),
        )

        inside_lane = point_inside_lane(
            road_point,
            lane_polygon,
        )

        current_vehicle = {
            "class": detection["class"],
            "bbox": box,
            "center_x": center_x,
            "center_y": center_y,
            "area": area,
            "inside_lane": inside_lane,
        }

        current_vehicles.append(
            current_vehicle
        )

        if not inside_lane:
            continue

        best_match = None
        best_distance = float("inf")

        for previous in previous_vehicles:
            if (
                previous["class"]
                != current_vehicle["class"]
            ):
                continue

            movement_distance = (
                (
                    previous["center_x"]
                    - center_x
                )
                ** 2
                + (
                    previous["center_y"]
                    - center_y
                )
                ** 2
            ) ** 0.5

            if movement_distance < best_distance:
                best_distance = movement_distance
                best_match = previous

        if best_match is None:
            continue

        # Allow matching within a moderate image distance.
        if best_distance > 140:
            continue

        previous_area = best_match["area"]

        area_growth = (
            area - previous_area
        ) / max(previous_area, 1)

        # Growing by more than 12% suggests approach.
        approaching = area_growth > 0.12

        # Require a reasonably large vehicle to reduce
        # distant false alarms.
        large_enough = area > 5000

        if approaching and large_enough:
            wrong_way_detections.append(
                {
                    "class": detection["class"],
                    "bbox": box,
                    "area_growth": round(
                        area_growth,
                        2,
                    ),
                }
            )

    previous_vehicles = current_vehicles

    return {
        "wrong_way": (
            len(wrong_way_detections) > 0
        ),
        "wrong_way_vehicles": (
            wrong_way_detections
        ),
    }


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "service": "DRDO Integrated ADAS API",
            "status": "running",
            "vehicle_model": "loaded",
            "pothole_model": "loaded",
            "lane_model": "loaded",
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "vehicle_detector": "ready",
            "pothole_detector": "ready",
            "lane_detector": "ready",
        }
    )


@app.route(
    "/detect",
    methods=["POST", "OPTIONS"],
)
def detect():
    if request.method == "OPTIONS":
        return "", 204

    try:
        data = request.get_json(
            silent=True
        )

        if not data or "image" not in data:
            return jsonify(
                {"error": "No image received"}
            ), 400

        frame = decode_image(
            data["image"]
        )

        vehicle_result = (
            vehicle_detector.detect(frame)
        )

        pothole_result = (
            pothole_detector.detect(frame)
        )

        lane_result = (
            lane_detector.detect(frame)
        )

        detections = (
            vehicle_result["detections"]
            + pothole_result["detections"]
        )

        wrong_way_result = detect_wrong_way(
            vehicle_result["detections"],
            lane_result["lane_polygon"],
        )

        vehicle_count = (
            vehicle_result["vehicle_count"]
        )

        pedestrian_count = (
            vehicle_result[
                "pedestrian_count"
            ]
        )

        nearest_vehicle_distance = (
            vehicle_result[
                "nearest_distance_m"
            ]
        )

        pothole_count = (
            pothole_result[
                "pothole_count"
            ]
        )

        lane_departure = (
            lane_result[
                "lane_departure"
            ]
        )

        wrong_way = (
            wrong_way_result[
                "wrong_way"
            ]
        )

        # Alert priority.
        if wrong_way:
            alert = (
                "Warning. Wrong way vehicle ahead"
            )

        elif lane_departure:
            direction = (
                lane_result[
                    "departure_direction"
                ]
                or ""
            )

            alert = (
                f"Warning. Lane departure "
                f"{direction}"
            ).strip()

        elif pothole_count > 0:
            alert = "Warning. Pothole ahead"

        elif pedestrian_count > 0:
            alert = "Pedestrian detected"

        elif (
            nearest_vehicle_distance
            is not None
            and nearest_vehicle_distance <= 4
        ):
            alert = "Vehicle too close"

        elif vehicle_count >= 3:
            alert = "Traffic ahead"

        else:
            alert = "No Alert"

        return jsonify(
            {
                "detections": detections,

                "vehicle_count": vehicle_count,
                "pedestrian_count": pedestrian_count,
                "pothole_count": pothole_count,

                "nearest_vehicle_distance_m": (
                    nearest_vehicle_distance
                ),
                "nearest_distance_m": (
                    nearest_vehicle_distance
                ),

                "left_lane": (
                    lane_result[
                        "left_lane"
                    ]
                ),
                "right_lane": (
                    lane_result[
                        "right_lane"
                    ]
                ),
                "lane_polygon": (
                    lane_result[
                        "lane_polygon"
                    ]
                ),
                "lane_detected": (
                    lane_result[
                        "lane_detected"
                    ]
                ),
                "lane_status": (
                    lane_result[
                        "lane_status"
                    ]
                ),
                "lane_departure": (
                    lane_departure
                ),
                "departure_direction": (
                    lane_result[
                        "departure_direction"
                    ]
                ),
                "lane_center": (
                    lane_result[
                        "lane_center"
                    ]
                ),
                "lane_offset_ratio": (
                    lane_result[
                        "offset_ratio"
                    ]
                ),

                "wrong_way": wrong_way,
                "wrong_way_vehicles": (
                    wrong_way_result[
                        "wrong_way_vehicles"
                    ]
                ),

                "alert": alert,
            }
        )

    except ValueError as error:
        return jsonify(
            {"error": str(error)}
        ), 400

    except Exception as error:
        app.logger.exception(
            "Detection failed"
        )

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