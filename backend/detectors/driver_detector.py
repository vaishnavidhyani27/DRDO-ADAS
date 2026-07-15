import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)


LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
MOUTH = (61, 291, 13, 14)
POSE_POINTS = (1, 199, 33, 263, 61, 291)

MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)

AXIS_CORRECTION = np.diag([1.0, -1.0, -1.0])


class DriverDetector:
    def __init__(self):
        backend_dir = Path(__file__).resolve().parents[1]
        model_path = backend_dir / "models" / "face_landmarker.task"

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Face Landmarker model not found: {model_path}"
            )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self.landmarker = FaceLandmarker.create_from_options(options)

        self.start_time = time.monotonic()
        self.last_timestamp = -1

        self.eye_closed_since = None
        self.distracted_since = None
        self.yawn_since = None

        self.previous_pose = None

        print("MediaPipe driver detector loaded")

    @staticmethod
    def distance(point_a, point_b):
        return float(np.linalg.norm(point_a - point_b))

    def eye_aspect_ratio(self, points, indices):
        p1, p2, p3, p4, p5, p6 = [
            points[index] for index in indices
        ]

        horizontal = self.distance(p1, p4)

        if horizontal <= 1e-6:
            return 0.0

        return (
            self.distance(p2, p6)
            + self.distance(p3, p5)
        ) / (2.0 * horizontal)

    def mouth_aspect_ratio(self, points):
        left, right, upper, lower = [
            points[index] for index in MOUTH
        ]

        horizontal = self.distance(left, right)

        if horizontal <= 1e-6:
            return 0.0

        return self.distance(upper, lower) / horizontal

    def next_timestamp(self):
        timestamp = int(
            (time.monotonic() - self.start_time) * 1000
        )

        if timestamp <= self.last_timestamp:
            timestamp = self.last_timestamp + 1

        self.last_timestamp = timestamp
        return timestamp

    def estimate_head_pose(self, points, width, height):
        image_points = np.array(
            [points[index] for index in POSE_POINTS],
            dtype=np.float64,
        )

        focal_length = float(width)

        camera_matrix = np.array(
            [
                [focal_length, 0.0, width / 2.0],
                [0.0, focal_length, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        success, rotation_vector, _ = cv2.solvePnP(
            MODEL_POINTS,
            image_points,
            camera_matrix,
            np.zeros((4, 1), dtype=np.float64),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        corrected = AXIS_CORRECTION @ rotation_matrix

        sy = np.sqrt(
            corrected[0, 0] ** 2
            + corrected[1, 0] ** 2
        )

        if sy >= 1e-6:
            pitch = np.arctan2(
                corrected[2, 1],
                corrected[2, 2],
            )
            yaw = np.arctan2(
                -corrected[2, 0],
                sy,
            )
            roll = np.arctan2(
                corrected[1, 0],
                corrected[0, 0],
            )
        else:
            pitch = np.arctan2(
                -corrected[1, 2],
                corrected[1, 1],
            )
            yaw = np.arctan2(
                -corrected[2, 0],
                sy,
            )
            roll = 0.0

        pose = {
            "yaw": -float(np.degrees(yaw)),
            "pitch": float(np.degrees(pitch)),
            "roll": float(np.degrees(roll)),
        }

        if self.previous_pose is not None:
            alpha = 0.65

            for key in pose:
                difference = (
                    pose[key] - self.previous_pose[key]
                )

                if abs(difference) > 45:
                    pose[key] = self.previous_pose[key]
                else:
                    pose[key] = (
                        alpha * pose[key]
                        + (1 - alpha)
                        * self.previous_pose[key]
                    )

        self.previous_pose = pose

        return {
            key: round(value, 1)
            for key, value in pose.items()
        }

    @staticmethod
    def classify_direction(head_pose):
        if head_pose is None:
            return "Unknown"

        yaw = head_pose["yaw"]
        pitch = head_pose["pitch"]

        if yaw <= -20:
            return "Looking Left"

        if yaw >= 20:
            return "Looking Right"

        if pitch >= 18:
            return "Looking Down"

        if pitch <= -15:
            return "Looking Up"

        return "Forward"

    def no_driver_result(self):
        self.eye_closed_since = None
        self.distracted_since = None
        self.yawn_since = None
        self.previous_pose = None

        return {
            "face_detected": False,
            "driver_status": "No Driver",
            "direction": "Unknown",
            "drowsy": False,
            "eyes_closed": False,
            "yawning": False,
            "ear": None,
            "mar": None,
            "head_pose": None,
            "alert": "No driver detected",
        }

    def detect(self, frame):
        if frame is None or frame.size == 0:
            raise ValueError("Empty driver frame")

        height, width = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)

        media_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        result = self.landmarker.detect_for_video(
            media_image,
            self.next_timestamp(),
        )

        if not result.face_landmarks:
            return self.no_driver_result()

        landmarks = result.face_landmarks[0]

        points = np.array(
            [
                [
                    landmark.x * width,
                    landmark.y * height,
                ]
                for landmark in landmarks
            ],
            dtype=np.float64,
        )

        left_ear = self.eye_aspect_ratio(
            points,
            LEFT_EYE,
        )

        right_ear = self.eye_aspect_ratio(
            points,
            RIGHT_EYE,
        )

        ear = (left_ear + right_ear) / 2.0
        mar = self.mouth_aspect_ratio(points)

        head_pose = self.estimate_head_pose(
            points,
            width,
            height,
        )

        direction = self.classify_direction(head_pose)
        now = time.monotonic()

        eyes_closed = ear < 0.20
        mouth_open = mar > 0.55

        if eyes_closed:
            if self.eye_closed_since is None:
                self.eye_closed_since = now
        else:
            self.eye_closed_since = None

        eye_closure_duration = (
            now - self.eye_closed_since
            if self.eye_closed_since is not None
            else 0.0
        )

        drowsy = eye_closure_duration >= 1.5

        if mouth_open:
            if self.yawn_since is None:
                self.yawn_since = now
        else:
            self.yawn_since = None

        yawn_duration = (
            now - self.yawn_since
            if self.yawn_since is not None
            else 0.0
        )

        yawning = yawn_duration >= 1.0

        distracted = direction not in (
            "Forward",
            "Unknown",
        )

        if distracted:
            if self.distracted_since is None:
                self.distracted_since = now
        else:
            self.distracted_since = None

        distraction_duration = (
            now - self.distracted_since
            if self.distracted_since is not None
            else 0.0
        )

        prolonged_distraction = (
            distraction_duration >= 2.0
        )

        if drowsy:
            status = "Drowsy"
            alert = "Warning. Driver appears drowsy"

        elif yawning:
            status = "Drowsy"
            alert = "Warning. Driver is yawning"

        elif prolonged_distraction:
            status = "Distracted"
            alert = (
                f"Warning. Driver is "
                f"{direction.lower()}"
            )

        else:
            status = "Attentive"
            alert = "No Alert"

        return {
            "face_detected": True,
            "driver_status": status,
            "direction": direction,
            "drowsy": drowsy,
            "eyes_closed": eyes_closed,
            "eye_closure_duration": round(
                eye_closure_duration,
                2,
            ),
            "yawning": yawning,
            "ear": round(ear, 3),
            "mar": round(mar, 3),
            "head_pose": head_pose,
            "alert": alert,
        }

    def close(self):
        self.landmarker.close()